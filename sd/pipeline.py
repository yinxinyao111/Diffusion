import torch 
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

# text -> image, image -> image, main pipeline
# given info will generate picture
def generate(prompt: str, 
             uncond_prompt: str, # negative prompt or empty string
             input_image = None, # image to image
             strength = 0.8, # attention paid to input image during denoising
             do_cfg = True, # model outputs with & without the prompt
             cfg_scale = 7.5, 
             sampler_name = "ddpm", 
             n_inference_steps = 50, 
             models = {}, 
             seed = None, 
             device = None, 
             idle_device = None, 
             tokenizer = None):
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("strength must be between 0 and 1")
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
            
        # RNG
        generator = torch.Generator(device = device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        
        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            # convert prompt into tokens using tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding = "max_length", max_length = 77).input_ids
            # convert tokens to tensor (batch, seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype = torch.long, device = device)
            # run through clip
            # (batch, seq_len) -> (batch, seq_len, dmodel = 768)
            cond_context = clip(cond_tokens)
            
            # ---------------------------------------------
            
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding = "max_length", max_length = 77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype = torch.long, device = device)
            # (batch, seq_len, dmodel = 768)
            uncond_context = clip(uncond_tokens)
            
            # concat two prompts
            # (batch = 2, seq_len = 77, dmodel = 768)
            context = torch.cat([cond_context, uncond_context])
        else: 
            # convert it into a list of tokens
            tokens = tokenizer.batch_encode_plus([prompt], padding = "max_length", max_length = 77).input_ids
            tokens = torch.tensor(tokens, dtype = torch.long, device = device)
            # (batch = 1, seq_len = 77, dmodel = 768)
            context = clip(tokens)
        to_idle(clip)
        
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown Sampler")
        
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        
        # for image to image generation
        if input_image is not None:
            encoder = models["encoder"]
            encoder.to(device)
            
            input_image_tensor = input_image.convert("RGB").resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # (H, W, C = 3)
            input_image_tensor = torch.tensor(input_image_tensor, dtype = torch.float32, device = device)
            
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (batch, H, W, C)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # encoder wants (B, C, H, W)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
            
            encoder_noise = torch.randn(latents_shape, generator = generator, device = device)
            
            # run image through encoder of VAE
            latents = encoder(input_image_tensor, encoder_noise)
            # shifts the noise level from which we start from
            sampler.set_strength(strength = strength)
            # tell sampler to add noise according to strength
            latents = sampler.add_noise(latents, sampler.timesteps[0])
            
            to_idle(encoder)
        else: 
            # no image input (text to image generation), sample from random noise N(0, I)
            latents = torch.randn(latents_shape, generator = generator, device = device)
        
        # load U-NET
        diffusion = models["diffusion"]
        diffusion.to(device)
        timesteps = tqdm(sampler.timesteps)
        
        # denoising loop (UNET predicts noise, scheduler removes it)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)
            # (batch, 4, latents_height = 64, latents_width = 64)
            model_input = latents
            if do_cfg:
                # (batch, 4, latents_height, latents_width) -> (2 * batch, 4, latents_height, latents_width)
                # make 2 copies of latents, one with the prompt and one without
                model_input = model_input.repeat(2,1,1,1)
                # model output is the predicted noise by the U-NET
                model_output = diffusion(model_input, context, time_embedding)
            # combine conditional and unconditional outputs
            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                # combined output = w * (cond - uncond) + uncond
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            # remove noise predicted by the UNET
            latents = sampler.step(timestep, latents, model_output)
        to_idle(diffusion)
        
        decoder = models["decoder"]
        decoder.to(device)
        images = decoder(latents) # actually just 1 image
        
        images = rescale(images, (-1, 1), (0, 255), clamp = True)
        # (batch, channel, Height, Width) -> (batch, Height, Width, channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]

def rescale(x, old_range, new_range, clamp = False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # (160,)
    freqs = torch.pow(10000, -torch.arange(start = 0, end = 160, dtype = torch.float32) / 160)
    # (1. 160)
    x = torch.tensor([timestep], dtype = torch.float32)[:, None] * freqs
    # (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim = -1)
