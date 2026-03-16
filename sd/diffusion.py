import torch
from torch import nn
from torch.nn import functional as F 
from attention import SelfAttention, CrossAttention

# Auxillary for "Diffusion" class
class TimeEmbedding(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 320)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        
        # (1, 1280)
        return x

# Auxillary for "UNET" class
def SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                # computes cross attention between latent and prompt
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                # match latent with its time step
                x = layer(x, time)
            else:
                x = layer(x)
        return x

# Auxillary for "UNET" class
class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)
            
# Auxillary for "Diffusion" class
class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.Module([
            # given a list of layers, will apply them one by one
            
            # Encoder: reducing image size but gradually increasing features per pixel
            # (batch, 4, height/8, width/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size = 3, padding = 1)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            # (batch, 320, height/8, width/8) -> (batch, 320, height/16, height/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size = 3, stride = 2, padding = 1)), # this is the encoder side, reducing image size now
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)), # 8: num of heads, 80: embed size
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            # (batch, 640, height/16, width/16) -> (batch, 640, height/32, width/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size = 3, stride = 2, padding = 1)), # keep decreasing image size
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            # (batch, 1280, height/32, width/32) -> (batch, 1280, height/64, width/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size = 3, stride = 2, padding = 1)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            # (batch, 1280, height/64, width/64) -> (batch, 1280, height/64, width/64) residual connection does not change the size
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])
        
        # Bottleneck
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160), # cross attention
            UNET_ResidualBlock(1280 1280)
        )
        
        # Decoder
        self.decoders = nn.ModuleList([
            # (batch, 2560, height/64, width/64) -> (batch, 1280, height/64, width/64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)), # 2560 features: output of bottleneck (1280 features) + encoder output skip connection (1280)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)), # increases image size
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)), # (batch, 320, height/8, width/8)
        ])

# U-Net
class Diffusion(nn.Module):
    def __init__(self):
        # give the U-Net the noisified image & time step at which it was noisified
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent: (batch, 4, height/8, width/8)
        # context (prompt): (batch, seq_len, d_model = 768)
        # time: (1, 320)
        
        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time) 
        # (batch, 4, height/8, width/8) -> (batch, 320, height/8, width/8)
        output = self.unet(latent, context, time)
        # (batch, 320, height/8, width/8) -> (batch, 4, height/8, width/8)
        output = self.final(output)
        # latent (batch, 4, height/8, width/8)
        return output

