import torch
from torch import nn
from torch.nn import functional as F

# 2 blocks from decoder
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

# encoder is a sequence of models, inherit nn.Sequential
class VAE_Encoder(nn.Sequential):
    def __init__(self):
        # a sequence of models in which each model reduces data dimenison but increases num of features
        super().__init__(
            # (batch, channels = 3, height = 512, width = 512) -> (batch, 128, 512, 512)
            nn.Conv2d(3, 128, kernel_size = 3, padding = 1),
            
            # (batch, 128, height, width) -> (batch, 128, height, width)
            VAE_ResidualBlock(128, 128), # combination of convolutions + normalizations, does not change image size
            
            # (batch, 128, height, width) -> (batch, 128, height, width)
            VAE_ResidualBlock(128, 128),
            
            # (batch, 128, 512, 512) -> (batch, 128, 255, 255)
            nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 0),
            
            # (batch, 128, 255, 255) -> (batch, 256, 255, 255)
            VAE_ResidualBlock(128, 256),
            
            # (batch, 256, 255, 255) -> (batch, 256, 255, 255)
            VAE_ResidualBlock(256, 256),
            
            # (batch, 256, 255, 255) -> (batch, 256, 127, 127)
            nn.Conv2d(256, 256, kernel_size = 3, stride = 2, padding = 0),
            
            # (batch, 256, 127, 127) -> (batch, 512, 127, 127)
            VAE_ResidualBlock(256, 512),
            
            # (batch, 512, 127, 127) -> (batch, 512, 127, 127)
            VAE_ResidualBlock(512, 512),
            
            # (batch, 512, 127, 127) -> (batch, 512, 63, 63)
            nn.Conv2d(512, 512, kernal_size = 3, stride = 2, padding = 0),
            
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            # (batch, 512, 63, 63) -> (batch, 512, 63, 63)
            VAE_ResidualBlock(512, 512),
            
            # (batch, 512, 63, 63) -> (batch, 512, 63, 63)
            VAE_AttentionBlock(512), # self attention over each pixel
            
            # (batch, 512, 63, 63) -> (batch, 512, 63, 63)
            VAE_ResidualBlock(512, 512),
            
            # (batch, 512, 63, 63) -> (batch, 512, 63, 63)
            nn.GroupNorm(32, 512), # 32 groups, 512 channels (num of features)
            
            nn.SiLU(),
            
            # (batch, 512, 63, 63) -> (batch, 8, 63, 63)
            nn.Conv2d(512, 8, kernel_size = 3, padding = 1), # bottleneck of encoder
            
            # (batch, 8, 63, 63) -> (batch, 8, 63, 63)
            nn.Conv2d(8, 8, kernel_size = 1, padding = 0)
        )

    # encoder + sampling from latent space
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # noise: (batch, out_channels = 8, height = 63, width = 63), same size as encoder output
        # x: (batch, channels = 3, height = 512, width = 512)
        for module in self:
            # apply special padding for conv that have stride
            if getattr(module, "stride", None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1)) # added pixels on the right and bottom side of image 
            x = module(x)
            
        # outputs of variational autoencoder is mean and log variance of latent space
        # (batch, 8, 63, 63) -> two tensors of shape (batch, 4, 63, 63)
        mean, log_variance = torch.chunk(x, 2, dim = 1)
        
        # clamp log variance to make value range acceptable
        # (batch, 4, 63, 63)
        log_variance = torch.clamp(log_variance, -30, 20)
        
        # log variance -> variance
        # (batch, 4, 63, 63)
        variance = log_variance.exp()
        
        # (batch, 4, 63, 63)
        stdev = variance.sqrt()
        
        # how to sample: z = N(0, 1) -> N(mean, variance) := x
        # x = mean + stdev * z
        x = mean + stdev * noise
        
        # scale output by a constant
        x *= 0.18215
        
        return x


        