import torch
from torch import nn
from torch.nn import functional as F

# 2 blocks from decoder
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

# encoder is a sequence of models, inherit nn.Sequential
class VAE_Encoder(nn.Sequential):
    