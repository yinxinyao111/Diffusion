import torch 
from torch import nn 
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias = True, out_proj_bias = True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x: torch.Tensor, causal_mask = False):
        # x: (batch, seq_len, d_model)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        
        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)
        
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model * 3) -> (batch, seq_len, d_model)
        q, k, v = self.in_proj(x).chunk(3, dim = -1)
        
        # (batch, h, seq_len, d_k)
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)
        
        # (batch, h, seq_len, seq_len)
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            mask = torch.ones_like(weight, dtype = torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_head)
        
        weight = F.softmax(weight, dim = -1)
        
        # (batch, h, seq_len, d_k)
        output = weight @ v
        
        # (batch, seq_len, h, d_k)
        output = output.transpose(1, 2)
        
        output = output.reshape(input_shape)
        
        output = self.out_proj(output)
        
        # (batch, seq_len, d_model)
        return output
        
        