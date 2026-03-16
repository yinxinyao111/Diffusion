import torch
from torch import nn 
from torch.nn import functional as F
from attention import SelfAttention

# Auxillary for CLIP 
class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_tokens):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))
    def forward(self, tokens):
        # (batch, seq_len) -> (batch, seq_len, d_model)
        x = self.token_embedding(x)
        x += self.position_embedding
        return x
    
class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(n_embd * 4, n_embd)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, d_model)
        residue = x
        # self attention + add and norm
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask = True)
        x += residue
        # feed forward + add and norm
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x) # quick GELU activation
        x = self.linear(x)
        x += residue
        return x
    
        

class CLIP(nn.Module):
    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77) # (vocab_size, embed_dim, seq_len)
        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12) # 12 heads for MHA, 12 layers
        ])
        self.layernorm = nn.LayerNorm(768)
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        # (batch, seq_len) -> (batch, seq_len, d_model = 768)
        state = self.embedding(tokens)
        
        for layer in self.layers:
            state = layer(state)
        output = self.layernorm(state)
        return output
    
