import torch
from torch import nn
from torch.nn import functional as F
import math

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True,  attn_drop=0., proj_drop=0.):
        super().__init__()
        # This combines the Wq, Wk and Wv matrices into one matrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        # This one represents the Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, causal_mask=False):
        # (Batch_Size, Seq_Len, Dim)
        input_shape = x.shape 
        
        # (Batch_Size, Seq_Len, Dim)
        batch_size, sequence_length, d_embed = input_shape 

        # (Batch_Size, Seq_Len, H, Dim / H)
        interim_shape = (batch_size, sequence_length, self.n_heads, d_embed // self.n_heads) 

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (Batch_Size, H, Seq_Len, Dim) @ (Batch_Size, H, Dim, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-2, -1)
        
        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
            # Fill the upper triangle with -inf
            weight.masked_fill_(mask, -torch.inf) 
        
        # Divide by d_k (Dim / H). 
        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight /= math.sqrt(self.d_head) 

        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = F.softmax(weight, dim=-1) 
        weight = self.attn_drop(weight)

        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v

        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2) 

        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output = output.reshape(input_shape) 

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.out_proj(output) 
        output = self.out_drop(output)
        
        # (Batch_Size, Seq_Len, Dim)
        return output

class ResAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.gn = nn.GroupNorm(32, channels)
        self.values = nn.Conv2d(channels, channels, 1, 1, 0)
        self.keys = nn.Conv2d(channels, channels, 1, 1, 0)
        self.queries = nn.Conv2d(channels, channels, 1, 1, 0)
        self.dense = nn.Conv2d(channels, channels, 1, 1, 0)
          
    def forward(self, x):
        # x.shape :(batch, channels, height, weight)
        n, c, h, w  = x.shape
           
        
        std = self.gn(x)
        values = self.values(std)
        keys = self.keys(std)
        queries = self.queries(std)
        
        # (batch, channels, height, weight) -> ((batch, channels, height*weight) -> (batch, height*weight, channels)
        queries = queries.reshape(n, c, h*w)
        queries.permute(0, 2, 1)
        # (batch, channels, height, weight) -> (batch, channels, height*weight)
        keys = keys.reshape(n, c, h*w)
        values = values.reshape(n, c, h*w)
        
        # (batch, height*weight, channels) @ (batch, channels, height*weight) -> (Batch_Size, height*weight, height*weight)
        weight = queries @ keys
        
        # (Batch_Size, height*weight, height*weight) @ (Batch_Size, channels, height*weight) -> (Batch_Size, channels, height*weight)
        weight = weight * (int(c)**(-0.5))
        attention = torch.softmax(weight , dim=2)
        attention = attention.permute(0, 2, 1)
        
        out = attention @ values
        
        # (Batch_Size, channels, height*weight  -> (batch, channels, height, weight)
        out = out.view(x.shape)       
        
        # (batch, channels, height, weight) -> (batch, channels, height, weight)
        c_Attn = self.dense(out)
        
        # (batch, channels, height, weight)
        return x + c_Attn
