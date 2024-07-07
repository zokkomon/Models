#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 17:35:02 2023

@author: zok
"""

from dataclasses import dataclass
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import precompute_theta_pos_frequencies, rotary_embeddings
from moe import MoeArgs, MoeLayer
from cache import CacheView, RotatingBufferCache
from xformers.ops.fmha import memory_efficient_attention

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1 # Later set in the build method
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None
    
@dataclass
class SimpleInputMetadata:
    # rope absolute positions
    positions: torch.Tensor

    @staticmethod
    def from_seqlen(seq_lens: List[int], device: torch.device) -> "SimpleInputMetadata":
        return SimpleInputMetadata(
            positions=torch.cat([torch.arange(0, seq_len) for seq_len in seq_lens]).to(
                device=device, dtype=torch.long
            )
        )
    
class RMSNORM(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        #The gamma parameter
        self.weight = nn.Parameter(torch.ones_like(dim))
        
    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float().type_as(x))
 


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        # Number of kv heads is less than query heads
        self.n_kv_heads = args.n_heads if self.n_kv_heads is None else args.n_kv_heads
        self.n_q_heads = args.n_heads
        self.n_rep = self.n_q_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)
        
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor,cache: Optional[CacheView], mask: Optional[torch.Tensor]):
        # b, 1, dim
        batch_size, seq_len, _ = x.shape
        
        # (B, 1, Dim) -> (B, 1, H_Q * Head_Dim) -> (B, 1, H_KV , Head_Dim)
        xq = self.wq(x).view(batch_size, seq_len, self.n_q_heads, self.head_dim)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV , Head_Dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # (B, 1, H_Q, Head_Dim) --> (B, 1, H_Q, Head_Dim)
        xq = rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = rotary_embeddings(xk, freqs_complex, device=x.device)
        
        if cache is None:
            key, val = xk, xv
        elif cache.prefill:
            key, val = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
        else:
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            key = key.view(
                batch_size * cache.sliding_window, self.n_kv_heads, self.head_dim
            )
            val = val.view(
                batch_size * cache.sliding_window, self.n_kv_heads, self.head_dim
            )
        
        # Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q in the same group.

        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = repeat_kv(key, self.n_rep)
        values = repeat_kv(val, self.n_rep)
        
        # xformers requires (B=1, S, H, D)
        xq, keys, values = xq[None, ...], key[None, ...], val[None, ...]
        out = memory_efficient_attention(
            xq, keys, values, None if cache is None else cache.mask
        )

        return self.wo(out.view(batch_size, self.n_heads * self.head_dim))
        
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
        
    def forward(self, x: torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
        
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        
        self.attention = SelfAttention(args)
        if args.moe is not None:
            self.feed_forward = MoeLayer(
                experts=[FeedForward(args=args) for _ in range(args.moe.num_experts)],
                gate=nn.Linear(args.dim, args.moe.num_experts, bias=False),
                moe_args=args.moe,
            )
        else:
            self.feed_forward = FeedForward(args)
        
        self.attention_norm = RMSNORM(args.dim, args.norm_eps)
        self.ffn_norm = RMSNORM(args.dim, args.norm_eps)
        
    def forward(self, x: torch.Tensor, cache: Optional[CacheView], freqs_complex: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), cache, freqs_complex, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
        
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size != -1
        
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        
        self.layers = nn.ModuleList()
        for lay in range(self.n_layers):
            self.layers.append(TransformerBlock(args))
        
        self.norm = RMSNORM(args.dim, args.norm_eps)
        self.ffn = nn.Linear(args.dim, self.vocab_size, bias=False)
        
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len, device= self.args.device)
        
    def forward(self, tokens: torch.Tensor, cache: Optional[RotatingBufferCache] = None):
        batch, seq_len = tokens.shape
        assert seq_len == 1
        
        if cache is not None:
           input_metadata = cache.get_input_metadata(seq_len)
        else:
           input_metadata = SimpleInputMetadata.from_seqlens(seq_len, self.device)
        
        h = self.tok_embeddings(tokens)
        
        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[input_metadata.positions]
        
        mask = True
        # if seq_len > 1:
        #     mask = torch.full(
        #         (seq_len, seq_len), float("-inf"), device=tokens.device
        #     )

        #     mask = torch.triu(mask, diagonal=1)

        #     # When performing key-value caching, we compute the attention scores
        #     # only for the new sequence. Thus, the matrix of scores is of size
        #     # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
        #     # j > cache_len + i, since row i corresponds to token cache_len + i.
        #     mask = torch.hstack([
        #         torch.zeros((seq_len, start_pos), device=tokens.device),
        #         mask
        #     ]).type_as(h)
        
        for local_layer_id, layer in enumerate(self.layers.values()):
            if cache is not None:
                assert input_metadata is not None
                cache_view = cache.get_view(local_layer_id, input_metadata)
            else:
                cache_view = None
            h = layer(h, freqs_complex, cache_view)
        
        if cache is not None:
            cache.update_seqlens(seq_len)
            h = layer(h, cache_view, freqs_complex, mask)
  
        out = self.ffn(self.norm(h)).float()
        
        return out.float()
        
        
        
        
        
