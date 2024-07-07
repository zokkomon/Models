
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 23:03:25 2023

@author: zok
"""

import torch
import torch.nn as nn
from functools import partial
import config
from patcher import PatchEmbedding
from MLP import Mlp
from attention import SelfAttention

class Block(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(num_heads, dim, in_proj_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ViT(nn.Module):
    
    def __init__(self, num_patches, img_size, num_classes, patch_size, embed_dim, 
                 num_heads, dropout, in_channels, depth=12, mlp_ratio=4., 
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None):
        super().__init__()
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        assert img_size % patch_size == 0 , 'Image dimensions must be divisible by the patch size.'
        
        self.embeddings_block = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
        x = self.norm(x)
        
        return x

# model = ViT(config.NUM_PATCHES, config.IMG_SIZE, config.NUM_CLASSES, config.PATCH_SIZE, config.EMBED_DIM, config.NUM_HEADS, config.DROPOUT, config.IN_CHANNELS).to(config.device)
# x = torch.randn(5, 3, 224, 224).to(config.device)
# print(model(x).shape) # BATCH_SIZE X NUM_CLASSES
