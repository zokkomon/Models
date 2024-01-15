#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:26:42 2024

@author: zok
"""

import torch
import torch.nn as nn
        
class SelfAttention(nn.Module):
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
      
        
