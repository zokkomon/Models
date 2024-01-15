#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:26:42 2024

@author: zok
"""

import torch
import torch.nn as nn


class Codebook(nn.Module):
    def __init__(self, args):
        super(Codebook, self).__init__()
        self.codebook_vectors = args.codebook_vectors
        self.latent_dim = args.latent_dim
        
        self.embeddings = nn.Embedding(self.codebook_vectors, self.latent_dim)
        self.embeddings.weight.data.uniform_(-1.0 / self.codebook_vectors, 1.0 / self.codebook_vectors)
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contigous()
        z = x.view(-1, self.latent_dim)
        
        dim = torch.sum(z**2, dim=1, keepdim=True) + \
            torch.sum(self.embeddings.weight**2, dim=1) - \
            2 * (torch.matmul(z, self.embeddings.weight.t()))
            
        min_encoding_indices = torch.argmin(dim, dim=1)
        z_quant = self.embeddings(min_encoding_indices).view(z.shape)
        
        loss = torch.mean((z_quant.detach() - x)**2) + self.betas * torch.mean((z_quant - x.detach())**2)
        
        z_quant = x + (z_quant - x).detach()  
        z_quant = z_quant.permute(0, 3, 1, 2)
        
        return z_quant, min_encoding_indices, loss
        
        
