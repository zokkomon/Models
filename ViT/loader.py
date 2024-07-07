#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 6 23:36:40 2023

@author: zok
"""
import config

import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(config.IMG_SIZE ),
    transforms.CenterCrop(config.IMG_SIZE ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImageCaptionDataset(Dataset):
    def __init__(self, img_dir, caption_dir):
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.caption_paths = [os.path.join(caption_dir, f) for f in os.listdir(caption_dir) if f.endswith('.txt')]
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = transform(img)

        with open(self.caption_paths[idx], 'r') as f:
            caption = f.read().strip()
            
        return img, caption


# Create dataset and data loader
dataset = ImageCaptionDataset(img_dir="/home/zok/Videos/Models/ViT/download/image/", caption_dir="/home/zok/Videos/Models/ViT/download/text/")

# def collate_fn(batch):
#     imgs, captions, input_ids, attention_masks = zip(*batch)
#     lengths = [len(input_id) for input_id in input_ids]
#     sorted_idx = sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True)
#     imgs = [imgs[i] for i in sorted_idx]
#     captions = [captions[i] for i in sorted_idx]
#     input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
#     attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True)
#     return imgs, captions, input_ids, attention_masks
    
dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)


    
