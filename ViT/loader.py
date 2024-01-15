#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 6 23:36:40 2023

@author: zok
"""

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import config

import os
import pandas as pd
from torchvision.io import read_image

directory="/home/zok/Videos/Models/ViT/download/text/"
print(os.listdir(directory))

# for idx,filename in enumerate(os.listdir(directory)):
        
#         if filename[idx].endswith(".txt"):
#             file_path = os.path.join(directory, filename[idx])
#             with open(file_path, "r") as file:
#                 text = file.read()
# print(text)

# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#         f = open(annotations_file, "r")
#         print(f.read()) 
#         self.img_labels = 
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label
        
# loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
