#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:48:08 2023

@author: zok
"""

import torch 

RANDOM_SEED = 42
BATCH_SIZE = 10
EPOCHS = 40
LEARNING_RATE = 1e-4
NUM_CLASSES = 1000
PATCH_SIZE = 16
IMG_SIZE = 224
IN_CHANNELS = 3
NUM_HEADS = 12
DROPOUT = 0.001
HIDDEN_DIM = 4
ADAM_WEIGHT_DECAY = 0
ADAM_BETAS = (0.9, 0.999)
ACTIVATION="gelu"
NUM_ENCODERS = 4
EMBED_DIM = (PATCH_SIZE ** 2) * IN_CHANNELS 
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2 

# random.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)
# torch.manual_seed(RANDOM_SEED)
# torch.cuda.manual_seed(RANDOM_SEED)
# torch.cuda.manual_seed_all(RANDOM_SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

device = "cpu" if torch.cuda.is_available() else "cpu"
