#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import time
import sys
import os

import torch
from torch import nn
import torch.nn.functional as F


from catr.configuration import Config
from catr.models.utils import NestedTensor, nested_tensor_from_tensor_list, get_rank
from catr.models.backbone import build_backbone
from catr.models.transformer import build_transformer
from catr.models.position_encoding import PositionEmbeddingSine
from catr.models.caption import MLP

from get_word_embeddings import *
from load_glove_840B_300d import *

import math
import torch.optim as optim
import copy
import json


# ### First, define our custom caption class

# In[2]:


class Xray_Captioner(nn.Module):
    def __init__(self, transformer, feature_dim, hidden_dim, vocab_size):
        super().__init__()
        self.input_proj = nn.Conv2d(
            feature_dim, hidden_dim, kernel_size=1) # project feature dimension
        self.position_embedding = PositionEmbeddingSine(hidden_dim//2, normalize=True)
        self.transformer = transformer
        self.mlp = MLP(hidden_dim, 512, vocab_size, 3)

    def forward(self, img_features, target, target_mask):
        # The input mask here is all zeros, meaning we look at the whole image
        # The mask here is more of a formality, oringinally implemented to 
        # let the model accept different image sizes. Not needed here.
        b, c, h, w = img_features.shape
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=img_features.device)

        # Get projected image features and positional embedding
        img_embeds = self.input_proj(img_features)
        pos = self.position_embedding(NestedTensor(img_embeds, mask))
        
        # Run through transformer -> linear -> softmax
        hs = self.transformer(img_embeds, mask,
                              pos, target, target_mask)
        out = self.mlp(hs.permute(1, 0, 2))
        return out
    

def build_model(config):
    transformer = build_transformer(config)
    model = Xray_Captioner(transformer, config.feature_dim, config.hidden_dim, config.vocab_size)
    criterion = torch.nn.CrossEntropyLoss()

    return model, criterion


# ### This method builds the model like we will during training/inference

# In[3]:


''' 
This method uses a config file to appropriately create the model.
This includes setting the device and specifying a random seed    
''' 
def main(config):
    # initialize device we're runnign this on
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')

    # specify the random seed for deterministic behavior
    seed = config.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # create the model
    model, criterion = build_model(config)
    model.to(device)
    
    # sanity check
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")
    
    return model, criterion


# In[4]:


#words = np.load("glove_embed.npy")
#with open('word2ind.json') as json_file: 
#    word2ind = json.load(json_file) 
#with open('ind2word.json') as json_file: 
#    ind2word = json.load(json_file) 


# ### Create a model

# In[5]:


# Create a sample config file
# feature_dim is not specified by default, so we need to set it
#config = Config()
#config.device = 'cpu' # if running without GPU
#config.feature_dim = 1024
#config.pad_token_id = word2ind["<S>"]
#config.hidden_dim = 300
#config.nheads = 10
#config.vocab_size = words.shape[0]
#config.__dict__["pre_embed"] = torch.from_numpy(words)

# Create the model!
#xray_model = main(config)
#xray_model = xray_model.double()


# In[6]:


# Helper function to create initial caption and mask
def create_evaluation_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, :] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


# In[7]:


#from dataset.cnn_utils import extract_image_features
#from dataset.dataset import ImageFeatureDataset
#from torch.utils.data import DataLoader


# In[8]:


#dataset = ImageFeatureDataset('../mimic_features/paths.csv',
#                             '../mimic_features/')
#dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
#iterations = iter(dataloader)


# In[9]:


# getting predicted sentence for an image
#image, note = next(iterations)
#start_token = word2ind["<S>"]
#current_caption, current_mask = create_evaluation_caption_and_mask(start_token, config.max_position_embeddings)
#iteration_number = 1
#last_word = start_token
#while iteration_number < config.max_position_embeddings and last_word != word2ind["</s>"]:
#    predictions = xray_model(image.double(), current_caption, current_mask)
#    # get highest predicted word
#    word = torch.argmax(predictions[:,0,:], axis=-1)
#    try:
#        print(ind2word[str(word.item())])
#    except:
#        pass
#    current_caption[:, iteration_number] = word
#    current_mask[:, iteration_number] = False
#    iteration_number += 1


# In[ ]:




