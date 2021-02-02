#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import statements
import numpy as np
import time
import sys
import os
import tqdm

import torch
from torch import nn
import torch.nn.functional as F


from catr.configuration import Config
from catr.models.utils import NestedTensor, nested_tensor_from_tensor_list, get_rank
from catr.models.backbone import build_backbone
from catr.models.transformer import build_transformer
from catr.models.position_encoding import PositionEmbeddingSine
from catr.models.caption import MLP

import json

from dataset.dataset import ImageFeatureDataset
from torch.utils.data import DataLoader
from transformer_ethan import *
sys.path.append(os.path.join(os.path.dirname("__file__"), "catr"))
from engine import train_one_epoch, evaluate


# In[ ]:


# set up config
def make_config():
    global words
    words = np.load("glove_embed.npy")
    with open('word2ind.json') as json_file: 
        global word2ind
        word2ind = json.load(json_file) 
    with open('ind2word.json') as json_file: 
        global ind2word
        ind2word = json.load(json_file) 
    global config
    config = Config()
    config.feature_dim = 1024
    config.pad_token_id = word2ind["<S>"]
    config.hidden_dim = 300
    config.nheads = 10
    config.batch_size = 64
    config.encoder_type = 1
    config.vocab_size = words.shape[0]
    config.dir = '../mimic_features'
    config.__dict__["pre_embed"] = torch.from_numpy(words).to(config.device)
    return 


# In[ ]:


def make_and_load_model():
    global model
    global criterion
    model, criterion = main(config)
    model = model.float()
    global device
    device = torch.device(config.device)
    model.to(device)
    global param_dicts
    param_dicts = [
        {"params": [p for n, p in model.named_parameters(
        ) if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config.lr_backbone,
        },
    ]
    global optimizer
    optimizer = torch.optim.AdamW(
        param_dicts, lr=config.lr, weight_decay=config.weight_decay)
    global lr_scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)
    global dataset_train
    dataset_train = ImageFeatureDataset(config, mode='train')
    global dataset_val
    dataset_val = ImageFeatureDataset(config, mode='val')
    global sampler_train
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    global sampler_val
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    global batch_sampler_train
    batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, config.batch_size, drop_last=True)
    global data_loader_train
    data_loader_train = DataLoader(
            dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers)
    global data_loader_val
    data_loader_val = DataLoader(dataset_val, config.batch_size,
                                 sampler=sampler_val, drop_last=False, num_workers=config.num_workers)
    print(f"Train: {len(dataset_train)}")
    print(f"Val: {len(dataset_val)}")
    if os.path.exists(config.checkpoint):
      print("Loading Checkpoint...")
    global checkpoint
      checkpoint = torch.load(config.checkpoint)#, map_location='cuda')
      model.load_state_dict(checkpoint['model'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
      config.start_epoch = checkpoint['epoch'] + 1

    print("Start Training..")


# In[ ]:


def main():
    make_config()
    print(f"Setup config with device {config.device}")
    make_and_load_model()
    print("Model loaded with checkpoint. Beginning training.")
    train_loss_hist = []
    val_loss_hist = []
    train_bleu_hist = []
    val_bleu_hist = []

    for epoch in range(config.start_epoch, config.epochs):
        print(f"Epoch: {epoch}")
        epoch_loss, train_bleu_score = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, config.clip_max_norm, word2ind)
        train_loss_hist.append(epoch_loss)
        train_bleu_hist.append(train_bleu_score)
        lr_scheduler.step()
        print(f"Training Bleu Score: {train_bleu_score}")
        print(f"Training Loss: {epoch_loss}")

        torch.save({
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'lr_scheduler': lr_scheduler.state_dict(),
             'epoch': epoch,
         }, config.checkpoint)

        validation_loss, val_bleu_score = evaluate(model, criterion, data_loader_val, device, word2ind)
        val_loss_hist.append(validation_loss)
        val_bleu_hist.append(val_bleu_score)
        print(f"Validation Bleu Score: {val_bleu_score}")
        print(f"Validation Loss: {validation_loss}")

        print()


# In[ ]:


if __name__ == "__main__":
    main()

