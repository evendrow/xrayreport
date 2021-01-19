#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
import torchvision.models as models


# In[ ]:


def load_cnn_chexpert():
    
    import os 
    import sys
    from jsonextended import edict
    
    initial_path = os.getcwd()
    sys.path.append(os.path.dirname(os.path.abspath("__file__")) + "/Chexpert")
    from model.classifier import Classifier
    path = os.getcwd()+"/Chexpert/config/"
    os.chdir(path)
    with open('example.json', 'r') as f:
        cfg = edict.LazyLoad(json.load(f))
    model = Classifier(cfg)
    model.load_state_dict(torch.load("pre_train.pth", map_location=torch.device('cpu')))
    modules = list(model.children())
    modules = list(modules[0].children())[:-1]
    model_for_use = nn.Sequential(*modules)
    os.chdir(initial_path)
    return model_for_use


# In[2]:


def run_utils(model):
    if model == "densenet121":
        densenet = models.densenet121(pretrained=True)
        modules = list(densenet.children())[:-1]
        model_for_use = nn.Sequential(*modules)
    elif model == "chexpert":
        model_for_use = load_cnn_chexpert()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model_for_use, preprocess


# In[3]:


def make_image_representation_8_8_1024(image, model, preprocess):
    image_dir = str(str(os.getcwd())+"/images/" + image)
    image = Image.open(image_dir)
    image = preprocess(image)
    image_batch = image.unsqueeze(0)
    model.eval()
    extracted_features = model(image_batch)
    np_features = extracted_features.detach().numpy()
    np_features = np_features.squeeze().transpose((1, 2, 0))
    return np_features


# In[4]:


def main_get_matrix_from_cnn(image_list, model):
    matrix_list = []
    model, preprocess = run_utils(model)
    for image in image_list:
        matrix = make_image_representation_8_8_1024(image, model, preprocess)
        matrix_list.append(matrix)
    final_matrix = np.asarray(matrix_list)
    return final_matrix

