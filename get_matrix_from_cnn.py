import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
import torchvision.models as models

import os 
import sys
import json
from jsonextended import edict

CHEXPERT_PATH = "Chexpert"
sys.path.append(os.path.join(os.path.dirname(os.path.abspath("__file__")), CHEXPERT_PATH))

from model.classifier import Classifier

def load_cnn_chexpert():
    # Find the config file
    cfg_path = os.path.join(CHEXPERT_PATH, "config/example.json")
    with open(cfg_path, 'r') as f:
        cfg = edict.LazyLoad(json.load(f))
    
    # Create model and take the last convolutional layer
    model_path = os.path.join(CHEXPERT_PATH, "config/pre_train.pth")
    model = Classifier(cfg)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    modules = list(model.children())
    last_layer = list(modules[0].children())[:-1]
    model_for_use = nn.Sequential(*last_layer)

    return model_for_use


def load_cnn_densenet():
    # load pretrained model and get last conv layer output
    densenet = models.densenet121(pretrained=True)
    modules = list(densenet.children())[:-1]
    model_for_use = nn.Sequential(*modules)
    return model_for_use


def run_utils(model_name):
    if model_name == "densenet121":
        model = load_cnn_densenet()
    elif model_name == "chexpert":
        model = load_cnn_chexpert()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


def make_image_representation_8_8_1024(data_dir, image, model, transform):
    # Get image
    image_dir = os.path.join(data_dir, "images/", image)
    image = Image.open(image_dir)
    image = transform(image)

    # Run image through CNN
    image_batch = image.unsqueeze(0)
    extracted_features = model(image_batch)
    np_features = extracted_features.detach().numpy()
    np_features = np_features.squeeze().transpose((1, 2, 0))

    return np_features


def extract_image_features(data_dir, image_list, model_name):
    matrix_list = []
    model, transform = run_utils(model_name)
    
    # set model to evaluation mode (to use running value of alpha, gamma in batchnorm)
    model.eval()

    for image in image_list:
        matrix = make_image_representation_8_8_1024(data_dir, image, model, transform)
        matrix_list.append(matrix)
    final_matrix = np.asarray(matrix_list)

    return final_matrix


def save_feature_matrix(matrix, image_list, save_path):
    paths = []
    for i in range(matrix.shape[0]):
        new_filename = os.path.basename(image_list[i]) # file.jpg
        new_filename = os.path.splitext(new_filename)[0]+".npy" # file.npy
        paths.append(new_filename)
        new_path = os.path.join(save_path, new_filename)
        np.save(new_path, matrix[i])

    # save paths list to a csv file for reading by the dataset class
    df = pd.DataFrame({"path": paths})  
    
    # saving the dataframe  
    df.to_csv(os.path.join(save_path, 'paths.csv'), index=None)


