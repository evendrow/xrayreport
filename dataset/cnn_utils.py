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
from tqdm import tqdm

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

    model.eval()

    return model_for_use


def load_cnn_densenet():
    # load pretrained model and get last conv layer output
    densenet = models.densenet121(pretrained=True)
    modules = list(densenet.children())[:-1]
    model_for_use = nn.Sequential(*modules)
    return model_for_use


def get_cnn(model_name):
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


def make_image_representation_8_8_1024(data_dir, images, model, transform):
    # Get image
    image_tensor = []
    for image_path in images:
        image_dir = os.path.join(data_dir, "images/", image_path)
        image = Image.open(image_dir)
        image = transform(image).cuda()
        image_tensor.append(image.unsqueeze(0))

    # Run image through CNN
    image_batch = torch.cat(image_tensor).cuda() #image.unsqueeze(0)
    # print("Tensor size: ", image_batch.shape)
    
    extracted_features = model(image_batch)

    np_features = extracted_features.detach().cpu().numpy()
    # np_features = np_features.squeeze().transpose((1, 2, 0))
    np_features = np_features.squeeze().transpose((0, 2, 3, 1))

    return np_features


def extract_image_features(data_dir, image_list, model_name):
    matrix_list = []
    model, transform = get_cnn(model_name)
    
    # set model to evaluation mode (to use running value of alpha, gamma in batchnorm)
    model.eval()
    model.cuda()
    with torch.no_grad():
        batch_size = 64
        for i in tqdm(range(0, len(image_list), batch_size)):
            images = image_list[i:min(i+batch_size,len(image_list))]
            matrix = make_image_representation_8_8_1024(data_dir, images, model, transform)
            matrix_list.append(matrix)
    # final_matrix = np.asarray(matrix_list)
    final_matrix = np.concatenate(matrix_list, axis=0)
    print("final matrix size: ", final_matrix.shape)

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


def save_annotations(features, clinical_notes, image_list, save_path):
    '''
    Saves features and clinical notes annotations
    - features: List of image features
    - clinical_notes: list of clinical notes

    NOTE: features and clinical_notes must have the same size (0th dim)

    Saves each feature/note pair to a .npy file as a dictionary
    {
        "features":  [features],
        "note":     [ "<S>", "Ethan", "is", "a", "big", "guy", ...]
    }
    '''
    paths = []
    for i in tqdm(range(features.shape[0])):
        new_filename = os.path.basename(image_list[i]) # file.jpg
        new_filename = os.path.splitext(new_filename)[0]+".npy" # file.npy
        paths.append(new_filename)
        new_path = os.path.join(save_path, new_filename)
        
        np.save(new_path, {
            "features": features[i], 
            "note":    clinical_notes[i]
        })

    # save paths list to a csv file for reading by the dataset class
    df = pd.DataFrame({"path": paths})  
    
    # saving the dataframe  
    df.to_csv(os.path.join(save_path, 'paths.csv'), index=None)

def save_annotations_double(features_chexpert, features_imagenet, clinical_notes,
                            image_list, save_path):
    '''
    Saves features and clinical notes annotations
    - features_chexpert: List of chexpert image features
    - features_imagenet: List of imagenet-pretrained network image features
    - clinical_notes: list of clinical notes

    NOTE: features and clinical_notes must have the same size (0th dim)

    Saves each feature/note pair to a .npy file as a dictionary
    {
        "features":  [features],
        "note":     [ "<S>", "Ethan", "is", "a", "big", "guy", ...]
    }
    '''
    paths = []
    for i in tqdm(range(len(image_list))):
        new_filename = os.path.basename(image_list[i]) # file.jpg
        new_filename = os.path.splitext(new_filename)[0]+".npy" # file.npy
        paths.append(new_filename)
        new_path = os.path.join(save_path, new_filename)
        
        np.save(new_path, {
            "features_chexpert": features_chexpert[i], 
            "features_imagenet": features_imagenet[i], 
            "note":    clinical_notes[i]
        })

    # save paths list to a csv file for reading by the dataset class
    df = pd.DataFrame({"path": paths})  
    
    # saving the dataframe  
    df.to_csv(os.path.join(save_path, 'paths.csv'), index=None)

