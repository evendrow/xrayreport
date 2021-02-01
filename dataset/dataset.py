import numpy as np
import pandas as pd
import os

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence

class ImageFeatureDataset(Dataset):
    """Single Image Feature dataset."""

    def __init__(self, config, mode='train', transform=None, pooling=None, kernel_size=2):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.config = config
        self.mode = mode
        self.limit = config.limit
        self.max_length = config.max_position_embeddings
        self.tensorTransform = transforms.ToTensor()
        self.transform = transform
        self.pooling = pooling
        self.kernel_size = kernel_size

        self.root_dir = os.path.join(config.dir, mode)
        self.landmarks_frame = pd.read_csv(os.path.join(self.root_dir, 'paths.csv'))
        

    def __len__(self):
        return 128#len(self.landmarks_frame)

    def get_annotation_features_list(self, annotation):
        return [annotation["features"]]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = 0

        # feature name is someothing like "123.npy"
        feature_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        annotation = np.load(feature_name, allow_pickle=True).item()
        
        features_list = self.get_annotation_features_list(annotation)

        note = torch.tensor(annotation["note"])
        note = note[:min(len(note), self.max_length+1)]
        note_padded = note.data.new((self.max_length+1)).fill_(0) # pad to max, fill with zeros
        
        note_size = min(len(note), len(note_padded))
        note_padded[0:note_size] = note

        note_mask = torch.ones_like(note_padded)
        note_mask[0:note_size] = 0
        note_mask = note_mask.bool()

        # Convert each feature type into Tensor, and pool as needed
        for i in range(len(features_list)):
            features_list[i] = self.tensorTransform(features_list[i])

            if self.pooling == "max":
                features_list[i] = nn.MaxPool2d(kernel_size, stride=kernel_size)(features_list[i])
            elif self.pooling == "avg":
                features_list[i] = nn.AvgPool2d(kernel_size, stride=kernel_size)(features_list[i])

            # Apply any given transforms
            if self.transform:
                features_list[i] = self.transform(features_list[i])

        # If only one feature (e.g. just chexpert), just return that
        if len(features_list) == 1:
            features_list = features_list[0]

        return features_list, note_padded, note_mask


# All we need to do is override which features we get
class ImageDoubleFeatureDataset(ImageFeatureDataset):
    def get_annotation_features_list(self, annotation):
        return [annotation["features_chexpert"], annotation["features_imagenet"]]




