import numpy as np
import pandas as pd
import os

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset

class ImageFeatureDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, pooling=None, kernel_size=2):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.tensorTransform = transforms.ToTensor()
        self.transform = transform
        self.pooling = pooling
        self.kernel_size = kernel_size

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # feature name is someothing like "123.npy"
        feature_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        annotation = np.load(feature_name, allow_pickle=True).item()
        
        features = annotation["features"]
        note = annotation["note"]

        # Convert to Torch tensor
        features = self.tensorTransform(features)

        if self.pooling == "max":
            features = nn.MaxPool2d(kernel_size, stride=kernel_size)(features)
        elif self.pooling == "avg":
            features = nn.AvgPool2d(kernel_size, stride=kernel_size)(features)

        # Apply any given transforms
        if self.transform:
            features = self.transform(features)

        return features, note
