import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class DefaultDataset(Dataset):
    def __init__(self, annotation_file, max_sequence_len=100, keep_share=0.25, random_seed=None, transform=None, target_transform=None):
        self.slide_labels = pd.read_csv(annotation_file, header=None)
        self.max_sequence_len = max_sequence_len
        self.keep_share = keep_share
        self.transform = transform
        self.target_transform = target_transform

        np.random.seed(random_seed)

    def __len__(self):
        return len(self.slide_labels)

    def __getitem__(self, idx):
        slide_path = self.slide_labels.iloc[idx, 0]

        image_path = np.random.choice(os.listdir(slide_path))
        image = read_image(slide_path + "/" + image_path)
        image = image / 255

        if int(np.sqrt(self.max_sequence_len)) != np.sqrt(self.max_sequence_len):
            raise ValueError("Max sequence length must be a perfect square.")
        if int(image.shape[1] / np.sqrt(self.max_sequence_len)) != (image.shape[1] / np.sqrt(self.max_sequence_len)):
            raise ValueError("The size of slices should be divisible by the max_sequence_len.")

        kernel_size = int(image.shape[1] / np.sqrt(self.max_sequence_len))
        patches = image.unfold(1, kernel_size, kernel_size).unfold(2, kernel_size, kernel_size)
        patches = patches.contiguous().view(3, self.max_sequence_len, kernel_size, kernel_size)
        patches = patches.swapaxes(0, 1)

        idx_to_keep = np.random.choice(range(len(patches)), size=int(len(patches) * self.keep_share), replace=False)
        patches = patches[idx_to_keep]

        label = self.slide_labels.iloc[idx, 1].astype("int")
        if self.transform:
            patches = self.transform(patches)
        if self.target_transform:
            label = self.target_transform(label)
        return patches, label
