import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class SingleDataset(Dataset):
    def __init__(self, annotation_file, max_sequence_len=100, transform=None, target_transform=None):
        self.slide_labels = pd.read_csv(annotation_file, header=None)

        large_image_list = []
        small_image_list = []
        target_list = []
        for slide_path, label in zip(self.slide_labels.iloc[:, 0], self.slide_labels.iloc[:, 1]):
            for large_image in os.listdir(slide_path):
                full_large_image_path = slide_path + "/" + large_image
                for i in range(max_sequence_len):
                    large_image_list.append(full_large_image_path)
                    small_image_list.append(i)
                    target_list.append(label)

        self.all_labels = pd.DataFrame(
            {"large_image": large_image_list, "small_image": small_image_list, "target": target_list}
        )

        self.max_sequence_len = max_sequence_len
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        image_path = self.all_labels.iloc[idx, 0]
        image = read_image(image_path)
        image = image / 255

        if int(np.sqrt(self.max_sequence_len)) != np.sqrt(self.max_sequence_len):
            raise ValueError("Max sequence length must be a perfect square.")
        if int(image.shape[1] / np.sqrt(self.max_sequence_len)) != (image.shape[1] / np.sqrt(self.max_sequence_len)):
            raise ValueError("The size of slices should be divisible by the max_sequence_len.")

        kernel_size = int(image.shape[1] / np.sqrt(self.max_sequence_len))
        patches = image.unfold(1, kernel_size, kernel_size).unfold(2, kernel_size, kernel_size)
        patches = patches.contiguous().view(3, self.max_sequence_len, kernel_size, kernel_size)
        patches = patches.swapaxes(0, 1)

        idx_to_keep = self.all_labels.iloc[idx, 1]
        patches = patches[idx_to_keep]

        label = self.all_labels.iloc[idx, 2].astype("int")

        if self.transform:
            patches = self.transform(patches)
        if self.target_transform:
            label = self.target_transform(label)
        return patches, label
