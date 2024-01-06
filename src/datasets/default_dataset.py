import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class DefaultDataset(Dataset):
    def __init__(self, annotation_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotation_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        image = read_image(img_path)
        image = image / 255
        label = self.img_labels.iloc[idx, 1].astype("int")
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
