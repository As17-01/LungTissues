import sys

import hydra
import medmnist
import omegaconf
import torch
from hydra_slayer import Registry
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

sys.path.append("../../../")

import src.datasets
import src.models
import src.utils


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:

    train_data = medmnist.NoduleMNIST3D(split="train", download=True)
    valid_data = medmnist.NoduleMNIST3D(split="val", download=True)
    test_data = medmnist.NoduleMNIST3D(split="test", download=True)


if __name__ == "__main__":
    main()
