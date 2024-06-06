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


# Change config_path to switch between experiment setup and train setup
@hydra.main(config_path="experiment_configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    device = src.utils.get_default_device()
    logger.info(f"Current device is {device}")

    num_workers = cfg.training_params.num_workers
    batch_size = cfg.training_params.batch_size
    lr = cfg.training_params.learning_rate
    num_epochs = cfg.training_params.num_epochs

    cfg_dct = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    registry = Registry()
    registry.add_from_module(src.datasets, prefix="src.datasets.")
    registry.add_from_module(src.models, prefix="src.models.")

    train_data = medmnist.NoduleMNIST3D(split="train", download=False)
    valid_data = medmnist.NoduleMNIST3D(split="val", download=False)
    test_data = medmnist.NoduleMNIST3D(split="test", download=False)

    train_dataloader = DataLoader(train_data, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, num_workers=num_workers, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, num_workers=num_workers, batch_size=batch_size, shuffle=False)

    train_dataloader = src.utils.DeviceDataLoader(train_dataloader, device)
    valid_dataloader = src.utils.DeviceDataLoader(valid_dataloader, device)
    test_dataloader = src.utils.DeviceDataLoader(test_dataloader, device)

    model = registry.get_from_params(**cfg_dct["model"])
    src.utils.to_device(model, device)

    with torch.no_grad():
        model.eval()
        history = [src.utils.evaluate(model, valid_dataloader, time_dimension=1)]

    history += src.utils.fit(num_epochs, lr, model, train_dataloader, valid_dataloader, test_dataloader, time_dimension=1)

    with torch.no_grad():
        model.eval()
        model.epoch_end(num_epochs, src.utils.evaluate(model, test_dataloader, time_dimension=1), "test")


if __name__ == "__main__":
    main()
