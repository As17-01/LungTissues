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


@hydra.main(config_path="experiment_configs", config_name="exp1", version_base="1.2")
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

    download = False
    if cfg_dct["dataset"] == "NoduleMNIST3D":
        train_data = medmnist.NoduleMNIST3D(split="train", download=download)
        valid_data = medmnist.NoduleMNIST3D(split="val", download=download)
        test_data = medmnist.NoduleMNIST3D(split="test", download=download)
    elif cfg_dct["dataset"] == "OrganMNIST3D":
        train_data = medmnist.OrganMNIST3D(split="train", download=download)
        valid_data = medmnist.OrganMNIST3D(split="val", download=download)
        test_data = medmnist.OrganMNIST3D(split="test", download=download)
    elif cfg_dct["dataset"] == "AdrenalMNIST3D":
        train_data = medmnist.AdrenalMNIST3D(split="train", download=download)
        valid_data = medmnist.AdrenalMNIST3D(split="val", download=download)
        test_data = medmnist.AdrenalMNIST3D(split="test", download=download)
    elif cfg_dct["dataset"] == "FractureMNIST3D":
        train_data = medmnist.FractureMNIST3D(split="train", download=download)
        valid_data = medmnist.FractureMNIST3D(split="val", download=download)
        test_data = medmnist.FractureMNIST3D(split="test", download=download)
    elif cfg_dct["dataset"] == "VesselMNIST3D":
        train_data = medmnist.VesselMNIST3D(split="train", download=download)
        valid_data = medmnist.VesselMNIST3D(split="val", download=download)
        test_data = medmnist.VesselMNIST3D(split="test", download=download)
    elif cfg_dct["dataset"] == "SynapseMNIST3D":
        train_data = medmnist.SynapseMNIST3D(split="train", download=download)
        valid_data = medmnist.SynapseMNIST3D(split="val", download=download)
        test_data = medmnist.SynapseMNIST3D(split="test", download=download)

    train_dataloader = DataLoader(train_data, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, num_workers=num_workers, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, num_workers=num_workers, batch_size=batch_size, shuffle=False)

    train_dataloader = src.utils.DeviceDataLoader(train_dataloader, device)
    valid_dataloader = src.utils.DeviceDataLoader(valid_dataloader, device)
    test_dataloader = src.utils.DeviceDataLoader(test_dataloader, device)

    model_list = [registry.get_from_params(**model) for model in cfg_dct["model"]]
    for i, model in enumerate(model_list):
        src.utils.to_device(model, device)

        with torch.no_grad():
            model.eval()
            history = [src.utils.evaluate(model, valid_dataloader, time_dimension=1)]

        history += src.utils.fit(num_epochs, lr, model, train_dataloader, valid_dataloader, test_dataloader, time_dimension=1, exp_name=cfg_dct["exp_name"], model_id=i)

        with torch.no_grad():
            model.eval()
            model.epoch_end(num_epochs, src.utils.evaluate(model, test_dataloader, time_dimension=1), "test")


if __name__ == "__main__":
    main()
