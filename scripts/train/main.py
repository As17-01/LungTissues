import datetime
import json
import pathlib
import sys

import hydra
import omegaconf
import torch
from hydra_slayer import Registry
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

sys.path.append("../../")

import src.datasets
import src.models.single
import src.models.stacked


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device."""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device."""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def evaluate(model, val_loader):
    """Evaluate the model's performance."""
    outputs = []
    logger.info("Evaluating...")
    for i, batch in enumerate(val_loader):
        if i % 100 == 0:
            logger.info(f"{i} / {len(val_loader)}")
        outputs.append(model.validation_step(batch))
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    """Train the model using gradient descent."""
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = pathlib.Path(f"./saved_models/{current_time}")
    save_dir.mkdir(exist_ok=True, parents=True)

    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        model.train()
        # Training Phase
        logger.info("Training...")
        for i, batch in enumerate(train_loader):
            if i % 100 == 0:
                logger.info(f"{i} / {len(train_loader)}")
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Validation phase
        with torch.no_grad():
            model.eval()

            result = evaluate(model, val_loader)
            model.epoch_end(epoch, result)
            history.append(result)

        if epoch % 100 == 99:
            torch.save(model.state_dict(), save_dir / f"epoch{epoch}.pt")
            with open(save_dir / f"epoch{epoch}.json", "w") as file:
                json.dump(history[-1], file)
    return history


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    load_dir = pathlib.Path(cfg.data.load_dir)

    device = get_default_device()
    logger.info(f"Current device is {device}")

    cfg_dct = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    registry = Registry()
    registry.add_from_module(src.datasets, prefix="src.datasets.")
    registry.add_from_module(src.models.single, prefix="src.models.single.")
    registry.add_from_module(src.models.stacked, prefix="src.models.stacked.")

    train_cfg = cfg_dct["dataset"].copy()
    train_cfg["annotation_file"] = load_dir / "train.csv"
    train_data = registry.get_from_params(**train_cfg)

    valid_cfg = cfg_dct["dataset"].copy()
    valid_cfg["annotation_file"] = load_dir / "valid.csv"
    valid_data = registry.get_from_params(**valid_cfg)

    train_dataloader = DataLoader(train_data, num_workers=4, batch_size=32, shuffle=True)
    valid_dataloader = DataLoader(valid_data, num_workers=4, batch_size=32, shuffle=False)

    train_dataloader = DeviceDataLoader(train_dataloader, device)
    valid_dataloader = DeviceDataLoader(valid_dataloader, device)

    model = registry.get_from_params(**cfg_dct["model"])
    to_device(model, device)

    with torch.no_grad():
        model.eval()
        history = [evaluate(model, valid_dataloader)]
    history += fit(cfg.training_params.num_epochs, 0.5, model, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    main()
