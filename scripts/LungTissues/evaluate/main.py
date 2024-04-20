import pathlib
import sys

import hydra
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
    load_dir = pathlib.Path(cfg.data.load_dir)

    num_workers = cfg.eval_params.num_workers
    batch_size = cfg.eval_params.batch_size

    device = src.utils.get_default_device()
    logger.info(f"Current device is {device}")

    cfg_dct = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    registry = Registry()
    registry.add_from_module(src.datasets, prefix="src.datasets.")
    registry.add_from_module(src.models, prefix="src.models.")

    valid_cfg = cfg_dct["dataset"].copy()
    valid_cfg["annotation_file"] = load_dir / "valid.csv"
    valid_data = registry.get_from_params(**valid_cfg)

    test_cfg = cfg_dct["dataset"].copy()
    test_cfg["annotation_file"] = load_dir / "test.csv"
    test_data = registry.get_from_params(**test_cfg)

    valid_dataloader = DataLoader(valid_data, num_workers=num_workers, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, num_workers=num_workers, batch_size=batch_size, shuffle=False)

    valid_dataloader = src.utils.DeviceDataLoader(valid_dataloader, device)
    test_dataloader = src.utils.DeviceDataLoader(test_dataloader, device)

    model = registry.get_from_params(**cfg_dct["model"])
    model.load_state_dict(torch.load(cfg.data.saved_model_path))
    src.utils.to_device(model, device)

    with torch.no_grad():
        model.eval()
        history_val = src.utils.predict(model, valid_dataloader)
        history_test = src.utils.predict(model, test_dataloader)


if __name__ == "__main__":
    main()
