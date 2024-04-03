import pathlib

import hydra
from loguru import logger
from omegaconf import DictConfig

from medmnist import PneumoniaMNIST

@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    save_dir = pathlib.Path(cfg.data.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    logger.info("Downloading train data")
    _ = PneumoniaMNIST("train", download=True)

    logger.info("Downloading val data")
    _ = PneumoniaMNIST("val", download=True)

    logger.info("Downloading test data")
    _ = PneumoniaMNIST("test", download=True)


if __name__ == "__main__":
    main()
