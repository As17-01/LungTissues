import hydra
from loguru import logger
from medmnist import PneumoniaMNIST
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    logger.info("Downloading data")
    _ = PneumoniaMNIST("train", download=True)


if __name__ == "__main__":
    main()
