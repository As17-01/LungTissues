import os
import pathlib

import hydra
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedShuffleSplit


def get_slide_names(load_dir: pathlib.Path):
    slide_names = []
    for file in os.listdir(load_dir):
        file_path = load_dir / file
        slide_names.append(file_path)
    return slide_names


def write_annotation_base(save_dir: pathlib.Path, data_set: pd.DataFrame, name: str) -> None:
    with open(save_dir / f"{name}.csv", "w") as ouf:
        for slide_dir, target in zip(data_set.slide, data_set.target):
            for image in os.listdir(slide_dir):
                line = slide_dir / image
                ouf.write("".join([str(line), ",", str(target), "\n"]))


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    load_dir = pathlib.Path(cfg.data.load_dir)
    save_dir = pathlib.Path(cfg.data.save_dir)

    logger.info("Loading data")
    tumor_slide_names = get_slide_names(load_dir / "tumor")
    normal_slide_names = get_slide_names(load_dir / "normal")

    annotation_base = pd.DataFrame(
        {
            "slide": tumor_slide_names + normal_slide_names,
            "target": np.ones(len(tumor_slide_names)).tolist() + np.zeros(len(normal_slide_names)).tolist(),
        }
    )

    test_size = 1 - cfg.experiment_params.train_size
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=cfg.experiment_params.state)
    for train_index, test_valid_index in split.split(annotation_base, annotation_base.target):
        train_set = annotation_base.iloc[train_index]
        test_valid_set = annotation_base.iloc[test_valid_index]

    test_size = cfg.experiment_params.val_size / test_size
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=cfg.experiment_params.state)
    for test_index, valid_index in split.split(test_valid_set, test_valid_set.target):
        valid_set = test_valid_set.iloc[valid_index]
        test_set = test_valid_set.iloc[test_index]

    write_annotation_base(save_dir, train_set, "train")
    write_annotation_base(save_dir, valid_set, "valid")
    write_annotation_base(save_dir, test_set, "test")


if __name__ == "__main__":
    main()
