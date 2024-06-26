import json
import os
import pathlib

import hydra
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig


def get_slide_names(load_dir: pathlib.Path):
    slide_names = []
    for file in os.listdir(load_dir):
        file_path = load_dir / file
        if os.path.isdir(file_path):
            slide_names.append(file_path)
    return slide_names


def write_annotation(save_dir: pathlib.Path, data_set: pd.DataFrame, name: str) -> None:
    with open(save_dir / f"{name}.csv", "w") as ouf:
        for slide_dir, target in zip(data_set.slide, data_set.target):
            ouf.write("".join([str(slide_dir), ",", str(target), "\n"]))


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    load_dir = pathlib.Path(cfg.data.load_dir)
    metadata_dir = pathlib.Path(cfg.data.metadata_dir)
    save_dir = pathlib.Path(cfg.data.save_dir)

    logger.info("Loading slide names")
    slide_names = get_slide_names(load_dir)

    with open(metadata_dir / "metadata.cart.LUAD.json", "r") as ouf:
        luad_metadata = json.load(ouf)
    with open(metadata_dir / "metadata.cart.LUSC.json", "r") as ouf:
        lusc_metadata = json.load(ouf)

    target_list = []
    for slide in slide_names:
        for annotation in luad_metadata:
            if annotation["file_id"] == slide.name:
                target_list.append(1)

        for annotation in lusc_metadata:
            if annotation["file_id"] == slide.name:
                target_list.append(0)
    annotation = pd.DataFrame(
        {
            "slide": slide_names,
            "target": target_list,
        }
    )
    logger.info(f"Num slides total: {len(annotation)}")

    valid_index = np.random.choice(len(annotation), cfg.experiment_params.valid_size, replace=False)
    is_valid_index = annotation.index.isin(valid_index)
    valid_set = annotation.iloc[is_valid_index].copy()
    annotation = annotation.iloc[~is_valid_index].reset_index(drop=True)
    logger.info(f"Validation size: {len(valid_set)}")

    test_index = np.random.choice(len(annotation), cfg.experiment_params.test_size, replace=False)
    is_test_index = annotation.index.isin(test_index)
    test_set = annotation.iloc[is_test_index].copy()
    train_set = annotation.iloc[~is_test_index].reset_index(drop=True)
    logger.info(f"Test size: {len(test_set)}")
    logger.info(f"Train size: {len(train_set)}")

    write_annotation(save_dir, train_set, "train")
    write_annotation(save_dir, valid_set, "valid")
    write_annotation(save_dir, test_set, "test")


if __name__ == "__main__":
    main()
