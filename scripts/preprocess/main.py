import os
import pathlib
import sys
from typing import List

import hydra
import openslide
from loguru import logger
from omegaconf import DictConfig

sys.path.append("../../")

import src


def get_file_path(dir: pathlib.Path, file_prefix: str) -> pathlib.Path:
    for file in os.listdir(dir):
        file_path = dir / file
        if file.startswith(file_prefix):
            return file_path


def get_slides_path(dir: pathlib.Path) -> List[pathlib.Path]:
    slides_path = []
    for file in os.listdir(dir / "raw"):
        file_path = dir / "raw" / file
        if os.path.isdir(file_path):
            slides_path.append(file_path)
    return slides_path


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    load_dir = pathlib.Path(cfg.data.load_dir)

    logger.info("Loading data")
    slides_path = get_slides_path(load_dir)
    biospecimen_path = get_file_path(load_dir, "biospecimen.cart")
    metadata_path = get_file_path(load_dir, "metadata.cart")

    tiler = src.tilers.DeepZoomStaticTiler(
        cfg.params.tile_size,
        cfg.params.overlap,
        cfg.params.quality,
        cfg.params.background_limit,
        cfg.params.limit_bounds,
    )

    logger.info("Tiling images")
    for _slide_dir in slides_path:
        _slide_path = get_file_path(_slide_dir, "TCGA")
        logger.info(f"Processing {_slide_path}")

        slide = openslide.open_slide(_slide_path)
        tiler.process(slide)


if __name__ == "__main__":
    main()
