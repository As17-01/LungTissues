import os
import pathlib
import queue
import sys
import time
from multiprocessing import Process
from multiprocessing import Queue
from typing import List

import hydra
import openslide
from loguru import logger
from omegaconf import DictConfig

sys.path.append("../../../")

from src.tilers import DeepZoomStaticTiler


def get_file_path(dir: pathlib.Path, file_prefix: str) -> pathlib.Path:
    for file in os.listdir(dir):
        file_path = dir / file
        if file.startswith(file_prefix):
            return file_path


def get_slides_path(load_dir: pathlib.Path, save_dir: pathlib.Path) -> List[pathlib.Path]:
    already_saved = []
    for file in os.listdir(save_dir):
        already_saved.append(file)

    slides_path = []
    for file in os.listdir(load_dir / "images"):
        if file not in already_saved:
            # TODO: remove second /file in the final version
            file_path = load_dir / "images" / file / file

            if os.path.isdir(file_path):
                slides_path.append(file_path)
    return slides_path


def process_slide(slide_queue):
    while True:
        try:
            tiler, slide_dir, save_dir = slide_queue.get_nowait()
        except queue.Empty:
            break
        else:
            file_name = pathlib.PurePath(slide_dir).name

            slide_path = get_file_path(slide_dir, "TCGA")
            logger.info(f"Processing {file_name}")

            slide = openslide.open_slide(slide_path)
            slide_save_dir = save_dir / file_name
            slide_save_dir.mkdir(exist_ok=True, parents=True)

            tiler.process(slide, slide_save_dir)
            logger.info(f"Done {file_name}")

            time.sleep(1)
    return True


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    load_dir = pathlib.Path(cfg.data.load_dir)
    save_dir = pathlib.Path(cfg.data.save_dir)

    logger.info("Loading data")
    slides_path = get_slides_path(load_dir, save_dir)
    logger.info(f"Num slides: {len(slides_path)}")

    tiler = DeepZoomStaticTiler(
        cfg.params.tile_size,
        cfg.params.overlap,
        cfg.params.quality,
        cfg.params.background_limit,
        cfg.params.limit_bounds,
    )

    slide_queue = Queue()
    logger.info("Preparing data")
    for slide_dir in slides_path:
        slide_queue.put((tiler, slide_dir, save_dir))

    logger.info("Tiling images")
    processes = []
    for _ in range(cfg.params.num_cpu):
        p = Process(target=process_slide, args=(slide_queue,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
