from multiprocessing import Process

import numpy as np
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
import cv2

from src.tilers.utils import normalize_tile


class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds, quality, _Bkg, _ROIpc, _Std):
        Process.__init__(self, name="TileWorker")
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._quality = quality
        self._slide = None
        self._Bkg = _Bkg
        self._ROIpc = _ROIpc
        self._Std = _Std

    def _get_dz(self, wsize=299, overlap=0):
        image = self._slide
        return DeepZoomGenerator(image, wsize, overlap, limit_bounds=self._limit_bounds)

    def run(self):
        self._slide = open_slide(self._slidepath)
        dz = self._get_dz(self._tile_size, self._overlap)
        while True:
            data = self._queue.get()

            (
                associated,
                level,
                address,
                outfile,
                format,
                outfile_bw,
                PercentMasked,
                SaveMasks,
                TileMask,
                Normalize,
                isrescale,
                resize_ratio,
                Adj_WindowSize,
                Adj_overlap,
            ) = data
            if associated is None:
                dz = self._get_dz(Adj_WindowSize, Adj_overlap)
            try:
                tile = dz.get_tile(level, address)
                # A single tile is being read
                # check the percentage of the image with "information". Should be above 50%
                gray = tile.convert("L")

                img_grey = cv2.cvtColor(np.asarray(tile), cv2.COLOR_BGR2GRAY)
                St = cv2.Canny(img_grey, 1, 255).std()

                bw = gray.point(lambda x: 0 if x < 230 else 1, "F")
                avgBkg = np.average(bw)

                # do not save non-square tiles near the edges
                NbPx = (tile.height * tile.width) - (self._tile_size + 2 * self._overlap) * (
                    self._tile_size + 2 * self._overlap
                )

                if avgBkg <= (self._Bkg / 100.0) and St >= self._Std and NbPx == 0:
                    # if an Aperio selection was made, check if is within the selected region
                    if PercentMasked >= (self._ROIpc / 100.0):
                        if Normalize != "":
                            tile = Image.fromarray(normalize_tile(tile, Normalize).astype("uint8"), "RGB")

                        if (isrescale) and (resize_ratio != 1):
                            if resize_ratio > 1:
                                tile = tile.resize(
                                    (
                                        min(
                                            (self._tile_size + 2 * self._overlap), int(round(tile.width / resize_ratio))
                                        ),
                                        min(
                                            (self._tile_size + 2 * self._overlap),
                                            int(round(tile.height / resize_ratio)),
                                        ),
                                    )
                                )
                            else:
                                tile = tile.resize(
                                    (int(round(tile.width / resize_ratio)), int(round(tile.height / resize_ratio)))
                                )
                        tile.save(outfile, quality=self._quality)
                self._queue.task_done()
            except Exception:
                print("image %s failed at dz.get_tile for level %f" % (self._slidepath, level))
                self._queue.task_done()
