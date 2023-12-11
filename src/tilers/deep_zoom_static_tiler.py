import time
from multiprocessing import JoinableQueue

from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from src.tilers.deep_zoom_image_tiler import DeepZoomImageTiler
from src.tilers.tile_worker import TileWorker


class DeepZoomStaticTiler:
    """Handles generation of tiles and metadata for all images in a slide."""

    def __init__(
        self,
        slidepath,
        basename,
        format,
        tile_size,
        overlap,
        limit_bounds,
        quality,
        workers,
        with_viewer,
        Bkg,
        basenameJPG,
        xmlfile,
        mask_type,
        ROIpc,
        oLabel,
        ImgExtension,
        SaveMasks,
        Mag,
        normalize,
        Fieldxml,
        pixelsize,
        pixelsizerange,
        Adj_WindowSize,
        resize_ratio,
        Best_level,
        Adj_overlap,
        Std,
    ):

        self._slide = open_slide(slidepath)
        self._basename = basename
        self._basenameJPG = basenameJPG
        self._xmlfile = xmlfile
        self._mask_type = mask_type
        self._format = format
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._with_viewer = with_viewer
        self._Bkg = Bkg
        self._ROIpc = ROIpc
        self._dzi_data = {}
        self._xmlLabel = oLabel
        self._ImgExtension = ImgExtension
        self._SaveMasks = SaveMasks
        self._Mag = Mag
        self._normalize = normalize
        self._Fieldxml = Fieldxml
        self._pixelsize = pixelsize
        self._pixelsizerange = pixelsizerange
        self._rescale = False
        self._resize_ratio = resize_ratio
        self._Best_level = Best_level
        self._Adj_WindowSize = Adj_WindowSize
        self._Adj_overlap = Adj_overlap
        self._Std = Std
        for _ in range(workers):
            TileWorker(
                self._queue,
                slidepath,
                self._Adj_WindowSize,
                self._Adj_overlap,
                limit_bounds,
                quality,
                self._Bkg,
                self._ROIpc,
                self._Std,
            ).start()

    def run(self):
        self._run_image()
        self._shutdown()

    def _run_image(self):
        """Run a single image from self._slide."""
        dz = DeepZoomGenerator(self._slide, self._Adj_WindowSize, self._Adj_overlap, limit_bounds=self._limit_bounds)
        tiler = DeepZoomImageTiler(
            dz,
            self._basename,
            self._format,
            None,
            self._queue,
            self._slide,
            self._basenameJPG,
            self._xmlfile,
            self._mask_type,
            self._xmlLabel,
            self._ROIpc,
            self._ImgExtension,
            self._SaveMasks,
            self._Mag,
            self._normalize,
            self._Fieldxml,
            self._pixelsize,
            self._pixelsizerange,
            self._Best_level,
            self._resize_ratio,
            self._Adj_WindowSize,
            self._Adj_overlap,
        )

        time.sleep(3)
        tiler.run()

    def _shutdown(self):
        for _ in range(self._workers):
            self._queue.put(None)
        self._queue.join()
