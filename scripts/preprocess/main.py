# import json
import os
# import re
# import shutil
# import subprocess
# import sys
# import time
# from glob import glob
# from multiprocessing import JoinableQueue
# from multiprocessing import Process
# from optparse import OptionParser
# from unicodedata import normalize
# from xml.dom import minidom
# import cv2
# import numpy as np
# import openslide
# import pydicom as dicom
# import scipy.misc
# from imageio import imread
# from imageio import imwrite as imsave
# from openslide import ImageSlide
# from openslide import open_slide
# from openslide.deepzoom import DeepZoomGenerator
# from PIL import Image
# from PIL import ImageCms
# from PIL import ImageDraw
# from skimage import color
# from skimage import io
# import csv
# import pandas as pd
import hydra
from omegaconf import DictConfig
import pathlib
from loguru import logger
from typing import List

# VIEWER_SLIDE_NAME = "slide"
# Image.MAX_IMAGE_PIXELS = None


def get_file_path(dir: pathlib.Path, file_prefix: str) -> pathlib.Path:
    for file in os.listdir(dir):
        file_path = dir / file
        if file.startswith(file_prefix): 
            return file_path

def get_images_path(dir: pathlib.Path) -> List[pathlib.Path]:
    images_path = []
    for file in os.listdir(dir):
        file_path = dir / file
        if os.path.isdir(file_path): 
            images_path.append(file_path)
    return images_path


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    load_dir = pathlib.Path(cfg.data.load_dir)

    logger.info("Loading data")
    images_path = get_images_path(load_dir)
    biospecimen_path = get_file_path(load_dir, "biospecimen.cart")
    metadata_path = get_file_path(load_dir, "metadata.cart")

    logger.info(images_path)
    logger.info(biospecimen_path)
    logger.info(metadata_path)

if __name__ == "__main__":
    main()

#     print(slidepath)
#     # get  mages from the data/ file.
#     files = glob(slidepath)
#     ImgExtension = slidepath.split(".")[-1]
#     # files
#     print("list of files:")
#     print(files)
#     print("***********************")

#     files = sorted(files)
#     for imgNb in range(len(files)):
#         filename = files[imgNb]
#         ## New WindowSize
#         if (opts.Mag <= 0) and (opts.pixelsizerange < 0):
#             slide = open_slide(filename)
#             # calculate the best window size before rescaling to reach desired final pizelsize
#             try:
#                 Objective = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
#                 OrgPixelSizeX = float(slide.properties["openslide.mpp-x"])
#                 OrgPixelSizeY = float(slide.properties["openslide.mpp-y"])
#             except:
#                 try:
#                     for nfields in slide.properties["tiff.ImageDescription"].split("|"):
#                         if "AppMag" in nfields:
#                             Objective = float(nfields.split(" = ")[1])
#                         if "MPP" in nfields:
#                             OrgPixelSizeX = float(nfields.split(" = ")[1])
#                             OrgPixelSizeY = OrgPixelSizeX
#                 except:
#                     print("Error: No information found in the header")
#                     continue
#             Desired_FoV_um = opts.pixelsize * opts.tile_size
#             AllPxSizes = [OrgPixelSizeX * pow(2, nn) for nn in range(0, 12)]
#             AllBoxSizes = [round(Desired_FoV_um / (OrgPixelSizeX * pow(2, nn))) for nn in range(0, 12)]
#             # Prevent resizing from smaller box size... unless target pixel size if below scanned pixelsize
#             if sum(np.array(AllBoxSizes) > 1196) > 0:
#                 for nn in range(0, 12):
#                     if AllBoxSizes[nn] < opts.tile_size:
#                         AllBoxSizes[nn] = 2000000
#                 # Final_pixel_size_Diff = [abs(AllBoxSizes[x] / opts.tile_size * AllPxSizes[x] - opts.pixelsize) for x in range(0,12)]
#                 # Final_pixel_size_Diff = [abs(AllBoxSizes[x] / (opts.tile_size + 2 * opts.overlap) * AllPxSizes[x] - opts.pixelsize) for x in range(0,12)]
#                 Final_pixel_size_Diff = [abs(AllBoxSizes[x] * AllPxSizes[x] - Desired_FoV_um) for x in range(0, 12)]
#                 print(AllPxSizes)
#                 print(AllBoxSizes)
#                 print(Final_pixel_size_Diff)
#                 Best_level = [
#                     index for index, value in enumerate(Final_pixel_size_Diff) if value == min(Final_pixel_size_Diff)
#                 ][-1]
#             else:
#                 Best_level = 0
#             Adj_WindowSize = AllBoxSizes[Best_level]
#             # dz = DeepZoomGenerator(image, Adj_WindowSize, opt.overlap,limit_bounds=opt.limit_bounds)
#             resize_ratio = float(Adj_WindowSize) / float(opts.tile_size)
#             if opts.overlap > 0:
#                 resize_ratio = ((resize_ratio) + (float(int(round(opts.overlap * resize_ratio))) / opts.overlap)) / 2
#             Adj_overlap = int(round(opts.overlap * resize_ratio))
#             print(
#                 "info: Objective:"
#                 + str(Objective)
#                 + "; OrgPixelSizeX"
#                 + str(OrgPixelSizeX)
#                 + "; Desired_FoV_um: "
#                 + str(Desired_FoV_um)
#                 + "; Best_level: "
#                 + str(Best_level)
#                 + "; resize_ratio: "
#                 + str(resize_ratio)
#                 + "; Adj_WindowSize:"
#                 + str(Adj_WindowSize)
#                 + "; self._tile_size: "
#                 + str(opts.tile_size),
#                 "; opts.overlap:",
#                 str(opts.overlap),
#                 "; Adj_overlap:",
#                 str(Adj_overlap),
#             )
#         else:
#             Best_level = -1
#             resize_ratio = 1
#             # Adj_WindowSize = self._tile_size
#             Adj_WindowSize = opts.tile_size
#             Adj_overlap = opts.overlap

#         # print(filename)
#         opts.basenameJPG = os.path.splitext(os.path.basename(filename))[0]
#         print("processing: " + opts.basenameJPG + " with extension: " + ImgExtension)

#         if "dcm" in ImgExtension:
#             print("convert %s dcm to jpg" % filename)
#             if opts.tmp_dcm is None:
#                 parser.error("Missing output folder for dcm>jpg intermediate files")
#             elif not os.path.isdir(opts.tmp_dcm):
#                 parser.error("Missing output folder for dcm>jpg intermediate files")

#             if filename[-3:] == "jpg":
#                 continue
#             ImageFile = dicom.read_file(filename)
#             im1 = ImageFile.pixel_array
#             maxVal = float(im1.max())
#             minVal = float(im1.min())
#             height = im1.shape[0]
#             width = im1.shape[1]
#             depth = im1.shape[2]
#             print(height, width, depth, minVal, maxVal)
#             if opts.srh == 1:
#                 print(filename)
#                 imgn = os.path.splitext(os.path.basename(filename))[0].split("_")[0]
#                 foldname1 = filename.split("/")[-2]
#                 foldname2 = filename.split("/")[-3]
#                 opts.basenameJPG = foldname2 + "_" + imgn + "_" + foldname1
#                 image = np.zeros((width, depth, 3), "uint8")
#                 image = im1
#             else:
#                 image = np.zeros((height, width, 3), "uint8")
#                 image[..., 0] = ((im1[:, :].astype(float) - minVal) / (maxVal - minVal) * 255.0).astype(int)
#                 image[..., 1] = ((im1[:, :].astype(float) - minVal) / (maxVal - minVal) * 255.0).astype(int)
#                 image[..., 2] = ((im1[:, :].astype(float) - minVal) / (maxVal - minVal) * 255.0).astype(int)

#             filename = os.path.join(opts.tmp_dcm, opts.basenameJPG + ".jpg")
#             imsave(filename, image)

#             output = os.path.join(opts.basename, opts.basenameJPG)

#             try:
#                 DeepZoomStaticTiler(
#                     filename,
#                     output,
#                     opts.format,
#                     opts.tile_size,
#                     opts.overlap,
#                     opts.limit_bounds,
#                     opts.quality,
#                     opts.workers,
#                     opts.with_viewer,
#                     opts.Bkg,
#                     opts.basenameJPG,
#                     opts.xmlfile,
#                     opts.mask_type,
#                     opts.ROIpc,
#                     "",
#                     ImgExtension,
#                     opts.SaveMasks,
#                     opts.Mag,
#                     opts.normalize,
#                     opts.Fieldxml,
#                     opts.pixelsize,
#                     opts.pixelsizerange,
#                     Adj_WindowSize,
#                     resize_ratio,
#                     Best_level,
#                     Adj_overlap,
#                     opts.Std,
#                 ).run()
#             except Exception as e:
#                 print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))
#                 print(e)

#         elif opts.xmlfile != "":
#             # Check if Aperio or Qupath annotations
#             xmldir = os.path.join(opts.xmlfile, opts.basenameJPG + ".xml")
#             jsondir = os.path.join(opts.xmlfile, opts.basenameJPG + ".json")
#             csvdir = os.path.join(opts.xmlfile, opts.basenameJPG + ".csv")
#             tifdir = os.path.join(opts.xmlfile, opts.basenameJPG + ".tif")
#             geojsondir = os.path.join(opts.xmlfile, opts.basenameJPG + ".geojson")
#             # print("xml:")
#             # print(xmldir)
#             if (
#                 os.path.isfile(xmldir)
#                 | os.path.isfile(jsondir)
#                 | os.path.isfile(csvdir)
#                 | os.path.isfile(tifdir)
#                 | os.path.isfile(geojsondir)
#             ):
#                 if os.path.isfile(xmldir):
#                     AnnotationMode = "Aperio"
#                 elif os.path.isfile(jsondir):
#                     AnnotationMode = "QuPath"
#                     xmldir = jsondir
#                 elif os.path.isfile(csvdir):
#                     AnnotationMode = "Omero"
#                     xmldir = csvdir
#                 elif os.path.isfile(tifdir):
#                     AnnotationMode = "ImageJ"
#                     xmldir = tifdir
#                 elif os.path.isfile(geojsondir):
#                     AnnotationMode = "QuPath_orig"
#                     xmldir = geojsondir
#                 if (opts.mask_type == 1) or (opts.oLabelref != ""):
#                     # either mask inside ROI, or mask outside but a reference label exist
#                     xml_labels, xml_valid = xml_read_labels(xmldir, opts.Fieldxml, AnnotationMode)
#                     if opts.mask_type == 1:
#                         # No inverse mask
#                         Nbr_ROIs_ForNegLabel = 1
#                     elif opts.oLabelref != "":
#                         # Inverse mask and a label reference exist
#                         Nbr_ROIs_ForNegLabel = 0

#                     for oLabel in xml_labels:
#                         # print("label is %s and ref is %s" % (oLabel, opts.oLabelref))
#                         if (opts.oLabelref in oLabel) or (opts.oLabelref == ""):
#                             # is a label is identified
#                             if opts.mask_type == 0:
#                                 # Inverse mask and label exist in the image
#                                 Nbr_ROIs_ForNegLabel += 1
#                                 # there is a label, and map is to be inverted
#                                 output = os.path.join(opts.basename, oLabel + "_inv", opts.basenameJPG)
#                                 if not os.path.exists(os.path.join(opts.basename, oLabel + "_inv")):
#                                     os.makedirs(os.path.join(opts.basename, oLabel + "_inv"))
#                             else:
#                                 Nbr_ROIs_ForNegLabel += 1
#                                 output = os.path.join(opts.basename, oLabel, opts.basenameJPG)
#                                 if not os.path.exists(os.path.join(opts.basename, oLabel)):
#                                     os.makedirs(os.path.join(opts.basename, oLabel))
#                             if 1:
#                                 # try:
#                                 DeepZoomStaticTiler(
#                                     filename,
#                                     output,
#                                     opts.format,
#                                     opts.tile_size,
#                                     opts.overlap,
#                                     opts.limit_bounds,
#                                     opts.quality,
#                                     opts.workers,
#                                     opts.with_viewer,
#                                     opts.Bkg,
#                                     opts.basenameJPG,
#                                     opts.xmlfile,
#                                     opts.mask_type,
#                                     opts.ROIpc,
#                                     oLabel,
#                                     ImgExtension,
#                                     opts.SaveMasks,
#                                     opts.Mag,
#                                     opts.normalize,
#                                     opts.Fieldxml,
#                                     opts.pixelsize,
#                                     opts.pixelsizerange,
#                                     Adj_WindowSize,
#                                     resize_ratio,
#                                     Best_level,
#                                     Adj_overlap,
#                                     opts.Std,
#                                 ).run()
#                             # except:
#                             # 	print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))
#                         if Nbr_ROIs_ForNegLabel == 0:
#                             print("label %s is not in that image; invert everything" % (opts.oLabelref))
#                             # a label ref was given, and inverse mask is required but no ROI with this label in that map --> take everything
#                             oLabel = opts.oLabelref
#                             output = os.path.join(opts.basename, opts.oLabelref + "_inv", opts.basenameJPG)
#                             if not os.path.exists(os.path.join(opts.basename, oLabel + "_inv")):
#                                 os.makedirs(os.path.join(opts.basename, oLabel + "_inv"))
#                             if 1:
#                                 DeepZoomStaticTiler(
#                                     filename,
#                                     output,
#                                     opts.format,
#                                     opts.tile_size,
#                                     opts.overlap,
#                                     opts.limit_bounds,
#                                     opts.quality,
#                                     opts.workers,
#                                     opts.with_viewer,
#                                     opts.Bkg,
#                                     opts.basenameJPG,
#                                     opts.xmlfile,
#                                     opts.mask_type,
#                                     opts.ROIpc,
#                                     oLabel,
#                                     ImgExtension,
#                                     opts.SaveMasks,
#                                     opts.Mag,
#                                     opts.normalize,
#                                     opts.Fieldxml,
#                                     opts.pixelsize,
#                                     opts.pixelsizerange,
#                                     Adj_WindowSize,
#                                     resize_ratio,
#                                     Best_level,
#                                     Adj_overlap,
#                                     opts.Std,
#                                 ).run()

#                 else:
#                     # Background
#                     oLabel = "non_selected_regions"
#                     output = os.path.join(opts.basename, oLabel, opts.basenameJPG)
#                     if not os.path.exists(os.path.join(opts.basename, oLabel)):
#                         os.makedirs(os.path.join(opts.basename, oLabel))
#                     try:
#                         DeepZoomStaticTiler(
#                             filename,
#                             output,
#                             opts.format,
#                             opts.tile_size,
#                             opts.overlap,
#                             opts.limit_bounds,
#                             opts.quality,
#                             opts.workers,
#                             opts.with_viewer,
#                             opts.Bkg,
#                             opts.basenameJPG,
#                             opts.xmlfile,
#                             opts.mask_type,
#                             opts.ROIpc,
#                             oLabel,
#                             ImgExtension,
#                             opts.SaveMasks,
#                             opts.Mag,
#                             opts.normalize,
#                             opts.Fieldxml,
#                             opts.pixelsize,
#                             opts.pixelsizerange,
#                             Adj_WindowSize,
#                             resize_ratio,
#                             Best_level,
#                             Adj_overlap,
#                             opts.Std,
#                         ).run()
#                     except Exception as e:
#                         print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))
#                         print(e)

#             else:
#                 if (ImgExtension == ".jpg") | (ImgExtension == ".dcm"):
#                     print("Input image to be tiled is jpg or dcm and not svs - will be treated as such")
#                     output = os.path.join(opts.basename, opts.basenameJPG)
#                     try:
#                         DeepZoomStaticTiler(
#                             filename,
#                             output,
#                             opts.format,
#                             opts.tile_size,
#                             opts.overlap,
#                             opts.limit_bounds,
#                             opts.quality,
#                             opts.workers,
#                             opts.with_viewer,
#                             opts.Bkg,
#                             opts.basenameJPG,
#                             opts.xmlfile,
#                             opts.mask_type,
#                             opts.ROIpc,
#                             "",
#                             ImgExtension,
#                             opts.SaveMasks,
#                             opts.Mag,
#                             opts.normalize,
#                             opts.Fieldxml,
#                             opts.pixelsize,
#                             opts.pixelsizerange,
#                             Adj_WindowSize,
#                             resize_ratio,
#                             Best_level,
#                             Adj_overlap,
#                             opts.Std,
#                         ).run()
#                     except Exception as e:
#                         print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))
#                         print(e)

#                 else:
#                     print(
#                         "No xml file found for slide %s.svs (expected: %s). Directory or xml file does not exist"
#                         % (opts.basenameJPG, xmldir)
#                     )
#                     continue
#         else:
#             print("start tiling")
#             output = os.path.join(opts.basename, opts.basenameJPG)
#             if os.path.exists(output + "_files"):
#                 print("Image %s already tiled" % opts.basenameJPG)
#                 continue

#             DeepZoomStaticTiler(
#                 filename,
#                 output,
#                 opts.format,
#                 opts.tile_size,
#                 opts.overlap,
#                 opts.limit_bounds,
#                 opts.quality,
#                 opts.workers,
#                 opts.with_viewer,
#                 opts.Bkg,
#                 opts.basenameJPG,
#                 opts.xmlfile,
#                 opts.mask_type,
#                 opts.ROIpc,
#                 "",
#                 ImgExtension,
#                 opts.SaveMasks,
#                 opts.Mag,
#                 opts.normalize,
#                 opts.Fieldxml,
#                 opts.pixelsize,
#                 opts.pixelsizerange,
#                 Adj_WindowSize,
#                 resize_ratio,
#                 Best_level,
#                 Adj_overlap,
#                 opts.Std,
#             ).run()

#     print("End")


# class TileWorker(Process):
#     """A child process that generates and writes tiles."""

#     def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds, quality, _Bkg, _ROIpc, _Std):
#         Process.__init__(self, name="TileWorker")
#         self._queue = queue
#         self._slidepath = slidepath
#         self._tile_size = tile_size
#         self._overlap = overlap
#         self._limit_bounds = limit_bounds
#         self._quality = quality
#         self._slide = None
#         self._Bkg = _Bkg
#         self._ROIpc = _ROIpc
#         self._Std = _Std

#     def RGB_to_lab(self, tile):
#         Lab = color.rgb2lab(tile)
#         return Lab

#     def Lab_to_RGB(self, Lab):
#         newtile = (color.lab2rgb(Lab) * 255).astype(np.uint8)
#         return newtile

#     def normalize_tile(self, tile, NormVec):
#         Lab = self.RGB_to_lab(tile)
#         TileMean = [0, 0, 0]
#         TileStd = [1, 1, 1]
#         newMean = NormVec[0:3]
#         newStd = NormVec[3:6]
#         for i in range(3):
#             TileMean[i] = np.mean(Lab[:, :, i])
#             TileStd[i] = np.std(Lab[:, :, i])
#             tmp = ((Lab[:, :, i] - TileMean[i]) * (newStd[i] / TileStd[i])) + newMean[i]
#             if i == 0:
#                 tmp[tmp < 0] = 0
#                 tmp[tmp > 100] = 100
#                 Lab[:, :, i] = tmp
#             else:
#                 tmp[tmp < -128] = 128
#                 tmp[tmp > 127] = 127
#                 Lab[:, :, i] = tmp
#         tile = self.Lab_to_RGB(Lab)
#         return tile

#     def run(self):
#         self._slide = open_slide(self._slidepath)
#         last_associated = None
#         dz = self._get_dz(None, self._tile_size, self._overlap)
#         while True:
#             data = self._queue.get()
#             if data is None:
#                 self._queue.task_done()
#                 break

#             (
#                 associated,
#                 level,
#                 address,
#                 outfile,
#                 format,
#                 outfile_bw,
#                 PercentMasked,
#                 SaveMasks,
#                 TileMask,
#                 Normalize,
#                 isrescale,
#                 resize_ratio,
#                 Adj_WindowSize,
#                 Adj_overlap,
#             ) = data
#             if last_associated != associated:
#                 dz = self._get_dz(associated, Adj_WindowSize, Adj_overlap)
#                 last_associated = associated
#             try:
#                 tile = dz.get_tile(level, address)
#                 # A single tile is being read
#                 # check the percentage of the image with "information". Should be above 50%
#                 gray = tile.convert("L")
#                 img_grey = cv2.cvtColor(np.asarray(tile), cv2.COLOR_BGR2GRAY)
#                 St = cv2.Canny(img_grey, 1, 255).std()
#                 bw = gray.point(lambda x: 0 if x < 230 else 1, "F")
#                 arr = np.array(np.asarray(bw))
#                 avgBkg = np.average(bw)
#                 bw = gray.point(lambda x: 0 if x < 230 else 1, "1")
#                 # do not save non-square tiles nearr the edges
#                 print(tile.height, tile.width, self._tile_size + 2 * self._overlap)
#                 NbPx = (tile.height * tile.width) - (self._tile_size + 2 * self._overlap) * (
#                     self._tile_size + 2 * self._overlap
#                 )
#                 # check if the image is mostly background
#                 print(
#                     "res: "
#                     + outfile
#                     + " is "
#                     + str(avgBkg)
#                     + " and std "
#                     + str(St)
#                     + " (threshold: "
#                     + str(self._Bkg / 100.0)
#                     + " and "
#                     + str(self._Std)
#                     + " and "
#                     + str(NbPx)
#                     + ") PercentMasked: %.6f, %.6f" % (PercentMasked, self._ROIpc / 100.0)
#                 )
#                 if avgBkg <= (self._Bkg / 100.0) and St >= self._Std and NbPx == 0:
#                     # if an Aperio selection was made, check if is within the selected region
#                     if PercentMasked >= (self._ROIpc / 100.0):
#                         if Normalize != "":
#                             tile = Image.fromarray(self.normalize_tile(tile, Normalize).astype("uint8"), "RGB")

#                         if (isrescale) and (resize_ratio != 1):
#                             print(
#                                 self._tile_size,
#                                 tile.width,
#                                 resize_ratio,
#                                 tile.height,
#                                 int(round(tile.width / resize_ratio)),
#                                 int(round(tile.height / resize_ratio)),
#                                 min((self._tile_size + 2 * self._overlap), int(round(tile.width / resize_ratio))),
#                                 min((self._tile_size + 2 * self._overlap), int(round(tile.height / resize_ratio))),
#                             )
#                             if resize_ratio > 1:
#                                 tile = tile.resize(
#                                     (
#                                         min(
#                                             (self._tile_size + 2 * self._overlap), int(round(tile.width / resize_ratio))
#                                         ),
#                                         min(
#                                             (self._tile_size + 2 * self._overlap),
#                                             int(round(tile.height / resize_ratio)),
#                                         ),
#                                     )
#                                 )
#                             else:
#                                 tile = tile.resize(
#                                     (int(round(tile.width / resize_ratio)), int(round(tile.height / resize_ratio)))
#                                 )

#                         tile.save(outfile, quality=self._quality)
#                         if bool(SaveMasks) == True:
#                             height = TileMask.shape[0]
#                             width = TileMask.shape[1]
#                             TileMaskO = np.zeros((height, width, 3), "uint8")
#                             maxVal = float(TileMask.max())
#                             if maxVal == 0:
#                                 maxVal = 1
#                             TileMaskO[..., 0] = (TileMask[:, :].astype(float) / maxVal * 255.0).astype(int)
#                             TileMaskO[..., 1] = (TileMask[:, :].astype(float) / maxVal * 255.0).astype(int)
#                             TileMaskO[..., 2] = (TileMask[:, :].astype(float) / maxVal * 255.0).astype(int)
#                             TileMaskO = np.array(Image.fromarray(TileMaskO).resize((arr.shape[0], arr.shape[1])))
#                             TileMaskO[TileMaskO < 10] = 0
#                             TileMaskO[TileMaskO >= 10] = 255
#                             imsave(
#                                 outfile_bw + str(PercentMasked) + str(".jpg"), TileMaskO
#                             )  # (outfile_bw, quality=self._quality)

#                 self._queue.task_done()
#             except Exception as e:
#                 print("image %s failed at dz.get_tile for level %f" % (self._slidepath, level))
#                 print(e)
#                 self._queue.task_done()

#     def _get_dz(self, associated=None, wsize=299, overlap=0):
#         if associated is not None:
#             image = ImageSlide(self._slide.associated_images[associated])
#         else:
#             image = self._slide
#         print("wsize and overlap: ", str(wsize), " ", str(overlap))
#         return DeepZoomGenerator(image, wsize, overlap, limit_bounds=self._limit_bounds)


# class DeepZoomImageTiler(object):
#     """Handles generation of tiles and metadata for a single image."""

#     def __init__(
#         self,
#         dz,
#         basename,
#         format,
#         associated,
#         queue,
#         slide,
#         basenameJPG,
#         xmlfile,
#         mask_type,
#         xmlLabel,
#         ROIpc,
#         ImgExtension,
#         SaveMasks,
#         Mag,
#         normalize,
#         Fieldxml,
#         pixelsize,
#         pixelsizerange,
#         Best_level,
#         resize_ratio,
#         Adj_WindowSize,
#         Adj_overlap,
#     ):
#         self._dz = dz
#         self._basename = basename
#         self._basenameJPG = basenameJPG
#         self._format = format
#         self._associated = associated
#         self._queue = queue
#         self._processed = 0
#         self._slide = slide
#         self._xmlfile = xmlfile
#         self._mask_type = mask_type
#         self._xmlLabel = xmlLabel
#         self._ROIpc = ROIpc
#         self._ImgExtension = ImgExtension
#         self._SaveMasks = SaveMasks
#         self._Mag = Mag
#         self._normalize = normalize
#         self._Fieldxml = Fieldxml
#         self._pixelsize = pixelsize
#         self._pixelsizerange = pixelsizerange
#         self._Best_level = Best_level
#         self._resize_ratio = resize_ratio
#         self._Adj_WindowSize = Adj_WindowSize
#         self._Adj_overlap = Adj_overlap

#     def run(self):
#         self._write_tiles()
#         self._write_dzi()

#     def _write_tiles(self):
#         ########################################3
#         # nc_added
#         Magnification = 20
#         tol = 2
#         # get slide dimensions, zoom levels, and objective information
#         Factors = self._slide.level_downsamples
#         try:
#             Objective = float(self._slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
#         except:
#             try:
#                 for nfields in _slide.properties["tiff.ImageDescription"].split("|"):
#                     if "AppMag" in nfields:
#                         Objective = float(nfields.split(" = ")[1])
#             except:
#                 print(self._basename + " - No Obj information found")
#                 print(self._ImgExtension)
#                 if ("jpg" in self._ImgExtension) | ("dcm" in self._ImgExtension) | ("tif" in self._ImgExtension):
#                     # Objective = self._ROIpc
#                     Objective = 1.0
#                     Magnification = Objective
#                     print("input is jpg - will be tiled as such with %f" % Objective)
#                 else:
#                     return
#         # calculate magnifications
#         Available = tuple(Objective / x for x in Factors)
#         # find highest magnification greater than or equal to 'Desired'
#         Mismatch = tuple(x - Magnification for x in Available)
#         AbsMismatch = tuple(abs(x) for x in Mismatch)
#         if len(AbsMismatch) < 1:
#             print(self._basename + " - Objective field empty!")
#             return
#         """
#         if(min(AbsMismatch) <= tol):
#             Level = int(AbsMismatch.index(min(AbsMismatch)))
#             Factor = 1
#         else: #pick next highest level, downsample
#             Level = int(max([i for (i, val) in enumerate(Mismatch) if val > 0]))
#             Factor = Magnification / Available[Level]
#         # end added
#         """
#         xml_valid = False
#         # a dir was provided for xml files

#         """
#         ImgID = os.path.basename(self._basename)
#         Nbr_of_masks = 0
#         if self._xmlfile != '':
#             xmldir = os.path.join(self._xmlfile, ImgID + '.xml')
#             print("xml:")
#             print(xmldir)
#             if os.path.isfile(xmldir):
#                 xml_labels, xml_valid = self.xml_read_labels(xmldir)
#                 Nbr_of_masks = len(xml_labels)
#             else:
#                 print("No xml file found for slide %s.svs (expected: %s). Directory or xml file does not exist" %  (ImgID, xmldir) )
#                 return
#         else:
#             Nbr_of_masks = 1
#         """

#         if True:
#             ImgID = os.path.basename(self._basename)
#             if os.path.isfile(os.path.join(self._xmlfile, ImgID + ".xml")):
#                 # If path exists, Aperio assumed
#                 xmldir = os.path.join(self._xmlfile, ImgID + ".xml")
#                 AnnotationMode = "Aperio"
#             elif os.path.isfile(os.path.join(self._xmlfile, ImgID + ".geojson")):
#                 xmldir = os.path.join(self._xmlfile, ImgID + ".geojson")
#                 AnnotationMode = "QuPath_orig"
#             elif os.path.isfile(os.path.join(self._xmlfile, ImgID + ".json")):
#                 # QuPath assumed
#                 xmldir = os.path.join(self._xmlfile, ImgID + ".json")
#                 AnnotationMode = "QuPath"
#             elif os.path.isfile(os.path.join(self._xmlfile, ImgID + ".csv")):
#                 xmldir = os.path.join(self._xmlfile, ImgID + ".csv")
#                 AnnotationMode = "Omero"
#             elif os.path.isfile(os.path.join(self._xmlfile, ImgID + ".tif")):
#                 xmldir = os.path.join(self._xmlfile, ImgID + ".tif")
#                 AnnotationMode = "ImageJ"
#             else:
#                 AnnotationMode = "None"

#             if AnnotationMode == "ImageJ":
#                 mask, xml_valid, Img_Fact = self.jpg_mask_read(xmldir)
#                 if xml_valid == False:
#                     print("Error: xml %s file cannot be read properly - please check format" % xmldir)
#                     return
#             elif (self._xmlfile != "") & (self._ImgExtension != "jpg") & (self._ImgExtension != "dcm"):
#                 mask, xml_valid, Img_Fact = self.xml_read(
#                     xmldir, self._xmlLabel, self._Fieldxml, AnnotationMode, self._ImgExtension
#                 )
#                 if xml_valid == False:
#                     print("Error: xml %s file cannot be read properly - please check format" % xmldir)
#                     return
#             elif (self._xmlfile != "") & (self._ImgExtension == "dcm"):
#                 mask, xml_valid, Img_Fact = self.jpg_mask_read(xmldir)
#                 if xml_valid == False:
#                     print("Error: xml %s file cannot be read properly - please check format" % xmldir)
#                     return

#             print("111 :" + str(self._Mag) + " and pixelsize:" + str(self._pixelsize))
#             if self._Mag <= 0:
#                 if self._pixelsize > 0:
#                     level_range = [level for level in range(self._dz.level_count - 1, -1, -1)]
#                     # print(self._slide.properties)
#                     try:
#                         OrgPixelSizeX = float(self._slide.properties["openslide.mpp-x"])
#                         OrgPixelSizeY = float(self._slide.properties["openslide.mpp-y"])
#                     except:
#                         try:
#                             for nfields in slide.properties["tiff.ImageDescription"].split("|"):
#                                 if "MPP" in nfields:
#                                     OrgPixelSizeX = float(nfields.split(" = ")[1])
#                                     OrgPixelSizeY = OrgPixelSizeX
#                         except:
#                             print("Error: no pixelsize found in the header of %s" % self._basename)
#                             DesiredLevel = -1
#                             return
#                     AllPixelSizeDiffX = [
#                         (abs(OrgPixelSizeX * pow(2, self._dz.level_count - (level + 1)) - self._pixelsize))
#                         for level in range(self._dz.level_count - 1, -1, -1)
#                     ]
#                     AllPixelSizeDiffY = [
#                         (abs(OrgPixelSizeY * pow(2, self._dz.level_count - (level + 1)) - self._pixelsize))
#                         for level in range(self._dz.level_count - 1, -1, -1)
#                     ]
#                     IndxX = AllPixelSizeDiffX.index(min(AllPixelSizeDiffX))
#                     IndxY = AllPixelSizeDiffY.index(min(AllPixelSizeDiffY))
#                     levelX = AllPixelSizeDiffX[IndxX]
#                     levelY = AllPixelSizeDiffY[IndxY]
#                     if IndxX != IndxY:
#                         print("Error: X and Y pixel sizes are too different for %s" % self._basename)
#                         return
#                     if (levelX > self._pixelsizerange) and (self._pixelsizerange >= 0):
#                         print("Error: no pixelsize within the desired range for %s" % self._basename)
#                         return
#                     if (levelY > self._pixelsizerange) and (self._pixelsizerange >= 0):
#                         print("Error: no pixelsize within the desired range for %s" % self._basename)
#                         return
#                     if self._pixelsizerange < 0:
#                         level_range = [level for level in range(self._dz.level_count - 1, -1, -1)]
#                         IndxX = self._Best_level
#                     DesiredLevel = level_range[IndxX]
#                     print("**info: OrgPixelSizeX:" + str(OrgPixelSizeX) + "; DesiredLevel:" + str(DesiredLevel))
#                     if not os.path.exists(("/".join(self._basename.split("/")[:-1]))):
#                         os.makedirs(("/".join(self._basename.split("/")[:-1])))
#                     with open(
#                         os.path.join(("/".join(self._basename.split("/")[:-1])), "pixelsizes.txt"), "a"
#                     ) as file_out:
#                         file_out.write(
#                             self._basenameJPG
#                             + "\t"
#                             + str(OrgPixelSizeX * pow(2, IndxX))
#                             + "\t"
#                             + str(OrgPixelSizeX * pow(2, IndxX) * self._resize_ratio)
#                             + "\n"
#                         )

#             print(range(self._dz.level_count - 1, -1, -1))
#             for level in range(self._dz.level_count - 1, -1, -1):
#                 ThisMag = Available[0] / pow(2, self._dz.level_count - (level + 1))
#                 if self._Mag > 0:
#                     if ThisMag != self._Mag:
#                         continue
#                 elif self._pixelsize > 0:
#                     if level != DesiredLevel:
#                         continue
#                     else:
#                         tiledir_pixel = os.path.join("%s_files" % self._basename, str(self._pixelsize))

#                 ########################################
#                 tiledir = os.path.join("%s_files" % self._basename, str(ThisMag))
#                 if not os.path.exists(tiledir):
#                     os.makedirs(tiledir)
#                     if self._pixelsize > 0:
#                         os.symlink(str(ThisMag), tiledir_pixel, target_is_directory=True)
#                 cols, rows = self._dz.level_tiles[level]
#                 if xml_valid:
#                     # print("xml valid")
#                     """# If xml file is used, check for each tile what are their corresponding coordinate in the base image
#                     IndX_orig, IndY_orig = self._dz.level_tiles[-1]
#                     CurrentLevel_ReductionFactor = (Img_Fact * float(self._dz.level_dimensions[-1][0]) / float(self._dz.level_dimensions[level][0]))
#                     startIndX_current_level_conv = [int(i * CurrentLevel_ReductionFactor) for i in range(cols)]
#                     print("***********")
#                     endIndX_current_level_conv = [int(i * CurrentLevel_ReductionFactor) for i in range(cols)]
#                     endIndX_current_level_conv.append(self._dz.level_dimensions[level][0])
#                     endIndX_current_level_conv.pop(0)

#                     startIndY_current_level_conv = [int(i * CurrentLevel_ReductionFactor) for i in range(rows)]
#                     #endIndX_current_level_conv = [i * CurrentLevel_ReductionFactor - 1 for i in range(rows)]
#                     endIndY_current_level_conv = [int(i * CurrentLevel_ReductionFactor) for i in range(rows)]
#                     endIndY_current_level_conv.append(self._dz.level_dimensions[level][1])
#                     endIndY_current_level_conv.pop(0)
#                     """

#                 for row in range(rows):
#                     for col in range(cols):
#                         InsertBaseName = False
#                         if InsertBaseName:
#                             tilename = os.path.join(
#                                 tiledir, "%s_%d_%d.%s" % (self._basenameJPG, col, row, self._format)
#                             )
#                             tilename_bw = os.path.join(
#                                 tiledir, "%s_%d_%d_mask.%s" % (self._basenameJPG, col, row, self._format)
#                             )
#                         else:
#                             tilename = os.path.join(tiledir, "%d_%d.%s" % (col, row, self._format))
#                             tilename_bw = os.path.join(tiledir, "%d_%d_mask.%s" % (col, row, self._format))
#                         if xml_valid:

#                             Dlocation, Dlevel, Dsize = self._dz.get_tile_coordinates(level, (col, row))
#                             if self._ImgExtension == "mrxs":
#                                 print(Dlocation, Dlevel, Dsize, level, col, row)
#                                 aa, bb, cc = self._dz.get_tile_coordinates(level, (0, 0))
#                                 Dlocation = tuple(map(lambda i, j: i - j, Dlocation, aa))
#                                 print(Dlocation, Dlevel, Dsize, level, col, row)
#                             Ddimension = tuple(
#                                 [
#                                     pow(2, (self._dz.level_count - 1 - level)) * x
#                                     for x in self._dz.get_tile_dimensions(level, (col, row))
#                                 ]
#                             )
#                             startIndY_current_level_conv = int((Dlocation[1]) / Img_Fact)
#                             endIndY_current_level_conv = int((Dlocation[1] + Ddimension[1]) / Img_Fact)
#                             startIndX_current_level_conv = int((Dlocation[0]) / Img_Fact)
#                             endIndX_current_level_conv = int((Dlocation[0] + Ddimension[0]) / Img_Fact)

#                             if self._ImgExtension == "scn":
#                                 startIndY_current_level_conv = int(
#                                     ((Dlocation[1]) - self._dz.get_tile_coordinates(level, (0, 0))[0][1]) / Img_Fact
#                                 )
#                                 endIndY_current_level_conv = int(
#                                     (
#                                         (Dlocation[1] + Ddimension[1])
#                                         - self._dz.get_tile_coordinates(level, (0, 0))[0][1]
#                                     )
#                                     / Img_Fact
#                                 )
#                                 startIndX_current_level_conv = int(
#                                     ((Dlocation[0]) - self._dz.get_tile_coordinates(level, (0, 0))[0][0]) / Img_Fact
#                                 )
#                                 endIndX_current_level_conv = int(
#                                     (
#                                         (Dlocation[0] + Ddimension[0])
#                                         - self._dz.get_tile_coordinates(level, (0, 0))[0][0]
#                                     )
#                                     / Img_Fact
#                                 )

#                             TileMask = mask[
#                                 startIndY_current_level_conv:endIndY_current_level_conv,
#                                 startIndX_current_level_conv:endIndX_current_level_conv,
#                             ]
#                             PercentMasked = mask[
#                                 startIndY_current_level_conv:endIndY_current_level_conv,
#                                 startIndX_current_level_conv:endIndX_current_level_conv,
#                             ].mean()

#                             if self._mask_type == 0:
#                                 # keep ROI outside of the mask
#                                 PercentMasked = 1.0 - PercentMasked

#                             if PercentMasked > 0:
#                                 print("PercentMasked_p %.4f" % (PercentMasked))
#                             else:
#                                 print("PercentMasked_0 %.4f" % (PercentMasked))

#                         else:
#                             PercentMasked = 1.0
#                             TileMask = []

#                         if not os.path.exists(tilename):
#                             if self._Best_level == -1:
#                                 self._queue.put(
#                                     (
#                                         self._associated,
#                                         level,
#                                         (col, row),
#                                         tilename,
#                                         self._format,
#                                         tilename_bw,
#                                         PercentMasked,
#                                         self._SaveMasks,
#                                         TileMask,
#                                         self._normalize,
#                                         False,
#                                         self._resize_ratio,
#                                         self._Adj_WindowSize,
#                                         self._Adj_overlap,
#                                     )
#                                 )
#                             else:
#                                 self._queue.put(
#                                     (
#                                         self._associated,
#                                         level,
#                                         (col, row),
#                                         tilename,
#                                         self._format,
#                                         tilename_bw,
#                                         PercentMasked,
#                                         self._SaveMasks,
#                                         TileMask,
#                                         self._normalize,
#                                         True,
#                                         self._resize_ratio,
#                                         self._Adj_WindowSize,
#                                         self._Adj_overlap,
#                                     )
#                                 )

#                         self._tile_done()

#     def _tile_done(self):
#         self._processed += 1
#         count, total = self._processed, self._dz.tile_count
#         if count % 100 == 0 or count == total:
#             print(
#                 "Tiling %s: wrote %d/%d tiles" % (self._associated or "slide", count, total), end="\r", file=sys.stderr
#             )
#             if count == total:
#                 print(file=sys.stderr)

#     def _write_dzi(self):
#         with open("%s.dzi" % self._basename, "w") as fh:
#             fh.write(self.get_dzi())

#     def get_dzi(self):
#         return self._dz.get_dzi(self._format)

#     def jpg_mask_read(self, xmldir):
#         # Original size of the image
#         ImgMaxSizeX_orig = float(self._dz.level_dimensions[-1][0])
#         ImgMaxSizeY_orig = float(self._dz.level_dimensions[-1][1])
#         print("Image Size:")
#         print(ImgMaxSizeX_orig, ImgMaxSizeY_orig)
#         # Number of centers at the highest resolution
#         cols, rows = self._dz.level_tiles[-1]
#         Img_Fact = 1
#         try:
#             # xmldir: change extension from xml to *jpg
#             try:
#                 xmldir = xmldir
#                 xmlcontent = imread(xmldir)
#             except:
#                 xmldir = xmldir[:-4] + "mask.jpg"
#                 xmlcontent = imread(xmldir)
#             xmlcontent = xmlcontent - np.min(xmlcontent)
#             mask = xmlcontent / np.max(xmlcontent)
#             # we want image between 0 and 1
#             xml_valid = True
#         except:
#             xml_valid = False
#             print("error with minidom.parse(xmldir)")
#             return [], xml_valid, 1.0
#         print("Mask size orig:")
#         print(mask.shape)
#         return mask, xml_valid, Img_Fact

#     def xml_read(self, xmldir, Attribute_Name, Fieldxml, AnnotationMode, ImgExt):
#         # Original size of the image
#         ImgMaxSizeX_orig = float(self._dz.level_dimensions[-1][0])
#         ImgMaxSizeY_orig = float(self._dz.level_dimensions[-1][1])
#         print("image size:", str(ImgMaxSizeX_orig), ", ", str(ImgMaxSizeY_orig))

#         cols, rows = self._dz.level_tiles[-1]

#         if ImgExt == "scn" and AnnotationMode == "Aperio":
#             tmp = ImgMaxSizeX_orig
#             ImgMaxSizeX_orig = ImgMaxSizeY_orig
#             ImgMaxSizeY_orig = tmp

#         NewFact = max(ImgMaxSizeX_orig, ImgMaxSizeY_orig) / min(max(ImgMaxSizeX_orig, ImgMaxSizeY_orig), 15000.0)
#         Img_Fact = float(ImgMaxSizeX_orig) / 5.0 / float(cols)

#         if AnnotationMode == "Omero":
#             print("Omero annotations not yet ready")
#             f = open(xmldir, "r")
#             reader = csv.reader(f)
#             headers = next(reader, None)
#             xmlcontent = {}
#             for ncol in headers:
#                 xmlcontent[ncol] = []
#             for nrow in reader:
#                 for h, r in zip(headers, nrow):
#                     xmlcontent[h].append(r)
#             f.close()

#             xml_valid = True
#             xy = {}
#             xy_neg = {}
#             NbRg = 0
#             x_min = ImgMaxSizeX_orig
#             x_max = 0
#             y_min = ImgMaxSizeY_orig
#             y_max = 0
#             print(xmlcontent["image_name"])
#             print(xmlcontent["text"])
#             print(self._slide.properties)
#             if True:
#                 for eachR in range(len(xmlcontent["image_name"])):
#                     labeltag = xmlcontent["text"][eachR]
#                     whichRegion = xmlcontent["image_name"][eachR].split("[")[-1]
#                     whichRegion = whichRegion.split("]")[0]
#                     whichRegion = str(int(whichRegion) - 1)
#                     if ImgExt == "scn":
#                         x_min = min(
#                             x_min, int(int(self._slide.properties["openslide.region[" + whichRegion + "].x"]) / NewFact)
#                         )
#                         y_min = min(
#                             y_min, int(int(self._slide.properties["openslide.region[" + whichRegion + "].y"]) / NewFact)
#                         )
#                         x_max = max(
#                             x_max,
#                             int(
#                                 (
#                                     int(self._slide.properties["openslide.region[" + whichRegion + "].width"])
#                                     + int(self._slide.properties["openslide.region[" + whichRegion + "].x"])
#                                 )
#                                 / NewFact
#                             ),
#                         )
#                         y_max = max(
#                             y_max,
#                             int(
#                                 (
#                                     int(self._slide.properties["openslide.region[" + whichRegion + "].height"])
#                                     + int(self._slide.properties["openslide.region[" + whichRegion + "].y"])
#                                 )
#                                 / NewFact
#                             ),
#                         )
#                     else:
#                         x_min = 0
#                         y_min = 0
#                         x_max = ImgMaxSizeX_orig / NewFact
#                         y_max = ImgMaxSizeY_orig / NewFact
#                     if (Attribute_Name == []) | (Attribute_Name == ""):
#                         # No filter on label name
#                         isLabelOK = True
#                     elif Attribute_Name == labeltag:
#                         isLabelOK = True
#                     elif Attribute_Name == "non_selected_regions":
#                         isLabelOK = True
#                     else:
#                         isLabelOK = False
#                     if isLabelOK:
#                         regionID = str(NbRg)
#                         xy[regionID] = []
#                         if xmlcontent["type"][eachR] == "rectangle":
#                             tmp_x = float(xmlcontent["X"][eachR])
#                             tmp_y = float(xmlcontent["Y"][eachR])
#                             tmp_w = float(xmlcontent["Width"][eachR])
#                             tmp_h = float(xmlcontent["Height"][eachR])
#                             tmp_v2 = [
#                                 tmp_x,
#                                 tmp_y,
#                                 tmp_x + tmp_w,
#                                 tmp_y,
#                                 tmp_x + tmp_w,
#                                 tmp_y + tmp_h,
#                                 tmp_x,
#                                 tmp_y + tmp_h,
#                             ]
#                         elif xmlcontent["type"][eachR] == "ellipse":
#                             tmp_x = float(xmlcontent["X"][eachR])
#                             tmp_y = float(xmlcontent["Y"][eachR])
#                             radiusX = float(xmlcontent["RadiusX"][eachR])
#                             radiusY = float(xmlcontent["RadiusY"][eachR])
#                             tmp_v2 = []
#                             for Xi in np.linspace(tmp_x - radiusX / 2, tmp_x + radiusX / 2, np.ceil(radiusX / 20))[
#                                 2:-1
#                             ]:
#                                 tmp_v2.append(Xi)
#                                 tmp_v2.append(tmp_y + radiusY / 2 * np.sqrt(1 - (Xi - tmp_x) ** 2 / (radiusX / 2) ** 2))
#                             for Xi in np.linspace(tmp_x + radiusX / 2, tmp_x - radiusX / 2, np.ceil(radiusX / 20))[
#                                 2:-1
#                             ]:
#                                 tmp_v2.append(Xi)
#                                 tmp_v2.append(tmp_y - radiusY / 2 * np.sqrt(1 - (Xi - tmp_x) ** 2 / (radiusX / 2) ** 2))
#                         else:
#                             tmp_v = re.sub(",", " ", xmlcontent["Points"][eachR]).split()
#                             tmp_v2 = [float(ii) for ii in tmp_v]
#                         if ImgExt == "scn":
#                             for i in range(1, len(tmp_v2), 2):
#                                 tmp_v2[i] = tmp_v2[i] + int(
#                                     self._slide.properties["openslide.region[" + whichRegion + "].y"]
#                                 )
#                             for i in range(0, len(tmp_v2), 2):
#                                 tmp_v2[i] = tmp_v2[i] + int(
#                                     self._slide.properties["openslide.region[" + whichRegion + "].x"]
#                                 )
#                         vertices = list(tmp_v2)
#                         NbRg += 1
#                         xy[regionID] = [ii / NewFact for ii in vertices]
#                         # no field for "negative region" - if it is, create a "xy_neg[regionID]"

#             if ImgExt == "scn":
#                 x_start = ImgMaxSizeX_orig
#                 y_start = ImgMaxSizeY_orig
#                 for kk in range(0, 100):
#                     try:
#                         x_start = min(
#                             x_start, int(int(self._slide.properties["openslide.region[" + str(kk) + "].x"]) / NewFact)
#                         )
#                         y_start = min(
#                             y_start, int(int(self._slide.properties["openslide.region[" + str(kk) + "].y"]) / NewFact)
#                         )
#                     except:
#                         break
#                 for kk in xy.keys():
#                     for i in range(1, len(xy[kk]), 2):
#                         xy[kk][i] = xy[kk][i] - y_start
#                     for i in range(0, len(xy[kk]), 2):
#                         xy[kk][i] = xy[kk][i] - x_start
#         elif AnnotationMode == "Aperio":
#             try:
#                 xmlcontent = minidom.parse(xmldir)
#                 xml_valid = True
#             except:
#                 xml_valid = False
#                 print("error with minidom.parse(xmldir)")
#                 return [], xml_valid, 1.0
#             xy = {}
#             xy_neg = {}
#             NbRg = 0
#             labelIDs = xmlcontent.getElementsByTagName("Annotation")
#             for labelID in labelIDs:
#                 if (Attribute_Name == []) | (Attribute_Name == ""):
#                     isLabelOK = True
#                 else:
#                     try:
#                         labeltag = labelID.getElementsByTagName("Attribute")[0]
#                         if Attribute_Name == labeltag.attributes[Fieldxml].value:
#                             isLabelOK = True
#                         else:
#                             isLabelOK = False
#                     except:
#                         isLabelOK = False
#                 if Attribute_Name == "non_selected_regions":
#                     isLabelOK = True

#                 if isLabelOK:
#                     regionlist = labelID.getElementsByTagName("Region")
#                     for region in regionlist:
#                         vertices = region.getElementsByTagName("Vertex")
#                         NbRg += 1
#                         regionID = region.attributes["Id"].value + str(NbRg)
#                         NegativeROA = region.attributes["NegativeROA"].value
#                         # print("%d vertices" % len(vertices))
#                         if len(vertices) > 0:
#                             # print( len(vertices) )
#                             if NegativeROA == "0":
#                                 xy[regionID] = []
#                                 for vertex in vertices:
#                                     # get the x value of the vertex / convert them into index in the tiled matrix of the base image
#                                     x = int(round(float(vertex.attributes["X"].value) / NewFact))
#                                     y = int(round(float(vertex.attributes["Y"].value) / NewFact))
#                                     xy[regionID].append((x, y))

#                             elif NegativeROA == "1":
#                                 xy_neg[regionID] = []
#                                 for vertex in vertices:
#                                     # get the x value of the vertex / convert them into index in the tiled matrix of the base image
#                                     x = int(round(float(vertex.attributes["X"].value) / NewFact))
#                                     y = int(round(float(vertex.attributes["Y"].value) / NewFact))
#                                     xy_neg[regionID].append((x, y))

#         ## End Aperio
#         elif AnnotationMode == "QuPath_orig":
#             print("QuPath geojson annotation file detected")
#             xmlcontent = json.load(open(xmldir))
#             xml_valid = True
#             xy = {}
#             xy_neg = {}
#             NbRg = 0
#             # json
#             # featurecoll
#             if "features" in xmlcontent.keys():
#                 for eachR in range(len(xmlcontent["features"])):
#                     labeltag = "unlabelled"
#                     if "properties" in xmlcontent["features"][eachR].keys():
#                         if "classification" in xmlcontent["features"][eachR]["properties"].keys():
#                             if "name" in xmlcontent["features"][eachR]["properties"]["classification"].keys():
#                                 labeltag = xmlcontent["features"][eachR]["properties"]["classification"]["name"]
#                     if (Attribute_Name == []) | (Attribute_Name == ""):
#                         isLabelOK = True
#                     elif Attribute_Name == labeltag:
#                         isLabelOK = True
#                     elif Attribute_Name == "non_selected_regions":
#                         isLabelOK = True
#                     else:
#                         isLabelOK = False
#                     if isLabelOK:
#                         if "geometry" in xmlcontent["features"][eachR].keys():
#                             xmlcontent["features"][eachR]["geometry"]["coordinates"]
#                             # coordinates
#                             vertices = []
#                             for eachv in range(len(xmlcontent["features"][eachR]["geometry"]["coordinates"])):
#                                 if xmlcontent["features"][eachR]["geometry"]["type"] == "LineString":
#                                     # HALO format
#                                     vertices.append(xmlcontent["features"][eachR]["geometry"]["coordinates"][eachv][0])
#                                     vertices.append(xmlcontent["features"][eachR]["geometry"]["coordinates"][eachv][1])
#                                 else:
#                                     # QuPath format
#                                     for eachXY in range(
#                                         len(xmlcontent["features"][eachR]["geometry"]["coordinates"][eachv])
#                                     ):
#                                         print(xmlcontent["features"][eachR]["geometry"]["coordinates"][eachv])
#                                         vertices.append(
#                                             xmlcontent["features"][eachR]["geometry"]["coordinates"][eachv][eachXY][0]
#                                         )
#                                         vertices.append(
#                                             xmlcontent["features"][eachR]["geometry"]["coordinates"][eachv][eachXY][1]
#                                         )
#                             regionID = str(NbRg)
#                             xy[regionID] = [ii / NewFact for ii in vertices]
#                             NbRg += 1

#         elif AnnotationMode == "QuPath":
#             print("QuPath annotation file detected")
#             xmlcontent = json.load(open(xmldir))
#             xml_valid = True
#             xy = {}
#             xy_neg = {}
#             NbRg = 0
#             if "annotations" in xmlcontent.keys():
#                 for eachR in range(len(xmlcontent["annotations"])):
#                     if "class" in xmlcontent["annotations"][eachR].keys():
#                         labeltag = xmlcontent["annotations"][eachR]["class"]
#                         if (Attribute_Name == []) | (Attribute_Name == ""):
#                             # No filter on label name
#                             isLabelOK = True
#                         elif Attribute_Name == labeltag:
#                             isLabelOK = True
#                         elif Attribute_Name == "non_selected_regions":
#                             isLabelOK = True
#                         else:
#                             isLabelOK = False
#                         if isLabelOK:
#                             regionID = str(NbRg)
#                             xy[regionID] = []
#                             vertices = xmlcontent["annotations"][eachR]["points"]
#                             NbRg += 1
#                             xy[regionID] = [ii / NewFact for ii in vertices]
#                             # no field for "negative region" - if it is, create a "xy_neg[regionID]"
#             elif "dictionaries" in xmlcontent.keys():
#                 # Imagedrive format
#                 NbRg = 0
#                 for eachROI in range(len(xmlcontent["dictionaries"])):
#                     vertices = []
#                     if "label" in xmlcontent["dictionaries"][eachROI].keys():
#                         labeltag = xmlcontent["dictionaries"][eachROI]["label"]
#                         if (Attribute_Name == []) | (Attribute_Name == ""):
#                             # No filter on label name
#                             isLabelOK = True
#                         elif Attribute_Name == labeltag:
#                             isLabelOK = True
#                         elif Attribute_Name == "non_selected_regions":
#                             isLabelOK = True
#                         else:
#                             isLabelOK = False
#                         if isLabelOK:
#                             for eachXY in range(len(xmlcontent["dictionaries"][eachROI]["path"]["segments"])):
#                                 vertices.append(xmlcontent["dictionaries"][eachROI]["path"]["segments"][eachXY][0])
#                                 vertices.append(xmlcontent["dictionaries"][eachROI]["path"]["segments"][eachXY][1])
#                             regionID = str(NbRg)
#                             xy[regionID] = [ii / NewFact for ii in vertices]
#                             NbRg += 1

#         #### Remove 2 spaces ###
#         print("Img_Fact:")
#         print(NewFact)
#         img = Image.new("L", (int(ImgMaxSizeX_orig / NewFact), int(ImgMaxSizeY_orig / NewFact)), 0)
#         for regionID in xy.keys():
#             xy_a = xy[regionID]
#             # print(xy_a)
#             ImageDraw.Draw(img, "L").polygon(xy_a, outline=255, fill=255)
#         for regionID in xy_neg.keys():
#             xy_a = xy_neg[regionID]
#             ImageDraw.Draw(img, "L").polygon(xy_a, outline=255, fill=0)
#         # img = img.resize((cols,rows), Image.ANTIALIAS)
#         mask = np.array(img)
#         if AnnotationMode == "Omero":
#             if Attribute_Name == "non_selected_regions" or self._mask_type == 0:
#                 mask[:, : x_min - x_start] = 255
#                 mask[: y_min - y_start, :] = 255
#                 mask[:, x_max - x_start :] = 255
#                 mask[y_max - y_start :, :] = 255

#         if ImgExt == "scn" and AnnotationMode == "Aperio":
#             mask = np.rot90(mask)
#         if Attribute_Name == "non_selected_regions":
#             Image.fromarray(255 - mask).save(
#                 os.path.join(
#                     os.path.split(self._basename[:-1])[0],
#                     "mask_" + os.path.basename(self._basename) + "_" + Attribute_Name + ".jpeg",
#                 )
#             )
#         else:
#             if self._mask_type == 0:
#                 Image.fromarray(255 - mask).save(
#                     os.path.join(
#                         os.path.split(self._basename[:-1])[0],
#                         "mask_" + os.path.basename(self._basename) + "_" + Attribute_Name + "_inv.jpeg",
#                     )
#                 )
#             else:
#                 Image.fromarray(mask).save(
#                     os.path.join(
#                         os.path.split(self._basename[:-1])[0],
#                         "mask_" + os.path.basename(self._basename) + "_" + Attribute_Name + ".jpeg",
#                     )
#                 )
#         return mask / 255.0, xml_valid, NewFact
#         # Img_Fact


# class DeepZoomStaticTiler(object):
#     """Handles generation of tiles and metadata for all images in a slide."""

#     def __init__(
#         self,
#         slidepath,
#         basename,
#         format,
#         tile_size,
#         overlap,
#         limit_bounds,
#         quality,
#         workers,
#         with_viewer,
#         Bkg,
#         basenameJPG,
#         xmlfile,
#         mask_type,
#         ROIpc,
#         oLabel,
#         ImgExtension,
#         SaveMasks,
#         Mag,
#         normalize,
#         Fieldxml,
#         pixelsize,
#         pixelsizerange,
#         Adj_WindowSize,
#         resize_ratio,
#         Best_level,
#         Adj_overlap,
#         Std,
#     ):
#         if with_viewer:
#             # Check extra dependency before doing a bunch of work
#             import jinja2
#         self._slide = open_slide(slidepath)
#         self._basename = basename
#         self._basenameJPG = basenameJPG
#         self._xmlfile = xmlfile
#         self._mask_type = mask_type
#         self._format = format
#         self._tile_size = tile_size
#         self._overlap = overlap
#         self._limit_bounds = limit_bounds
#         self._queue = JoinableQueue(2 * workers)
#         self._workers = workers
#         self._with_viewer = with_viewer
#         self._Bkg = Bkg
#         self._ROIpc = ROIpc
#         self._dzi_data = {}
#         self._xmlLabel = oLabel
#         self._ImgExtension = ImgExtension
#         self._SaveMasks = SaveMasks
#         self._Mag = Mag
#         self._normalize = normalize
#         self._Fieldxml = Fieldxml
#         self._pixelsize = pixelsize
#         self._pixelsizerange = pixelsizerange
#         self._rescale = False
#         self._resize_ratio = resize_ratio
#         self._Best_level = Best_level
#         self._Adj_WindowSize = Adj_WindowSize
#         self._Adj_overlap = Adj_overlap
#         self._Std = Std
#         for _i in range(workers):
#             TileWorker(
#                 self._queue,
#                 slidepath,
#                 self._Adj_WindowSize,
#                 self._Adj_overlap,
#                 limit_bounds,
#                 quality,
#                 self._Bkg,
#                 self._ROIpc,
#                 self._Std,
#             ).start()
#             print(
#                 "worker " + str(workers) + " started with " + str(self._Adj_WindowSize), ", " + str(self._Adj_overlap)
#             )

#     def run(self):
#         self._run_image()
#         if self._with_viewer:
#             for name in self._slide.associated_images:
#                 self._run_image(name)
#             self._write_html()
#             self._write_static()
#         self._shutdown()

#     def _run_image(self, associated=None):
#         """Run a single image from self._slide."""
#         if associated is None:
#             image = self._slide
#             if self._with_viewer:
#                 basename = os.path.join(self._basename, VIEWER_SLIDE_NAME)
#             else:
#                 basename = self._basename
#         else:
#             image = ImageSlide(self._slide.associated_images[associated])
#             basename = os.path.join(self._basename, self._slugify(associated))

#         dz = DeepZoomGenerator(image, self._Adj_WindowSize, self._Adj_overlap, limit_bounds=self._limit_bounds)

#         tiler = DeepZoomImageTiler(
#             dz,
#             basename,
#             self._format,
#             associated,
#             self._queue,
#             self._slide,
#             self._basenameJPG,
#             self._xmlfile,
#             self._mask_type,
#             self._xmlLabel,
#             self._ROIpc,
#             self._ImgExtension,
#             self._SaveMasks,
#             self._Mag,
#             self._normalize,
#             self._Fieldxml,
#             self._pixelsize,
#             self._pixelsizerange,
#             self._Best_level,
#             self._resize_ratio,
#             self._Adj_WindowSize,
#             self._Adj_overlap,
#         )
#         import time

#         time.sleep(3)

#         tiler.run()
#         print("end tiler")
#         self._dzi_data[self._url_for(associated)] = tiler.get_dzi()

#     def _url_for(self, associated):
#         if associated is None:
#             base = VIEWER_SLIDE_NAME
#         else:
#             base = self._slugify(associated)
#         return "%s.dzi" % base

#     def _write_html(self):
#         import jinja2

#         env = jinja2.Environment(loader=jinja2.PackageLoader(__name__), autoescape=True)
#         template = env.get_template("slide-multipane.html")
#         associated_urls = dict((n, self._url_for(n)) for n in self._slide.associated_images)
#         try:
#             mpp_x = self._slide.properties[openslide.PROPERTY_NAME_MPP_X]
#             mpp_y = self._slide.properties[openslide.PROPERTY_NAME_MPP_Y]
#             mpp = (float(mpp_x) + float(mpp_y)) / 2
#         except (KeyError, ValueError):
#             mpp = 0
#         # Embed the dzi metadata in the HTML to work around Chrome's
#         # refusal to allow XmlHttpRequest from file:///, even when
#         # the originating page is also a file:///
#         data = template.render(
#             slide_url=self._url_for(None),
#             slide_mpp=mpp,
#             associated=associated_urls,
#             properties=self._slide.properties,
#             dzi_data=json.dumps(self._dzi_data),
#         )
#         with open(os.path.join(self._basename, "index.html"), "w") as fh:
#             fh.write(data)

#     def _write_static(self):
#         basesrc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
#         basedst = os.path.join(self._basename, "static")
#         self._copydir(basesrc, basedst)
#         self._copydir(os.path.join(basesrc, "images"), os.path.join(basedst, "images"))

#     def _copydir(self, src, dest):
#         if not os.path.exists(dest):
#             os.makedirs(dest)
#         for name in os.listdir(src):
#             srcpath = os.path.join(src, name)
#             if os.path.isfile(srcpath):
#                 shutil.copy(srcpath, os.path.join(dest, name))

#     @classmethod
#     def _slugify(cls, text):
#         text = normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
#         return re.sub("[^a-z0-9]+", "_", text)

#     def _shutdown(self):
#         for _i in range(self._workers):
#             self._queue.put(None)
#         self._queue.join()


# def ImgWorker(queue):
#     while True:
#         cmd = queue.get()
#         if cmd is None:
#             queue.task_done()
#             break
#         subprocess.Popen(cmd, shell=True).wait()
#         queue.task_done()


# def xml_read_labels(xmldir, Fieldxml, AnnotationMode):
#     if AnnotationMode == "Aperio":
#         try:
#             xmlcontent = minidom.parse(xmldir)
#             xml_valid = True
#         except:
#             xml_valid = False
#             print("error with minidom.parse(xmldir)")
#             return [], xml_valid
#         labeltag = xmlcontent.getElementsByTagName("Attribute")
#         xml_labels = []
#         for xmllabel in labeltag:
#             xml_labels.append(xmllabel.attributes[Fieldxml].value)
#         if xml_labels == []:
#             xml_labels = [""]
#         # print(xml_labels)
#     elif AnnotationMode == "QuPath_orig":
#         data = json.load(open(xmldir))
#         xml_labels = []
#         xml_valid = False
#         if "features" in data.keys():
#             for eachR in range(len(data["features"])):
#                 labeltag = "unlabelled"
#                 if "properties" in data["features"][eachR].keys():
#                     if "classification" in data["features"][eachR]["properties"].keys():
#                         if "name" in data["features"][eachR]["properties"]["classification"].keys():
#                             labeltag = data["features"][eachR]["properties"]["classification"]["name"]
#                 xml_labels.append(labeltag)
#         xml_labels = np.unique(xml_labels)
#         if len(xml_labels) > 0:
#             xml_valid = True
#     elif AnnotationMode == "QuPath":
#         data = json.load(open(xmldir))
#         xml_labels = []
#         xml_valid = False
#         if "annotations" in data.keys():
#             for eachR in range(len(data["annotations"])):
#                 xml_labels.append(data["annotations"][eachR]["class"])
#         elif "dictionaries" in data.keys():
#             for eachR in range(len(data["dictionaries"])):
#                 xml_labels.append(data["dictionaries"][eachR]["label"])
#         xml_labels = np.unique(xml_labels)
#         if len(xml_labels) > 0:
#             xml_valid = True
#     elif AnnotationMode == "Omero":
#         f = open(xmldir, "r")
#         reader = csv.reader(f)
#         headers = next(reader, None)
#         xmlcontent = {}
#         for ncol in headers:
#             xmlcontent[ncol] = []
#         for nrow in reader:
#             for h, r in zip(headers, nrow):
#                 xmlcontent[h].append(r)
#         f.close()
#         xml_labels = []
#         for eachR in range(len(xmlcontent["image_name"])):
#             xml_labels.append(xmlcontent["text"][eachR])
#         xml_labels = np.unique(xml_labels)
#         if len(xml_labels) > 0:
#             xml_valid = True
#     elif AnnotationMode == "ImageJ":
#         xml_labels = ["ROI_maxVal"]
#         xml_valid = True
#     return xml_labels, xml_valid
