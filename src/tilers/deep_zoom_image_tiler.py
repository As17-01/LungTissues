from __future__ import print_function

import json
import os
import re
from multiprocessing import JoinableQueue
from multiprocessing import Process
from xml.dom import minidom

import numpy as np
import openslide
from imageio import imread
from PIL import Image
from PIL import ImageDraw

Image.MAX_IMAGE_PIXELS = None
import csv


class DeepZoomImageTiler(object):
    """Handles generation of tiles and metadata for a single image."""

    def __init__(
        self,
        dz,
        basename,
        format,
        associated,
        queue,
        slide,
        basenameJPG,
        xmlfile,
        mask_type,
        xmlLabel,
        ROIpc,
        ImgExtension,
        SaveMasks,
        Mag,
        normalize,
        Fieldxml,
        pixelsize,
        pixelsizerange,
        Best_level,
        resize_ratio,
        Adj_WindowSize,
        Adj_overlap,
    ):
        self._dz = dz
        self._basename = basename
        self._basenameJPG = basenameJPG
        self._format = format
        self._associated = associated
        self._queue = queue
        self._processed = 0
        self._slide = slide
        self._xmlfile = xmlfile
        self._mask_type = mask_type
        self._xmlLabel = xmlLabel
        self._ROIpc = ROIpc
        self._ImgExtension = ImgExtension
        self._SaveMasks = SaveMasks
        self._Mag = Mag
        self._normalize = normalize
        self._Fieldxml = Fieldxml
        self._pixelsize = pixelsize
        self._pixelsizerange = pixelsizerange
        self._Best_level = Best_level
        self._resize_ratio = resize_ratio
        self._Adj_WindowSize = Adj_WindowSize
        self._Adj_overlap = Adj_overlap

    def run(self):
        self._write_tiles()
        self._write_dzi()

    def _write_tiles(self):
        Magnification = 20
        # get slide dimensions, zoom levels, and objective information
        Factors = self._slide.level_downsamples
        try:
            Objective = float(self._slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        except:
            try:
                for nfields in self._slide.properties["tiff.ImageDescription"].split("|"):
                    if "AppMag" in nfields:
                        Objective = float(nfields.split(" = ")[1])
            except:
                Objective = 1.0
                Magnification = Objective

        # calculate magnifications
        Available = tuple(Objective / x for x in Factors)
        # find highest magnification greater than or equal to 'Desired'
        Mismatch = tuple(x - Magnification for x in Available)
        AbsMismatch = tuple(abs(x) for x in Mismatch)
        if len(AbsMismatch) < 1:
            return

        xml_valid = False
        # a dir was provided for xml files

        ImgID = os.path.basename(self._basename)
        if os.path.isfile(os.path.join(self._xmlfile, ImgID + ".xml")):
            # If path exists, Aperio assumed
            xmldir = os.path.join(self._xmlfile, ImgID + ".xml")
            AnnotationMode = "Aperio"
        elif os.path.isfile(os.path.join(self._xmlfile, ImgID + ".geojson")):
            xmldir = os.path.join(self._xmlfile, ImgID + ".geojson")
            AnnotationMode = "QuPath_orig"
        elif os.path.isfile(os.path.join(self._xmlfile, ImgID + ".json")):
            # QuPath assumed
            xmldir = os.path.join(self._xmlfile, ImgID + ".json")
            AnnotationMode = "QuPath"
        elif os.path.isfile(os.path.join(self._xmlfile, ImgID + ".csv")):
            xmldir = os.path.join(self._xmlfile, ImgID + ".csv")
            AnnotationMode = "Omero"
        elif os.path.isfile(os.path.join(self._xmlfile, ImgID + ".tif")):
            xmldir = os.path.join(self._xmlfile, ImgID + ".tif")
            AnnotationMode = "ImageJ"
        else:
            AnnotationMode = "None"

        if AnnotationMode == "ImageJ":
            mask, xml_valid, Img_Fact = self.jpg_mask_read(xmldir)
            if xml_valid == False:
                return

        if self._Mag <= 0:
            if self._pixelsize > 0:
                level_range = [level for level in range(self._dz.level_count - 1, -1, -1)]
                try:
                    OrgPixelSizeX = float(self._slide.properties["openslide.mpp-x"])
                    OrgPixelSizeY = float(self._slide.properties["openslide.mpp-y"])
                except:
                    try:
                        for nfields in self._slide.properties["tiff.ImageDescription"].split("|"):
                            if "MPP" in nfields:
                                OrgPixelSizeX = float(nfields.split(" = ")[1])
                                OrgPixelSizeY = OrgPixelSizeX
                    except:
                        DesiredLevel = -1
                        return
                AllPixelSizeDiffX = [
                    (abs(OrgPixelSizeX * pow(2, self._dz.level_count - (level + 1)) - self._pixelsize))
                    for level in range(self._dz.level_count - 1, -1, -1)
                ]
                AllPixelSizeDiffY = [
                    (abs(OrgPixelSizeY * pow(2, self._dz.level_count - (level + 1)) - self._pixelsize))
                    for level in range(self._dz.level_count - 1, -1, -1)
                ]
                IndxX = AllPixelSizeDiffX.index(min(AllPixelSizeDiffX))
                IndxY = AllPixelSizeDiffY.index(min(AllPixelSizeDiffY))
                levelX = AllPixelSizeDiffX[IndxX]
                levelY = AllPixelSizeDiffY[IndxY]
                if IndxX != IndxY:
                    return
                if (levelX > self._pixelsizerange) and (self._pixelsizerange >= 0):
                    return
                if (levelY > self._pixelsizerange) and (self._pixelsizerange >= 0):
                    return
                if self._pixelsizerange < 0:
                    level_range = [level for level in range(self._dz.level_count - 1, -1, -1)]
                    IndxX = self._Best_level
                DesiredLevel = level_range[IndxX]
                if not os.path.exists(("/".join(self._basename.split("/")[:-1]))):
                    os.makedirs(("/".join(self._basename.split("/")[:-1])))
                with open(os.path.join(("/".join(self._basename.split("/")[:-1])), "pixelsizes.txt"), "a") as file_out:
                    file_out.write(
                        self._basenameJPG
                        + "\t"
                        + str(OrgPixelSizeX * pow(2, IndxX))
                        + "\t"
                        + str(OrgPixelSizeX * pow(2, IndxX) * self._resize_ratio)
                        + "\n"
                    )

        for level in range(self._dz.level_count - 1, -1, -1):
            ThisMag = Available[0] / pow(2, self._dz.level_count - (level + 1))
            if self._Mag > 0:
                if ThisMag != self._Mag:
                    continue
            elif self._pixelsize > 0:
                if level != DesiredLevel:
                    continue
                else:
                    tiledir_pixel = os.path.join("%s_files" % self._basename, str(self._pixelsize))

            tiledir = os.path.join("%s_files" % self._basename, str(ThisMag))
            if not os.path.exists(tiledir):
                os.makedirs(tiledir)
                if self._pixelsize > 0:
                    os.symlink(str(ThisMag), tiledir_pixel, target_is_directory=True)
            cols, rows = self._dz.level_tiles[level]

            for row in range(rows):
                for col in range(cols):
                    InsertBaseName = False
                    if InsertBaseName:
                        tilename = os.path.join(tiledir, "%s_%d_%d.%s" % (self._basenameJPG, col, row, self._format))
                        tilename_bw = os.path.join(
                            tiledir, "%s_%d_%d_mask.%s" % (self._basenameJPG, col, row, self._format)
                        )
                    else:
                        tilename = os.path.join(tiledir, "%d_%d.%s" % (col, row, self._format))
                        tilename_bw = os.path.join(tiledir, "%d_%d_mask.%s" % (col, row, self._format))
                    if xml_valid:

                        Dlocation, Dlevel, Dsize = self._dz.get_tile_coordinates(level, (col, row))
                        if self._ImgExtension == "mrxs":
                            print(Dlocation, Dlevel, Dsize, level, col, row)
                            aa, bb, cc = self._dz.get_tile_coordinates(level, (0, 0))
                            Dlocation = tuple(map(lambda i, j: i - j, Dlocation, aa))
                            print(Dlocation, Dlevel, Dsize, level, col, row)
                        Ddimension = tuple(
                            [
                                pow(2, (self._dz.level_count - 1 - level)) * x
                                for x in self._dz.get_tile_dimensions(level, (col, row))
                            ]
                        )
                        startIndY_current_level_conv = int((Dlocation[1]) / Img_Fact)
                        endIndY_current_level_conv = int((Dlocation[1] + Ddimension[1]) / Img_Fact)
                        startIndX_current_level_conv = int((Dlocation[0]) / Img_Fact)
                        endIndX_current_level_conv = int((Dlocation[0] + Ddimension[0]) / Img_Fact)

                        if self._ImgExtension == "scn":
                            startIndY_current_level_conv = int(
                                ((Dlocation[1]) - self._dz.get_tile_coordinates(level, (0, 0))[0][1]) / Img_Fact
                            )
                            endIndY_current_level_conv = int(
                                ((Dlocation[1] + Ddimension[1]) - self._dz.get_tile_coordinates(level, (0, 0))[0][1])
                                / Img_Fact
                            )
                            startIndX_current_level_conv = int(
                                ((Dlocation[0]) - self._dz.get_tile_coordinates(level, (0, 0))[0][0]) / Img_Fact
                            )
                            endIndX_current_level_conv = int(
                                ((Dlocation[0] + Ddimension[0]) - self._dz.get_tile_coordinates(level, (0, 0))[0][0])
                                / Img_Fact
                            )

                        TileMask = mask[
                            startIndY_current_level_conv:endIndY_current_level_conv,
                            startIndX_current_level_conv:endIndX_current_level_conv,
                        ]
                        PercentMasked = mask[
                            startIndY_current_level_conv:endIndY_current_level_conv,
                            startIndX_current_level_conv:endIndX_current_level_conv,
                        ].mean()

                        if self._mask_type == 0:
                            # keep ROI outside of the mask
                            PercentMasked = 1.0 - PercentMasked

                    else:
                        PercentMasked = 1.0
                        TileMask = []

                    if not os.path.exists(tilename):
                        if self._Best_level == -1:
                            self._queue.put(
                                (
                                    self._associated,
                                    level,
                                    (col, row),
                                    tilename,
                                    self._format,
                                    tilename_bw,
                                    PercentMasked,
                                    self._SaveMasks,
                                    TileMask,
                                    self._normalize,
                                    False,
                                    self._resize_ratio,
                                    self._Adj_WindowSize,
                                    self._Adj_overlap,
                                )
                            )
                        else:
                            self._queue.put(
                                (
                                    self._associated,
                                    level,
                                    (col, row),
                                    tilename,
                                    self._format,
                                    tilename_bw,
                                    PercentMasked,
                                    self._SaveMasks,
                                    TileMask,
                                    self._normalize,
                                    True,
                                    self._resize_ratio,
                                    self._Adj_WindowSize,
                                    self._Adj_overlap,
                                )
                            )

                    self._tile_done()

    def _tile_done(self):
        self._processed += 1

    def _write_dzi(self):
        with open("%s.dzi" % self._basename, "w") as fh:
            fh.write(self.get_dzi())

    def get_dzi(self):
        return self._dz.get_dzi(self._format)

    def jpg_mask_read(self, xmldir):

        Img_Fact = 1
        try:
            try:
                xmldir = xmldir
                xmlcontent = imread(xmldir)
            except:
                xmldir = xmldir[:-4] + "mask.jpg"
                xmlcontent = imread(xmldir)

            xmlcontent = xmlcontent - np.min(xmlcontent)
            mask = xmlcontent / np.max(xmlcontent)
            # we want image between 0 and 1
            xml_valid = True
        except:
            xml_valid = False
            return [], xml_valid, 1.0
        return mask, xml_valid, Img_Fact
