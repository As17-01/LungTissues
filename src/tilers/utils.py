import numpy as np
from skimage import color


def RGB_to_lab(tile):
    Lab = color.rgb2lab(tile)
    return Lab


def Lab_to_RGB(Lab):
    newtile = (color.lab2rgb(Lab) * 255).astype(np.uint8)
    return newtile


def normalize_tile(tile, NormVec):
    Lab = RGB_to_lab(tile)
    TileMean = [0, 0, 0]
    TileStd = [1, 1, 1]
    newMean = NormVec[0:3]
    newStd = NormVec[3:6]
    for i in range(3):
        TileMean[i] = np.mean(Lab[:, :, i])
        TileStd[i] = np.std(Lab[:, :, i])
        tmp = ((Lab[:, :, i] - TileMean[i]) * (newStd[i] / TileStd[i])) + newMean[i]
        if i == 0:
            tmp[tmp < 0] = 0
            tmp[tmp > 100] = 100
            Lab[:, :, i] = tmp
        else:
            tmp[tmp < -128] = 128
            tmp[tmp > 127] = 127
            Lab[:, :, i] = tmp
    tile = Lab_to_RGB(Lab)
    return tile
