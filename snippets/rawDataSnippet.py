import os

from data.dataKeys import viereckRasterHeaders, dreieckRasterHeaders
from data.dataLoader import loadDataAsDfs
from data.dataParser import parseAscFile
from helpers.csvTools import loadDataframe, writeDataframeToXlsx
from helpers.fileTools import writeLines
from helpers.seabornTools import createScatterPlot
from paths import rawDataFiles, projectBase
import numpy as np


def conversionTest1():
    src1 = rawDataFiles.dreieckRaster
    src2 = rawDataFiles.viereckRaster
    dst = projectBase + os.sep + "data"

    df1 = loadDataframe(src1, delimiter="	")

    newLines = parseAscFile(src2)
    fn = "cleanedAsc.csv"
    writeLines(fn, dstDir=dst, lines=newLines)
    df2 = loadDataframe(rawDataFiles.cleanedAsc, delimiter=";")

    fn1 = "dreieckRaster.xlsx"
    fn2 = "viereckRaster.xlsx"

    writeDataframeToXlsx(df1, dst, fn1)
    writeDataframeToXlsx(df2, dst, fn2)


def createScattterPlotTest2D():
    data, _ = loadDataAsDfs()
    x = data[dreieckRasterHeaders.x].tolist()
    y = data[dreieckRasterHeaders.y].tolist()

    x = np.array(x)
    y = np.array(y)

    N = len(x)
    data = np.zeros((N, 2))
    data[:, 0] = x
    data[:, 1] = y
    createScatterPlot(data)


def createScattterPlotTest3D():
    _, data = loadDataAsDfs()
    x = data[viereckRasterHeaders.x_Koordinate].tolist()
    y = data[viereckRasterHeaders.y_Koordinate].tolist()
    z = data[viereckRasterHeaders.z_Koordinate].tolist()

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    N = len(x)
    data = np.zeros((N, 3))
    data[:, 0] = x
    data[:, 1] = y
    data[:, 2] = z

    createScatterPlot(data)


# conversionTest1()
createScattterPlotTest2D()
createScattterPlotTest3D()
print("Finally done")
