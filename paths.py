import os

projectBase = "C:\\Users\\geiss\\OneDrive\\Desktop\\LainProject\\Python\\AramisGridInterpolation"


class projectDirs:
    data = "data"
    helpers = "helpers"
    snippets = "snippets"


class rawDataFiles:
    dreieckRaster = projectBase + os.sep + projectDirs.data + os.sep + "dreieck_raster.csv"
    viereckRaster = projectBase + os.sep + projectDirs.data + os.sep + "viereck_raster_temp.asc"
    cleanedViereckRaster = projectBase + os.sep + projectDirs.data + os.sep + "cleanedAsc.csv"


class AramisGrindInterpolationDir:
    base = projectBase
    dirname = "AramisGrindInterpolation"

    data = base + os.sep + "data"
    helpers = base + os.sep + "helpers"
    metadata = base + os.sep + "metadata"
    snippets = base + os.sep + "snippets"
    temp = base + os.sep + "temp"
    todos = base + os.sep + "todos"
    tests = base + os.sep + "tests"

    subDirs = [data, helpers, metadata, snippets, temp, tests, todos]


class DataDir:
    dirname = "data"

    parentDir = AramisGrindInterpolationDir.base
    base = AramisGrindInterpolationDir.base + os.sep + dirname

    cleanSamples = base + os.sep + "cleanSamples"
    rawSamples = base + os.sep + "rawSamples"
    samplesOld = base + os.sep + "samplesOld"

    subDirs = [cleanSamples, rawSamples, samplesOld]


class metadataDir:
    dirname = "metadata"
    parentDir = AramisGrindInterpolationDir.base
    base = AramisGrindInterpolationDir.base + os.sep + dirname

    sampleAnalysis = base + os.sep + "sampleAnalysis"

    subdirs = [sampleAnalysis]
