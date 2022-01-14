import os
import traceback
import sys
import numpy as np
import warnings
import logging

import paths
from interpolate import interpolate

logging.getLogger().setLevel(logging.CRITICAL)

from backbone import alignData
from data.dataLoader import loadAllDataAsArrays, retrieveVoxels, loadDataByPath
from helpers.general_tools import execFunctionOnMatrix, sciF
from helpers.timeTools import myTimer
from metadata.metadataHelper import findBestKForKMeans, countsByCluster, visualizeCounts, createProjectedDataHistograms, \
    computeCountsTest, createScattterPlots, saveDistances, randomHexagonSampling, detailedRandomHexagonSampling, \
    viewTransformedDataSurroundings, countHexagones, colourDataByShape, viewPointMaps, checkGridStructure, \
    createInterpolation, createHeatmapForInterpolatedPoints
from paths import DataDir, MetadataDir, MetadataDirExtern

src = paths.DataDir.cleanSamples
dst = MetadataDir.sampleAnalysis
# src = DataDir.cleanLFT1086
# dstDir = MetadataDirExtern.base

interFfp = DataDir.cleanSamples + os.sep + "dreieck_raster2.csv"

show = False
save = True
dpi = 500

datas = loadAllDataAsArrays(src=src, normalizeData=False)
# dataNames = os.listdir(src)
files = os.listdir(src)

d = dict()
d["show"] = show
d["save"] = save
debug = True
N = len(files)
# numSamples = 100
se = myTimer("createMetadata", ".")
overwrite = False
# 85
# interDstDir = MetadataDir.interpolatedDataAnalysis
for i in range(0, N):
    se.start()
    try:
        print("Starting iteration: ", i)

        dstDir = dst + os.sep + files[i][:-4]

        ffp = src + os.sep + files[i]
        data = loadDataByPath(ffp)
        data = retrieveVoxels(data, normalizeData=False)

        hexagonsOnly = False

        # findBestKForKMeans(data, dstDir=dstDir, show=False, save=True, overwrite=False)
        # computeCountsTest(data, dstDir=dstDir, show=False, save=True, overwrite=False)
        # visualizeCounts(data, dstDir=dstDir, show=False, save=True, overwrite=False)
        # countsByCluster(data, dstDir=dstDir, show=False, save=True, overwrite=False)
        # createScattterPlots(data, dstDir=dstDir, show=False, save=True, overwrite=False)
        # createProjectedDataHistograms(data, dstDir=dstDir, show=False, save=True, overwrite=False)

        # saveDistances(data, dstDir, save=True)

        # viewTransformedDataSurroundings(data, dstDir=dstDir)
        # countHexagones(data, dstDir=dstDir)
        # viewPointMaps(data, dstDir, hexagonsOnly=hexagonsOnly, debug=debug, overwrite=True)
        checkGridStructure(data, dstDir, hexagonsOnly=hexagonsOnly, debug=debug, overwrite=True)

        # createInterpolation(data, interFfp, interDstDir, hexagonsOnly=False, overwrite=False, debug=False)
        # createInterpolation(data, ffp, dstDir, overwrite=False)

        # createHeatmapForInterpolatedPoints(srcDir=srcDir,
        #                                    dstDir=dstDir)

        # print(d)
        print("Done with iteration: ", i)
        print("--------------")
        print()
    except Exception as e:
        print(traceback.format_exc())
        if debug:
            raise e
    se.end()

print("Finally done")
