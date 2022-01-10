import os
import traceback
import sys
import numpy as np
import warnings
import logging

logging.getLogger().setLevel(logging.CRITICAL)

from backbone import alignData
from data.dataLoader import loadAllDataAsArrays, dataDfToArray, loadDataByPath
from helpers.general_tools import execFunctionOnMatrix, sciF
from helpers.timeTools import myTimer
from metadata.metadataHelper import findBestKForKMeans, countsByCluster, visualizeCounts, createProjectedDataHistograms, \
    computeCountsTest, createScattterPlots, saveDistances, randomHexagonSampling, detailedRandomHexagonSampling, \
    viewTransformedDataSurroundings, countHexagones, colourDataByShape, viewPointMaps, checkGridStructure
from paths import DataDir, metadataDir, metadataDirExtern

# src = DataDir.cleanSamples
# dstDir = metadataDir.sampleAnalysis
src = DataDir.cleanLFT1086
dstDir = metadataDirExtern.base

show = False
save = True
dpi = 500

datas = loadAllDataAsArrays(src=src, normalizeData=False)
# dataNames = os.listdir(src)
files = os.listdir(src)

d = dict()
d["show"] = show
d["save"] = save

N = len(files)
# numSamples = 100
se = myTimer("createMetadata", ".")
overwrite = False
# 85
for i in range(N):
    try:
        print("Starting iteration: ", i)
        ffp = src + os.sep + files[i]
        data = loadDataByPath(ffp)
        data = dataDfToArray(data, normalizeData=False)

        d["data"] = np.copy(data)
        d["dstDir"] = dstDir + os.sep + files[i][:-4]
        d["hexagonsOnly"] = False
        d["debug"] = False
        d["overwrite"] = overwrite
        # findBestKForKMeans(**d)
        # computeCountsTest(**d)
        # visualizeCounts(**d)
        # countsByCluster(**d)
        # createScattterPlots(**d)
        # createProjectedDataHistograms(**d)
        #
        # saveDistances(d["data"], d["dstDir"])
        se.start()
        # detailedRandomHexagonSampling(d["data"], d["dstDir"], numSamples=numSamples)
        # viewTransformedDataSurroundings(d["data"], d["dstDir"])
        # countHexagones(d["data"], d["dstDir"])
        # viewPointMaps(d["data"], d["dstDir"], d["hexagonsOnly"], d["debug"], d["overwrite"])
        checkGridStructure(d["data"], d["dstDir"], d["hexagonsOnly"], d["debug"], d["overwrite"])
        se.end()

        # print(d)
        print("Done with iteration: ", i)
        print("--------------")
        print()
    except Exception as e:
        print(traceback.format_exc())
print("Finally done")
