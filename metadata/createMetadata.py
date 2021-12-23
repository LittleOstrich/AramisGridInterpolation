import os

import numpy as np

from backbone import alignData
from data.dataLoader import loadAllDataAsArrays
from helpers.general_tools import execFunctionOnMatrix, sciF
from helpers.timeTools import myTimer
from metadata.metadataHelper import findBestKForKMeans, countsByCluster, visualizeCounts, createProjectedDataHistograms, \
    computeCountsTest, createScattterPlots, saveDistances, randomHexagonSampling, detailedRandomHexagonSampling, \
    viewTransformedDataSurroundings, countHexagones, colourDataByShape, viewPointMaps
from paths import DataDir, metadataDir

show = False
save = True
dpi = 500
src = DataDir.cleanSamples

datas = loadAllDataAsArrays(src=src, normalizeData=False)
dataNames = os.listdir(src)

d = dict()
d["show"] = show
d["save"] = save

N = len(datas)
numSamples = 100
se = myTimer("something", ".")

for i in range(N):
    print("Starting iteration: ", i)
    data = datas[i]

    d["data"] = np.copy(data)
    d["dstDir"] = metadataDir.sampleAnalysis + os.sep + dataNames[i][:-4]

    # findBestKForKMeans(**d)
    # computeCountsTest(**d)
    # visualizeCounts(**d)
    # countsByCluster(**d)
    # createScattterPlots(**d)
    # createProjectedDataHistograms(**d)
    #
    # saveDistances(d["data"], d["dstDir"])
    # detailedRandomHexagonSampling(d["data"], d["dstDir"], numSamples=numSamples)
    # viewTrans,formedDataSurroundings(d["data"], d["dstDir"])
    # countHexagones(d["data"], d["dstDir"])
    se.start()
    viewPointMaps(d["data"], d["dstDir"])
    se.end()
    
    print(d["dstDir"])
    print("Done with iteration: ", i)
    print("--------------")
    print()

print("Finally done")
