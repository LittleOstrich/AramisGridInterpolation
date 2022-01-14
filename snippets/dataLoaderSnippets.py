import os

from data.dataLoader import getAllPathes, dfToDictOfArrays, loadAllDataAsArrays, loadDataByPath
from paths import DataDir, MetadataDir


def findDataSnippet(sd, ff, debug=False):
    ds, fs = getAllPathes(sd=sd, filterFuntion=ff, debug=debug)
    return ds, fs


def test():
    src = DataDir.cleanSamples

    # dataNames = os.listdir(src)

    files = os.listdir(src)
    for f in files:
        ffp = src + os.sep + f
        df = loadDataByPath(ffp)
        d = dfToDictOfArrays(df)
        pass


test()
