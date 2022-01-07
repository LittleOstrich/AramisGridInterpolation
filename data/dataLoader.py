import os

from data.dataKeys import viereckRasterHeaders, dreieckRasterHeaders
from helpers.csvTools import loadDataframe, readCsv
from helpers.general_tools import recursiveWalk
from paths import projectBase, DataDir
import numpy as np


def loadAllDataAsDfs(src, N=100):
    files = os.listdir(src)
    dfs = list()

    for f in files:
        ffp = src + os.sep + f

        if ffp.endswith(".csv"):
            df = readCsv(ffp)
            dfs.append(df)

        if len(dfs) == N:
            break
            
    return dfs


def loadAllDataAsArrays(src, normalizeData=True, N=100):
    dfs = loadAllDataAsDfs(src=src)
    arrs = list()
    for df in dfs:
        arr = dataDfToArray(df, normalizeData)
        arrs.append(arr)
        if len(arrs) == N:
            break

    return arrs


def dataDfToArray(df, normalizeData=True):
    x = df[dreieckRasterHeaders.x].tolist()
    y = df[dreieckRasterHeaders.y].tolist()
    z = df[dreieckRasterHeaders.z].tolist()

    N = len(df)

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    if normalizeData:
        x = x / np.max(np.abs(x))
        y = y / np.max(np.abs(y))
        z = z / np.max(np.abs(z))

    data = np.zeros((N, 3))
    data[:, 0] = x
    data[:, 1] = y
    data[:, 2] = z

    return data


def getAllPathes(sd=".", filterFuntion=None, debug=False):
    ds, fs = recursiveWalk(sd, filterFuntion, debug=debug)
    return ds, fs
