import paths
from data.dataLoader import getAllPathes


def findDataSnippet(sd, ff, debug=False):
    ds, fs = getAllPathes(sd=sd, filterFuntion=ff, debug=debug)
    return ds, fs
