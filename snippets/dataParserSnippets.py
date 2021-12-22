import paths
import os
from data.dataLoader import getAllPathes
from data.dataParser import createDataSubFolders, parseAllRawDreieckDataCsvs


def createDataSubFoldersSnippet():
    ret = createDataSubFolders()
    print("Done with: createDataSubFoldersSnippet")


def parseAllRawDreieckDataCsvsSnippet():
    parseAllRawDreieckDataCsvs()


# createDataSubFoldersSnippet()
parseAllRawDreieckDataCsvsSnippet()
