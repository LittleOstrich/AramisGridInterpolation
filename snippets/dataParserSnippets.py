import paths
import os
from data.dataLoader import getAllPathes
from data.dataParser import createDataSubFolders, parseAllRawDreieckDataCsvs


def createDataSubFoldersSnippet():
    ret = createDataSubFolders()
    print("Done with: createDataSubFoldersSnippet")


def parseAllRawDreieckDataCsvsSnippet(srcDir, dstDir):
    parseAllRawDreieckDataCsvs(srcDir=srcDir, dstDir=dstDir)


# createDataSubFoldersSnippet()
parseAllRawDreieckDataCsvsSnippet(srcDir=paths.DataDir.rawLFT1086, dstDir=paths.DataDir.cleanLFT1086)
