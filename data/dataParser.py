import os

from helpers.timeTools import myTimer
from paths import DataDir

newHeaders = [0, 1, 2, 3, 4, 5, 6]


def createDataSubFolders():
    rawSamples = os.listdir(DataDir.rawSamples)

    dstPath = DataDir.cleanSamples
    os.chdir(dstPath)
    for rawS in rawSamples:
        dirname = rawS.split(".")[0]

        if os.path.isdir(dirname):
            print("Dir already exists: ", dirname)
        else:
            os.makedirs(name=dirname, exist_ok=False)
            print("Create new dir: ", dirname)
    print()
    return rawSamples


def parseDreieckDataCsv(src, dst, createSubFolders=False):
    print("Creating subfolders...")
    if createSubFolders:
        createDataSubFolders()

    print("Reading: ", src)

    assert os.path.isfile(src)

    srcFile = open(src, "r")

    readInOk = False
    newLines = list()
    while (line := srcFile.readline()):
        if line.__contains__("id"):
            readInOk = True
        if readInOk:
            newLine = line.replace("\t", ";")
            newLines.append(newLine)
    # print(newLines)
    srcFile.close()

    dstFile = open(dst, "w+")
    dstFile.writelines(newLines)
    dstFile.close()
    print("Done creating all Data")


def parseAllRawDreieckDataCsvs(srcDir=DataDir.rawSamples, dstDir=DataDir.cleanSamples):
    rawSamples = os.listdir(srcDir)
    N = len(rawSamples)
    parseTimer = myTimer(name="parseTimer", reportFrequency=10)
    for i in range(N):
        parseTimer.start()
        name = rawSamples[i]
        src = srcDir + os.sep + name
        dst = dstDir + os.sep + name
        parseDreieckDataCsv(src=src, dst=dst)
        parseTimer.end()


def parseAscFile(src):
    f = open(src, "r")

    lines = f.readlines()
    N = len(lines)

    newLines = list()
    M = len(newHeaders)

    for i in range(N):
        temp = lines[i]

        temp = temp.replace(";", "")
        temp = temp.split(" ")[:M]

        newLine = ""
        for j in range(M):
            newLine = newLine + temp[j] + ";"
        newLine = newLine[:-1] + "\n"
        newLines.append(newLine)

    return newLines
