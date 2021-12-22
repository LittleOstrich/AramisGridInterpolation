import os

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


def parseDreieckDataCsv(src, dst):
    print("Creating subfolders...")
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
    print(newLines)
    srcFile.close()

    dstFile = open(dst, "w")
    dstFile.writelines(newLines)
    dstFile.close()
    print("Done creating all Data")


def parseAllRawDreieckDataCsvs():
    rawSamples = os.listdir(DataDir.rawSamples)
    N = len(rawSamples)
    for i in range(N):
        name = rawSamples[i]
        src = DataDir.rawSamples + os.sep + name
        dst = DataDir.cleanSamples + os.sep + name.split(".")[0] + os.sep + name
        parseDreieckDataCsv(src=src, dst=dst)


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
