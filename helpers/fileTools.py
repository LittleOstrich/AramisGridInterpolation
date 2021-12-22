import os


def writeLines(fn, dstDir, lines):
    ffp = dstDir + os.sep + fn
    f = open(ffp, "w+")
    for line in lines:
        f.write(line)
    f.close()
