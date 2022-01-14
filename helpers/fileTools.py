import os
import time

from helpers.general_tools import removeDir


def writeLines(fn, dstDir, lines):
    ffp = dstDir + os.sep + fn
    f = open(ffp, "w+")
    for line in lines:
        f.write(line)
    f.close()


def cleanup(dstDir, overwrite=False):
    if os.path.exists(dstDir):
        if overwrite:
            removeDir(dstDir)
            time.sleep(1)
        else:
            return
    os.makedirs(dstDir, exist_ok=True)
