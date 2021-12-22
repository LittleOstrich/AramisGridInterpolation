import os
from datetime import datetime
import time
import numpy as np

from helpers import helperKeys


def transformDate_yyyymmdd(date):
    date = str(date)
    y = int(date[0:4])
    m = int(date[4:6])
    d = int(date[6:])

    date = datetime(year=y, month=m, day=d)
    return date


def addDateToFn(fn):  # kinda hacky..
    parts = fn.split(".")
    if len(fn) > 0:
        fn = parts[0] + "_"
    fn = fn + str(timeAsString()) + "." + parts[1]
    return fn


def timeAsString():
    t = time.localtime()
    current_time = time.strftime("%H_%M_%S", t)
    return current_time


def date():
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    return dt_string


class myTimer:

    def __init__(self, name, dst="temp", saveReport=False):
        self.name = name
        self.counter = 0
        self.starts = list()
        self.ends = list()
        self.on = False

        os.makedirs(dst, exist_ok=True)
        self.dst = dst

    def start(self):
        assert self.on == False
        self.counter = self.counter + 1
        s = time.time()
        self.starts.append(s)
        self.on = True

    def end(self):
        assert self.on == True
        e = time.time()
        self.ends.append(e)
        self.on = False
        self.report()

    def report(self, write=False):
        starts = np.array(self.starts)
        ends = np.array(self.ends)

        diffs = ends - starts

        avg = np.mean(diffs)
        total = np.sum(diffs)

        print("Total run time: ", total)
        print("Average run time: ", avg)

        if write:
            headers = helperKeys.timeToolKeys
            fn = addDateToFn(self.name)
            ffp = self.dst + os.sep + fn + ".csv"
            iterations = [helperKeys.timeToolKeys.iteration] + np.arange(len(diffs)).tolist()
            durations = [helperKeys.timeToolKeys.duration] + diffs.tolist()

            # listsToCsv(lists, dst="temp", name="runTimeReport", withDate=True)
