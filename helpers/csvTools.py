import os
import traceback
from typing import List

import pandas as pd

from helpers.general_tools import copy_report
from helpers.timeTools import addDateToFn


def list_to_csv(vals, delim=";"):
    try:
        s = ""
        for i in vals:
            s = s + str(i) + delim
        s = s[0:len(s) - 1] + "\n"
    except Exception as e:
        excep = str(traceback.format_exc())
        print(vals)
        print(excep)
        raise e
    return s


def listsGet(lists, j, i):
    try:
        return str(lists[j][i])
    except:
        return str(-1)


def listsToCsv(lists, dst="temp" + os.sep + "defaultName.csv", withDate=False, delim=";", debug=False,
               deleteIfExists=False):
    numKeys = len(lists)
    numRows = len(lists[0])

    ffp = dst

    if ffp.endswith(".csv"):
        pass
    else:
        ffp = ffp + ".csv"

    if withDate:
        ffp = addDateToFn(ffp)

    if deleteIfExists:
        os.remove(ffp)

    f = open(ffp, "a+")
    for i in range(numRows):
        line = ""
        for j in range(numKeys):
            line = line + listsGet(lists, j, i) + delim
        line = line[:-1]
        f.write(line + "\n")
    f.close()
    if debug:
        print("Writing to: ", ffp)
    return ffp


def match_row_to_dataframe(row, df: pd.DataFrame, keys):
    matches = df.copy()
    for key in keys:
        matches = matches[matches[key] == row[key]]
    return matches


def select_rows_by_dict(df: pd.DataFrame, d: dict):
    matches = df.copy()
    matches = matches.applymap(str)
    keys = d.keys()

    for key in keys:
        vals = d[key]
        col = matches[key]
        if isinstance(vals, str):
            matches = matches[col == vals]
        elif isinstance(vals, List):
            matches = matches[col.isin([*vals])]
    return matches


def match_row_to_dataframe2(row, df: pd.DataFrame, keys):
    matches = pd.DataFrame(df.keys().tolist())
    N = len(df)
    for i in range(N):
        row2 = df.loc[i]
        add = True
        for key in keys:
            val1 = row[key]
            val2 = row2[key]
            if val1 != val2:
                add = False
        if add:
            matches.append(row)
    return matches


def readCsv(fn, header=0, delimiter=";", engine='python'):
    try:
        df = pd.read_csv(fn, header=header, delimiter=delimiter, engine=engine)
    except Exception as e:
        print("Arguments: ", fn, header, delimiter, engine)
        raise e
    return df


loadDataframe = readCsv


def writeCsv(df: pd.DataFrame, dst, fn):
    df.to_csv("n_clusters.csv", sep=";")


def writeDataframeToXlsx(df: pd.DataFrame, dst, fn):
    if fn.endswith(".csv"):
        pass
    else:
        fn = fn + ".csv"
    fp = dst + os.sep + fn
    df.to_xlsx(fp, index=False)


def csv_to_xlsx(src, dst=None, debug=False):
    assert src.endswith(".csv")
    df = pd.read_csv(src, header=0, delimiter=";", engine='python')

    if dst is None or src == dst:
        dst = src[:-4] + ".xlsx"
    if debug:
        print("Writing to: ", dst)
    try:
        df.to_excel(dst, index=False)
    except Exception as e:
        print("Ups.. likely the excel workbook package is not installed...")
        print("No xlsx was created..")
        print(e)


def finish_up_report_files(fn, fn2):
    copy_report(fn, fn2)
    csv_to_xlsx(fn)
    csv_to_xlsx(fn2)


def dict_to_csv_string(dic: dict, delim=";"):
    row = ""
    for key in dic.keys():
        val = dic[key]
        row = row + val + delim
    return row[:-1]


def append_row_to_csv(fn, row, delim=";", debug=0):
    f = open(fn, "a+")
    written = list_to_csv(row, delim)
    f.write(written)
    f.close()
    return written


def append_list_to_csv(fn, headers, delim=";", debug=0):
    f = open(fn, "a+")
    written = list_to_csv(headers, delim)
    f.write(written)
    f.close()
    return written


def rows_to_csv(fname, rows, delim=","):
    for row in rows:
        append_row_to_csv(fname, row, delim=";")


def append_row_to_dst(fn, dst, row, delim=";", debug=0):
    fn = dst + os.sep + fn
    f = open(fn, "a+")
    written = list_to_csv(row, delim)
    f.write(written)
    f.close()
    return written


def append_dict_to_csv(fn, dic: dict, delim=";", out=False):
    row = list()
    for key in dic.keys():
        val = dic[key]
        row.append(val)
    if out:
        print(row)
    append_row_to_csv(fn, row, delim)
    return row


def csv_to_list_of_dicts(fn):
    l = list()

    df = pd.read_csv(fn, header=0, delimiter=";")
    keys = df.keys()
    N = df.last_valid_index()

    for i in range(N):
        d = dict()
        for key in keys:
            col = df[key]
            val = col[i]
            d[key] = val
        l.append(d)
    return l


def append_incomplete_dict_to_csv_by_headers(fn, dic: dict, headers, delim=";", out=False):
    row = list()
    for key in headers:
        val = dic.get(key, 'None')
        row.append(val)
    if out:
        print(row)
    append_row_to_csv(fn, row, delim)
    return row


def append_dict_to_csv_by_keys(fn, dic: dict, headers, delim=";", out=False):
    row = list()
    for key in headers:
        val = dic.get(key, "'None'")
        row.append(val)
    if out:
        print(row)
    append_row_to_csv(fn, row, delim)
