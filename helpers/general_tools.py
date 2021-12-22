import string

import inspect
import imageio
import numpy as np
import os
import pydicom
import sys
import time

import matplotlib.pyplot as plt
import random
import psutil
import shutil

import paths

file_sep = "_"


def get_memory_usage(message=None):
    if message is not None:
        print(message)
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print("psutil.cpu_percent: ", psutil.cpu_percent())
    print("psutil.virtual_memory: ", psutil.virtual_memory())  # physical memory usage
    print('memory % used:', psutil.virtual_memory()[2])
    print('memory use:', memoryUse)
    print("...........")


def create_random_id(k=10):
    random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=k))
    return random_id


def arrays_match(ps, qs):
    len_ps = len(ps)
    len_qs = len(qs)
    if len_ps != len_qs:
        print(len_ps, len_qs)
        assert False
    errors = list()
    for i in range(len_ps):
        if ps[i] != qs[i]:
            errors.append(i)
    encountered_errors = len(errors) != 0
    if encountered_errors:
        print("The following entries in the arrays don't match!")
        for i in errors:
            print(ps[i], qs[i])
    if not encountered_errors:
        print("No errors encountered!")
    return encountered_errors


def create_dictionary_by_keys_and_values(keys, values):
    dic = dict()
    n = len(keys)
    for i in range(n):
        dic[keys[i]] = values[i]
    return dic


def extract_filename(fp, type=True):
    fn = fp.split(os.sep)[-1]
    return fn


def load_img_pixel_array(fp):
    ds = pydicom.dcmread(fp)
    return ds.pixel_array


def get_patient_ids_by_befunde():
    temp_patient_ids = os.listdir(get_befunde_path())
    patient_ids = list()
    for temp_patient_id in temp_patient_ids:
        patient_id = temp_patient_id[0: -4]
        patient_ids.append(patient_id)
    return patient_ids


def create_filename(parts, file_type):
    """
    @param parts:
    @param file_type:
    @return:
    """
    s = ""
    for part in parts:
        s = s + part + file_sep
    s = s[:-1] + file_type
    return s


def get_befunde_path():
    befunde_path = "..\\Befunde"
    return befunde_path


def get_base_dcm_path():
    befunde_pathes = ["F:\\Breast-US-Mamm",
                      "E:\\Breast-US-Mamm",
                      "Befunde"]
    for p in befunde_pathes:
        if os.path.exists(p):
            return p
    raise Exception("Befunde files couldn't be located.")


def list_to_row(l, delim=";"):
    s = ""
    for i in l:
        s = s + str(i) + delim
    s = s[0:len(s) - 1] + "\n"
    s = s.replace("'", "")
    return s


def write_to_csv(fn, line):
    f = open(fn, "a+")
    f.write(line + "\n")
    f.close()


def pretty_print(args):
    N = len(args)
    for i in range(N):
        print(i, args[i])


def pretty_print_dic(dic: dict):
    N = len(dic)
    ak = list(dic.keys())
    for i in range(N):
        key = ak[i]
        print(str(key), dic[key])


def colourSpace(numColours=10):
    import matplotlib.cm as cm
    colors = cm.rainbow(np.linspace(0, 1, numColours))
    return colors


getColours = colourSpace
getColourSpace = colourSpace


def get_file_name_from_fp(fn):
    fn = fn.split(os.sep)
    fn = fn[-1]
    lpl = len(fn.split(".")[-1])
    fn = fn[:-(lpl + 1)]
    return fn


def create_empty_dict_by_keys(keys):
    dic = dict()
    for key in keys:
        dic[key] = None
    return dic


def get_default_images_output_dir():
    return "F:\\temp"


def save_img(fn, img):
    if not fn.endswith(".png"):
        fn = fn + ".png"
    img = np.array(img, dtype=np.uint8)
    imageio.imwrite(fn, img)


#
# def save_imgs(imgs, fn, img_titel=None):
#     n = len(imgs)
#     plt.figure(figsize=(n * 4, 4))
#
#     def format_cursor_data(data):
#         return "[" + str(data) + "]"
#
#     for i in range(n):
#         ax = plt.subplot(1, n, i + 1)
#         tmp = plt.imshow(imgs[i])
#         tmp.format_cursor_data = format_cursor_data
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#
#     if fn is None:
#         fn = get_default_images_output_dir() + "\\" + str(getTime())
#
#     if img_titel is not None:
#         plt.title(img_titel)
#
#     plt.savefig(fn, dpi=500)
#     plt.close()


def plot_img(imgs, titels=None):
    n = len(imgs)
    plt.figure(figsize=(n, 1))

    def format_cursor_data(data):
        return "[" + str(data) + "]"

    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        tmp = plt.imshow(imgs[i])
        tmp.format_cursor_data = format_cursor_data
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    keyboardClick = False
    while keyboardClick != True:
        keyboardClick = plt.waitforbuttonpress()

    plt.close()


def get_dirs(src):
    os.chdir(src)
    all_subdirs = [d for d in os.listdir(".") if os.path.isdir(d)]
    return all_subdirs


def get_indices_of_data_to_keep(old_list, pathes_to_remove):
    indices_to_keep = list()
    c = 0
    for p in old_list:
        if p not in pathes_to_remove:
            indices_to_keep.append(c)
        c = c + 1
    return indices_to_keep


def get_excluded():
    excluded = "data" + os.sep + "excluded.txt"
    excluded2 = ".." + os.sep + excluded
    if not os.path.exists(excluded):
        excluded = excluded2

    f = open(excluded, "r")
    lines_temp = f.readlines()
    lines = list()
    for line in lines_temp:
        line = line.strip()
        lines.append(line)
    return lines


def get_bad_images():
    f = open("temp/bad_images.txt", "r")
    excluded = list()
    for line in f.readlines():
        excluded.append(line.strip())
    return excluded


def add_id_to_fn(id, fn):
    fn = id + "_" + fn
    return fn


def sort_by_indices(indices, *arrays):
    l = list()
    for i in range(len(arrays)):
        arg = arrays[i]
        arg = [arg[i] for i in indices]
        l.append(arg)
    return l


def get_size(s, l):
    size = sys.getsizeof(l)
    print(s + " ", size)


def write_dic_to_file(fn, dic):
    f = open(fn, "a+")
    for key in dic.keys():
        s = str(key) + "=" + str(dic[key]) + "\n"
        f.write(s)
    f.close()


def debug_dic(dic, debug=1):
    for key in dic.keys():
        print(key, dic[key])


def create_dic_by_dic_and_keys(keys, old_dic):
    new_dic = dict()
    for key in keys:
        new_dic[key] = old_dic.get(key, "'None'")
    return new_dic


def copy_report(src, dst):
    import shutil
    shutil.copyfile(src, dst)


def convert_string_list_to_type(li, t):
    ret = list()

    for elem in li:
        elem = t(elem)
        ret.append(elem)
    return ret


def create_filename_by_list(args, ft):
    s = ""
    for arg in args:
        s = s + str(arg) + "_"
    s = s[:-1] + ft
    return s


def class_attributes(clazz):
    attributes = inspect.getmembers(clazz, lambda a: not (inspect.isroutine(a)))
    ret = [a[1] for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))]
    return ret


# key = att_value, value = att_identifier
# A = "att_a"
# a = 'att_a': 'A'
def class_attributes_values(clazz):
    d = clazz.__dict__
    attributes = inspect.getmembers(clazz, lambda a: not (inspect.isroutine(a)))
    attributes = [a[0] for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))]
    values = list()
    for k in d.keys():
        v = d[k]
        if v in attributes:
            values.append(v)
    return values


def class_attributes_keys(clazz):
    d = clazz.__dict__
    attributes = inspect.getmembers(clazz, lambda a: not (inspect.isroutine(a)))
    attributes = [a[1] for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))]
    keys = list()
    for k in d.keys():
        if k in attributes:
            keys.append(k)
    return keys


def class_attributes_as_dict(clazz):
    attributes = inspect.getmembers(clazz, lambda a: not (inspect.isroutine(a)))
    tuples = [a for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))]
    d = clazz.__dict__
    for t in tuples:
        d[t[1]] = t[0]
    return d


def clean_up_folders(folders):
    for folder in folders:
        assert ("tools" not in folder)
        if not os.path.isdir(folder):
            continue
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


def remove_files(files):
    for file in files:
        if os.path.isfile(file):
            os.remove(file)


def removeDir(src):
    try:
        shutil.rmtree(src)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (src, e))
        raise e


def arrayify(*args):
    ret = list()
    for arg in args:
        arg = np.asarray(arg, dtype=np.uint8)
        ret.append(arg)
    return ret


def sciF(x, k=5):
    ret = np.round(x, k)
    return ret


def execFunctionOnMatrix(X, f, args):
    N = len(X)
    M = len(X[0])

    for i in range(N):
        for j in range(M):
            x = X[i, j]
            X[i, j] = f(x, **args)
    return X


def print_shapes(*xs):
    for x in xs:
        print(np.shape(x))


def build_relative_path(args):
    path = os.getcwd()
    for arg in args:
        path = path + os.sep + arg
    os.makedirs(path, exist_ok=True)
    return path


#
# def build_abs_path_ubuntu(parts):
#     r = build_abs_path(parts)
#     r = "/" + r.replace("\\", "/")
#     return r


def convert_seconds_to_hh_mm_ss(t):
    ct = time.strftime("%H:%M:%S", time.gmtime(t))
    # ct = str(timedelta(seconds=t))
    return ct


def build_filename(args, dst, type=None):
    path = dst + os.sep
    for arg in args:
        path = path + arg + "_"
    path = path[:-1]
    if type is not None:
        path = path + type
    return path


def count(arg, debug=False):
    unique, counts = np.unique(arg, return_counts=True)
    d = dict(zip(unique, counts))

    if debug:
        for key in d.keys():
            print("Key: ", key, ":", "Val: ", d[key])
    print("--------------")
    return unique, counts


def recursiveWalk(sD=".", filterFuntion=None, debug=False):
    fs = set()
    ds = set()
    for root, dirs, files in os.walk(sD):
        for name in files:
            fp = os.path.join(root, name)
            if filterFuntion is None:
                fs.add(fp)
            else:
                if filterFuntion(fp):
                    fs.add(fp)
        for dir in dirs:
            dp = os.path.join(root, dir)
            if filterFuntion is None:
                ds.add(dp)
            else:
                if filterFuntion(dp):
                    ds.add(dp)
    ds = list(ds)
    fs = list(fs)
    if debug:
        for d in ds:
            print(d)
        for f in fs:
            print(f)
    return ds, fs
