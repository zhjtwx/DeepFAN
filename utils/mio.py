# -*- coding: utf-8 -*-
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')


import sys
import os
import os.path as osp
import codecs
import pickle
import json
import numpy
from scipy import misc
import glob
import logging
import logging.config
import time

def load_map_dir(map_file_path, sep=' '):
    """
    Load file like "path_complex path_compress"
    """
    map_file = open(map_file_path)
    map_dir = {}
    for line in map_file:
        line = line.strip()
        if len(line) == 0:
            continue
        elems = line.split(sep)
        map_dir[elems[0]] = elems[1]
    return map_dir

def print_detail(_file, message):
    print ('[%s | %s] %s' % (time.asctime(), _file, str(message)))

def iterate_files(root, include_post=(), exclude_post=()):
    """
    recurrently find files
    :param root:
    :param include_post: default=(), the list of the post of included file names, e.g., ['.dcm']
    :param exclude_post: default=(), the list of the post of excluded file names, e.g., ['.txt']
    :return:
    """
    assert osp.isdir(root),'%s is not a directory' % root
    result = []
    for root,dirs,files in os.walk(root, topdown=True):
        for fl in files:
            if len(include_post) != 0:
                if osp.splitext(fl)[-1] in include_post:
                    result.append(os.path.join(root,fl))
            else:
                if osp.splitext(fl)[-1] not in exclude_post:
                    result.append(os.path.join(root, fl))
    return result

def mkdir_safe(d):
    """
    Make Multi-Directories safety and thread friendly.
    """
    sub_dirs = d.split('/')
    cur_dir = ''
    max_check_times = 5
    sleep_seconds_per_check = 0.001
    for i in range(len(sub_dirs)):
        cur_dir += sub_dirs[i] + '/'
        for check_iter in range(max_check_times):
            if not os.path.exists(cur_dir):
                try:
                    os.mkdir(cur_dir.encode('gb2312'))
                except Exception as e:
                    print ('[WARNING] ', str(e))
                    time.sleep(sleep_seconds_per_check)
                    continue
            else:
                break


def pickle_save(path, obj):
    """
    Save obj as pickle format
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(path):
    """
    Load obj from pickle file
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
    return None


def json_save(path, obj):
    """
    Save obj as json format
    """
    with open(path, 'w') as f:
        f.write(json.dumps(obj))


def json_load(path):
    """
    Load obj from json file
    """
    with open(path, 'r') as f:
        return json.load(f)
    return None


def load_string_list(file_path, is_utf8=False):
    """
    Load string list from mitok file
    """
    try:
        if is_utf8:
            f = codecs.open(file_path, 'r', 'utf-8')
        else:
            f = open(file_path)
        l = []
        for item in f:
            item = item.strip()
            if len(item) == 0:
                continue
            l.append(item)
        f.close()
    except IOError:
        print ('open error', file_path)
        return None
    else:
        return l


def save_string_list(file_path, l, is_utf8=False):
    """
    Save string list as mitok file
    - file_path: file path
    - l: list store strings
    """
    if is_utf8:
        f = codecs.open(file_path, 'w', 'utf-8')
    else:
        f = open(file_path, 'w')
    for item in l[:-1]:
        f.write(item + '\n')
    if len(l) >= 1:
        f.write(l[-1])
    f.close()


def create_log_config(save_path, log_conf_path):
    lines = load_string_list(log_conf_path)
    new_lines = []
    for line in lines:
        new_lines.append(line.replace('xxx.log', save_path))
    save_dir = os.path.dirname(save_path)
    mkdir_safe(save_dir)
    save_string_list(save_dir + '/logging.conf', new_lines)
    return save_dir + '/logging.conf'

