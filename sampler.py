from __future__ import division
import os
import shutil
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import config as config
from torch.autograd import Variable
import numpy as np

import json
import io
import glob
import codecs
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
        print('open error %s' % file_path)
        return None
    else:
        return l

def weight_load(sampler_list_dir):
    weights = load_string_list(sampler_list_dir)
    if weights is not None:
        weights = [float(one_weight) for one_weight in weights]
    else:
        weights = []
    return weights