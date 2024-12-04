# -*- coding: UTF-8 -*-

import itertools
import json
import logging
import os
import shutil
import sys
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interp
from sklearn.metrics import auc
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.preprocessing import scale, MinMaxScaler

keywords = ('mask', 'image', 'label', 'dataset', 'pid', "Mask", "Image", "Set1", "Set2", "Set3", "Set4", "Set5")


def display_image(image):
    plt.imshow(image, cmap='gray')
    plt.show()


def makedir_ignore(p):
    if not os.path.exists(p):
        os.makedirs(p)


def makedir_delete(p):
    if os.path.exists(p):
        shutil.rmtree(p)
    os.makedirs(p)


def array2dict(arr1, arr2):
    d = {}
    for k, v in zip(arr1, arr2):
        d[k] = v
    return d


def dict_layer_switch(d):
    r = defaultdict(dict)
    for k1, v1 in d.items():
        for k2, v2 in v1.items():
            r[k2][k1] = v2
    return r


def preprocessing(df):
    """
    预处理，去除每一列的空值，并将非数值转化为数值型数据，分两步
    1. 如果本列含有null。
        - 如果是number类型
            如果全为空，则均置零；
            否则，空值的地方取全列平均值。
        - 如果不是number类型
            将空值置NA
    2. 如果本列不是数值型数据，则用label encoder转化为数值型
    :param df: dataframe
    :return: 处理后的dataframe
    """

    def process(c):
        if c.isnull().any().any():
            if np.issubdtype(c.dtype, np.number):
                new_c = c.fillna(c.mean())
                if new_c.isnull().any().any():
                    return pd.Series(np.zeros(c.size))
                return new_c
            else:
                return pd.Series(LabelEncoder().fit_transform(c.fillna("NA").values))
        else:
            if not np.issubdtype(c.dtype, np.number):
                return pd.Series(LabelEncoder().fit_transform(c.values))
        return c

    pre_df = df.copy()
    return pre_df.apply(lambda col: process(col))


def scale_on_feature(data):
    """
    对每一列feature进行归一化，使方差一样

    :param data: dataframe
    :return: 归一化后的dataframe
    """
    data_scale = data.copy()
    data_scale[data.columns] = scale(data_scale)
    return data_scale


def scale_on_min_max(data, feature_range=(0, 1)):
    """
    对每一列feature进行相同区间归一化，使方差一样

    :param data:
    :param feature_range: dataframe
    :return: 归一化后的dataframe
    """
    data_scale = data.copy()
    scaler = MinMaxScaler(feature_range=feature_range)
    data_scale[data.columns] = scaler.fit_transform(data_scale)
    return data_scale


def prepare_target(target, nb=None, map_dict=None, method="map"):
    prepared = np.array(target).copy()

    is_numeric = np.issubdtype(prepared.dtype, np.number)

    if method == "map":
        # map labels to other labels
        prepared = np.array(map(lambda x: map_dict[x], list(prepared)))
    elif method == "size":
        try:
            cutted = pd.qcut(prepared, nb)
            prepared = cutted.code
        except Exception:
            return -1
    elif method == "range":
        cutted = pd.cut(prepared, nb)
    else:
        raise Exception

    return prepared


# prepare_target([1,1,1,11.1], map_dict={1: 111, 11.1:"p"})
# prepare_target([1,1,2,3,3,3], nb=2, method='size')

def encode_l(label):
    le = LabelEncoder().fit(label)
    el = le.transform(label)
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    return le, el, mapping


def encode_b(label):
    le = LabelBinarizer().fit(label)
    el = le.transform(label)
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    return le, el, mapping


def save_json(content, file_path):
    with open(file_path, 'w') as f:
        json.dump(content, f, sort_keys=True, indent=4)


def load_json(file_path):
    data = None
    with open(file_path) as f:
        data = json.load(f)
    return data


def reverse_dict(my_map):
    if sys.version_info[0] == 2:
        inv_map = {v: k for k, v in my_map.iteritems()}
    else:
        inv_map = {v: k for k, v in my_map.items()}
    return inv_map


def prepare_feature_n_label(df_feature, df_label, tags=None, key="mask"):
    # drop NA
    merged = pd.merge(df_feature, df_label.dropna()[['mask', 'label']], on=key)
    if tags is not None:
        merged = pd.merge(merged, tags)
        tv_label = list(set(merged[merged.dataset == 0].label.tolist()))
        merged = merged[merged.label.isin(tv_label)]
    feature = merged[[x for x in merged.columns if x not in ["label", "dataset"]]]
    label = merged[["label"]]
    if tags is not None:
        new_tags = merged[["dataset"]]
        return feature, label, new_tags
    return feature, label


def choose_feature(feature_file, use_pyradiomics=True):
    feature_classes = ['glcm',
                       'gldm',
                       'glrlm',
                       'glszm',
                       'ngtdm',
                       'shape',
                       'firstorder']

    if not use_pyradiomics:
        feature_classes = [
            "glcm",
            "glrlm",
            "shape",
            "firstorder"
        ]

    df = pd.read_csv(feature_file)
    columns = df.columns
    valid_columns = ['image', 'mask'] + [x for x in columns if len([y for y in feature_classes if y in x[:len(y)]]) > 0]
    return df[valid_columns]


class info_filter(logging.Filter):
    def __init__(self, name):
        super(info_filter, self).__init__(name)
        self.level = logging.WARNING

    def filter(self, record):
        if record.levelno >= self.level:
            return True
        if record.name == self.name and record.levelno >= logging.INFO:
            return True
        return False


def get_compact_range(mask_arr):
    if np.sum(mask_arr) < 1:
        raise ValueError('Zero mask.')
    xyz = np.nonzero(mask_arr)
    xmin = np.min(xyz[0])
    xmax = np.max(xyz[0])
    ymin = np.min(xyz[1])
    ymax = np.max(xyz[1])
    zmin = np.min(xyz[2])
    zmax = np.max(xyz[2])
    result = ([xmin, xmax], [ymin, ymax], [zmin, zmax])
    return result


def plot_confusion_matrix(cm,
                          classes,
                          save_path,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    # plt.figure(2)
    plt.figure(num=200, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    matplotlib.rc('font', **{'size': 12})

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.clf()


def roc_for_cv(fpr_arr, tpr_arr, class_name, save_path, fold_nb=5):
    plt.figure(num=300, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    matplotlib.rc('font', **{'size': 12})

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for idx in range(fold_nb):
        fpr, tpr = fpr_arr[idx], tpr_arr[idx]
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='Fold %d (AUC = %0.3f)' % (idx, roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='orange',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([0, 1.])
    plt.ylim([0, 1.])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic - class ' + str(class_name))
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=300)
    plt.clf()


def roc_for_class(fpr_arr, tpr_arr, class_name, save_path):
    plt.figure(100, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    matplotlib.rc('font', **{'size': 12})

    for i in range(len(fpr_arr)):
        fpr, tpr = fpr_arr[i], tpr_arr[i]
        plt.plot(fpr, tpr, lw=2, label='Class %s ROC curve (AUC = %0.3f)' % (class_name, auc(fpr, tpr)))
    plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--', alpha=.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=300)
    plt.clf()


def roc_for_clfs(fpr_arr, tpr_arr, clf_names, title, save_path):
    fig = plt.figure(100, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot()
    matplotlib.rc('font', **{'size': 12})
    ax.set_xlabel('1-Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title(title)
    auc_by_clfs = dict()
    for i in range(len(clf_names)):
        fpr, tpr = fpr_arr[i], tpr_arr[i]
        clf_name = clf_names[i]
        auc_i = auc(fpr, tpr)
        auc_by_clfs[clf_name] = auc_i
        plt.plot(fpr, tpr, lw=2, label='%s AUC=%0.3f' % (clf_name, auc_i))
    plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--', alpha=.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=300)
    plt.clf()
    return auc_by_clfs


def bar_chart(ys, xlabels, ylabel=None, title='Bar chart', save_path=None):
    fig = plt.figure(100, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot()
    ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    plt.bar(range(len(ys)), ys, color='rgb', tick_label=xlabels)
    plt.legend(loc="lower right")
    if save_path is not None:
        plt.savefig(save_path, dpi=600)
    plt.clf()


def multibar_chart(num_lists, cates, xlabels, title='Bar chart', save_path=None, xlabel_rot=0):
    fig = plt.figure(100, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot()
    ax.set_title(title)
    x = list(range(len(num_lists[0])))
    xticks = np.asarray(x)
    total_width, n = 0.35, len(cates)
    width = total_width / n
    plt.bar(x, num_lists[0], width=width, align='center', label=cates[0], alpha=0.8)
    for sli in range(len(cates)):
        if sli == 0:
            continue
        num_list = num_lists[sli]
        c = cates[sli]
        for i in range(len(x)):
            x[i] += width
        plt.bar(x, num_list, width=width, align='center', label=c, alpha=0.8)
    plt.xticks(xticks + width / 2, xlabels, rotation=xlabel_rot)
    if None not in cates:
        plt.legend()
    if save_path is not None:
        plt.savefig(save_path, dpi=600)
    plt.close()


def curve_with_xlabels(y, xlabels, cates, title='', save_path=None):
    """
    y shape should be (s1, s2)
    """
    xticks = range(y.shape[-1])
    # fig = plt.figure(100, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(title)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=90)
    for sli in range(y.shape[0]):
        plt.plot(xticks, y[sli], label=cates[sli])
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, dpi=600)
    plt.close()


def bar_comparision_chart(num_lists, cates, xlabels, title='', save_path=None):
    fig = plt.figure(100, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot()
    ax.set_title(title)
    for sli, c in enumerate(cates):
        plt.bar(x=xlabels, height=num_lists[sli], label=c, alpha=0.8)
        for x, y in enumerate(num_lists[sli]):
            plt.text(x, y, '%s' % y, ha='center', va='bottom')
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, dpi=600)
    plt.close()
