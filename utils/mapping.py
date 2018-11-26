#!/usr/bin/env python

""" Some processing of ground truth information.

gt: Load ground truth labels for each video and save in gt dict.
label2index, index2label: As well unique mapping between name of complex
    actions and their order index.
define_K: define number of subactions from ground truth labeling
"""

__all__ = ['ground_truth', 'gt_with_0', 'define_K', 'label2index', 'index2label',
           'order', 'order_with_0']
__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import os
import numpy as np
import copy
import pickle

from utils.arg_pars import opt
from utils.utils import timing, dir_check

label2index = {}
index2label = {}


def create_mapping():
    global label2index, index2label

    root = os.path.join(opt.gt, 'mapping')
    filename = 'mapping.txt'

    with open(os.path.join(root, filename), 'r') as f:
        for line in f:
            idx, class_name = line.split()
            idx = int(idx)
            label2index[class_name] = idx
            index2label[idx] = class_name
        if not opt.bg and -1 in label2index:
            # change bg label from -1 to positive number
            new_bg_idx = max(index2label) + 1
            del index2label[label2index[-1]]
            label2index[-1] = new_bg_idx
            index2label[new_bg_idx] = -1




ground_truth = {}
gt_with_0 = ground_truth
order = {}
order_with_0 = {}


def load_obj(name):
    path = os.path.join(opt.gt, 'mapping', '%s.pkl' % name)
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        return None


def save_obj(obj, name):
    dir_check(os.path.join(opt.gt, 'mapping'))
    path = os.path.join(opt.gt, 'mapping', '%s.pkl' % name)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


@timing
def load_gt():
    global ground_truth, order, gt_with_0, order_with_0

    ground_truth = load_obj('gt_with_0')
    order = load_obj('order_with_0')

    if ground_truth is None or order is None:
        ground_truth, order = {}, {}
        for filename in os.listdir(opt.gt):
            if os.path.isdir(os.path.join(opt.gt, filename)):
                continue
            with open(os.path.join(opt.gt, filename), 'r') as f:
                labels = []
                local_order = []
                curr_lab = -1
                start, end = 0, 0
                for line in f:
                    line = line.split()[0]
                    try:
                        labels.append(label2index[line])
                        if curr_lab != labels[-1]:
                            if curr_lab != -1:
                                local_order.append([curr_lab, start, end])
                            curr_lab = labels[-1]
                            start = end
                        end += 1
                    except KeyError:
                        break
                else:
                    # executes every times when "for" wasn't interrupted by break
                    ground_truth[filename] = np.array(labels)
                    # add last labels

                    local_order.append([curr_lab, start, end])
                    order[filename] = local_order
        save_obj(ground_truth, 'gt_with_0')
        save_obj(order, 'order_with_0')
    gt_with_0 = ground_truth
    #print(list(gt_with_0.keys()))
    order_with_0 = order


def rid_of_zeros():
    global gt_with_0, ground_truth, order_with_0, order

    gt_with_0 = copy.deepcopy(ground_truth)
    order_with_0 = copy.deepcopy(order)

    gt_temp = load_obj('gt')
    order_temp = load_obj('order')

    if gt_temp is None:
        for key, value in ground_truth.items():
            # uniq_vals, indices = np.unique(value, return_index=True)
            if value[0] == 0:
                for idx, val in enumerate(value):
                    if val:
                        value[:idx] = val
                        break
            if value[-1] == 0:
                for idx, val in enumerate(np.flip(value, 0)):
                    if val:
                        value[-idx:] = val
                        break
            assert 0 not in value
            ground_truth[key] = value
        save_obj(ground_truth, 'gt')
    else:
        ground_truth = gt_temp

    if order_temp is None:
        for filename, fileorder in order.items():
            label, start, end = fileorder[0]
            if label == 0:
                fileorder[0] = [fileorder[1][0], start, end]
            label, start, end = fileorder[-1]
            if label == 0:
                fileorder[-1] = [fileorder[-2][0], start, end]
        save_obj(order, 'order')
    else:
        order = order_temp


def define_K(subaction):
    """Define number of subactions from ground truth labeling

    Args:
        subaction (str): name of complex activity
    Returns:
        number of subactions
    """
    uniq_labels = set()
    for filename, labels in ground_truth.items():
        if subaction in filename:
            uniq_labels = uniq_labels.union(labels)
    if -1 in uniq_labels:
        return len(uniq_labels) - 1
    else:
        return len(uniq_labels)

def sparse_gt():
    global gt_with_0, ground_truth, order_with_0, order
    for key, val in ground_truth.items():
        sparse_segm = [i for i in val[::10]]
        ground_truth[key] = sparse_segm
    gt_with_0 = copy.deepcopy(ground_truth)



create_mapping()
load_gt()
if not opt.zeros:
    rid_of_zeros()
#
# if 'VISION' in opt.data:
#     sparse_gt()



