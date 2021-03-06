#!/usr/bin/env python

"""Update parameters which directly depends on the dataset.
"""

__author__ = 'Anna Kukleva'
__date__ = 'November 2018'

import os

from utils.arg_pars import opt
from utils.utils import update_opt_str
from utils.logging_setup import path_logger


def update():
    opt.dataset_root = '/media/data/kukleva/lab/Breakfast'

    data_subfolder = ['feat/kinetics', 'feat/data', 'feat/s1', 'feat/video', opt.data][opt.data_type]
    opt.data = os.path.join(opt.dataset_root, data_subfolder)

    opt.gt = os.path.join(opt.data, opt.gt)

    opt.ext = ['gz', 'gz', 'txt', 'avi', opt.ext][opt.data_type]
    opt.feature_dim = [400, 64, 64, 0, opt.feature_dim][opt.data_type]

    opt.bg = False
    opt.zeros = False

    update_opt_str()

    logger = path_logger()

    vars_iter = list(vars(opt))
    for arg in sorted(vars_iter):
        logger.debug('%s: %s' % (arg, getattr(opt, arg)))

