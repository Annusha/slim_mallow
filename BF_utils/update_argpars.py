#!/usr/bin/env python

"""Update parameters which directly depends on the dataset.
"""

__author__ = 'Anna Kukleva'
__date__ = 'November 2018'

import os

from utils.arg_pars import opt
from utils.utils import update_opt_str


def update():
    data_subfolder = ['kinetics', 'data', 's1', 'video'][opt.data_type]
    opt.data = os.path.join(opt.dataset_root, 'feat', data_subfolder)

    opt.ext = ['gz', 'gz', 'txt', 'avi'][opt.data_type]
    opt.feature_dim = [400, 64, 64, 0][opt.data_type]

    update_opt_str()

