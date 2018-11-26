#!/usr/bin/env python

"""Grid search funtions where can be defined and evaluated numerous of
parameters.
"""

__author__ = 'Anna Kukleva'
__date__ = 'October 2018'

from utils.arg_pars import opt, logger

actions = ['coffee', 'cereals', 'milk', 'tea', 'juice', 'sandwich', 'salat', 'friedegg', 'scrambledegg', 'pancake']


def grid_search(**kwargs):
    n_params = len(kwargs)
    f = kwargs['f']
    del kwargs['f']
    keys = list(kwargs.keys())

    line = '\n\nSET: '
    for key, val in kwargs.items():
        setattr(opt, key, val)
        if key == 'lr':
            subline = '%s: %.1e\t'
        elif isinstance(val, float):
            subline = '%s: %.3f\t'
        elif isinstance(val, int):
            subline = '%s: %d\t'
        else:  # assume everything else
            val = str(val)
            subline = '%s: %s\t'
        subline = subline % (key, val)

    for arg in vars(opt):
        logger.debug('%s: %s' % (arg, getattr(opt, arg)))




