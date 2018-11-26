#!/usr/bin/env python

"""Slice sampling only for one dim functions.
 And commented versions are which I found in the git."""

__all__ = ['slice_sample']
__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import numpy as np


def slice_sample(init, burn_in, logpdf):
    x = init
    eps = 3
    max_width = 100
    for iter in range(burn_in):
        r = np.random.uniform(0, eps)
        u_ = np.random.uniform(0, logpdf(x))
        x_l = x - r
        x_r = x + (eps - r)
        lg_l = logpdf(x_l)
        while lg_l > u_ and abs(x_l - x_r) < max_width:
            lg_l = logpdf(x_l)
            x_l -= eps
        lg_r = logpdf(x_r)
        while lg_r > u_ and abs(x_l - x_r) < max_width:
            lg_r = logpdf(x_r)
            x_r += eps
        x_ = np.random.uniform(x_l, x_r)
        if logpdf(x_) > logpdf(x):
            x = x_
    return x


# def slice_sample(init, iters, sigma, step_out=True, joint_dist=None):
#     """
#     based on http://homepages.inf.ed.ac.uk/imurray2/teaching/09mlss/
#     """
#
#     dist = joint_dist
#
#     # set up empty sample holder
#     dim = len(init)
#     samples = np.zeros((dim, iters))
#
#     # initialize
#     xx = init.copy()
#
#     for i in range(iters):
#         perm = list(range(dim))
#         np.random.shuffle(perm)
#         last_llh = dist.logpdf(xx)
#
#         for _d in perm:
#             llh0 = last_llh + np.log(np.random.rand())
#             rr = np.random.rand(1)
#             x_l = xx.copy()
#             x_l[_d] = x_l[_d] - rr * sigma[_d]
#             x_r = xx.copy()
#             x_r[_d] = x_r[_d] + (1 - rr) * sigma[_d]
#
#             if step_out:
#                 llh_l = dist.logpdf(x_l)
#                 while llh_l > llh0:
#                     x_l[_d] = x_l[_d] - sigma[_d]
#                     llh_l = dist.logpdf(x_l)
#                 llh_r = dist.logpdf(x_r)
#                 while llh_r > llh0:
#                     x_r[_d] = x_r[_d] + sigma[_d]
#                     llh_r = dist.logpdf(x_r)
#
#             x_cur = xx.copy()
#             while True:
#                 xd = np.random.rand() * (x_r[_d] - x_l[_d]) + x_l[_d]
#                 x_cur[_d] = xd.copy()
#                 last_llh = dist.logpdf(x_cur)
#                 if last_llh > llh0:
#                     xx[_d] = xd.copy()
#                     break
#                 elif xd > xx[_d]:
#                     x_r[_d] = xd
#                 elif xd < xx[_d]:
#                     x_l[_d] = xd
#                 else:
#                     raise RuntimeError('Slice sampler shrank too far.')
#
#         # if i % 1000 == 0:
#         #     print('iteration', i)
#
#         samples[:, i] = xx.copy().ravel()
#
#     return samples


# def slice_sample(init, burn_in, logpdf):
#     x = init
#     eps = 1
#     max_pdf = -np.inf
#     x_return = init
#     for iter in range(burn_in):
#         r = np.random.uniform(0, eps)
#         if logpdf(x) > max_pdf:
#             max_pdf = logpdf(x)
#             x_return = x
#         u_ = np.random.uniform(0, logpdf(x))
#         # u_ = logpdf(x)
#         x_l = x - r
#         x_r = x + (eps - r)
#         lg_l = logpdf(x_l)
#         while lg_l > u_:
#             lg_l = logpdf(x_l)
#             x_l -= eps
#         lg_r = logpdf(x_r)
#         while lg_r > u_:
#             lg_r = logpdf(x_r)
#             x_r += eps
#         x = np.random.uniform(x_l, x_r)
#         # print(x, logpdf(x))
#     return x_return

