#!/usr/bin/env python

"""Hyper-parameters and logging set up

opt: include all hyper-parameters
logger: unified logger for the project
"""

__all__ = ['opt', 'logger']
__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import argparse
import logging
import datetime
import sys
import re
from os.path import join

parser = argparse.ArgumentParser()

parser.add_argument('--log', default='DEBUG',
                    help='DEBUG | INFO | WARNING | ERROR | CRITICAL '
                         'changing_tire'
                         'coffee'
                         'jump_car'
                         'cpr'
                         'repot')

parser.add_argument('--subaction', default='coffee',
                    help='measure accuracy for different subactivities')

parser.add_argument('--n_clusters', default=20,
                    help='number of clusters')
parser.add_argument('--n_segments', default=100,
                    help='number of segments per video for uniform segmentation')

parser.add_argument('--dataset_root', default='/media/data/kukleva/lab/Breakfast',
                    help='root folder for dataset'
                         '/media/data/kukleva/lab/Breakfast'
                         '/media/data/kukleva/lab/YTInstructions')
parser.add_argument('--data', default='feat',
                    help='folders in between root and data folder:'
                         'feat - for Breakfast'
                         'dt/dt_FV/dt_l2pn_c64_pc64'
                         'VISION_txt')
parser.add_argument('--gt', default='groundTruth',
                    help='folder with ground truth labels:')
parser.add_argument('--feature_dim', default=64,
                    help='feature dimensionality')
parser.add_argument('--end', default='gz',
                    help='extension of feature files')
parser.add_argument('--zeros', default=False, type=bool,
                    help='if True there can be SIL label (beginning and end)'
                            'that is zeros labels relating to non action'
                         'if False then 0 labels are demolished from ground truth '
                            'labeling at all (SIL label), only for Breakfast dataset')
parser.add_argument('--bg', default=False, type=bool,
                    help='if we need to apply part for modeling background')
parser.add_argument('--save_likelihood', default=False, type=bool)

###########################################
# hyperparams parameters for embeddings


parser.add_argument('--seed', default=0,
                    help='seed for random algorithms, everywhere')
parser.add_argument('--lr', default=1e-6, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_adj', default=True, type=bool,
                    help='will lr be multiplied by 0.1 in the middle')
parser.add_argument('--momentum', default=0.9,
                    help='momentum')
parser.add_argument('--weight_decay', default=1e-4,
                    help='regularization constant for l_2 regularizer of W')
parser.add_argument('--batch_size', default=256,
                    help='batch size for training embedding (default: 40)')
parser.add_argument('--num_workers', default=4,
                    help='number of threads for dataloading')
parser.add_argument('--embed_dim', default=40, type=int,
                    help='number of dimensions in embedded space')
parser.add_argument('--epochs', default=12, type=int,
                    help='number of epochs for training embedding')
parser.add_argument('--resume', default=False, type=bool,
                    help='load model for embeddings, if positive then it is number of '
                         'epoch which should be loaded')
parser.add_argument('--resume_str',
                    # default='',
                    # default='!norm.!conc._%s_mlp_!pose_full_vae0_time10.0_epochs90_embed20_n2_ordering_gmm1_one_!gt_lr0.001_lr_!zeros_b0_v1_l0_c1_',
                    # default='grid.vit._%s_mlp_!pose_full_vae1_time10.0_epochs90_embed20_n2_ordering_gmm1_one_!gt_lr0.001_lr_zeros_b0_v1_l0_c1_',
                    # default='fixed.order._%s_mlp_!pose_full_vae0_time10.0_epochs60_embed20_n1_!ordering_gmm1_one_!gt_lr0.0001_lr_zeros_b0_v1_l0_c1_',
                    # default='norm.conc._%s_mlp_!pose_full_vae1_time10.0_epochs60_embed20_n1_ordering_gmm1_one_!gt_lr0.0001_lr_!zeros_b0_v1_l0_c1_',

                    # default='10cl.joined_full_mlp_!pose_full_vae0_time10.0_epochs45_embed35_n1_!ordering_gmm1_one_!gt_lr0.001_lr_zeros_b0_v1_l0_c1_',

                    default='yti.(200,90,-3)_%s_mlp_!pose_full_vae0_time10.0_epochs90_embed200_n4_!ordering_gmm1_one_!gt_lr0.001_lr_zeros_b1_v1_l0_c1_',

                    # default='rank._%s_rank_!pose_full_vae0_time10.0_epochs30_embed30_n2_!ordering_gmm1_one_!gt_lr1e-06_lr_!zeros_b0_v1_l0_c1_b0_',
                    # default='rank._%s_rank_!pose_full_vae0_time10.0_epochs30_embed30_n2_!ordering_gmm1_one_!gt_lr1e-06_lr_zeros_b0_v1_l0_c1_b96_',

                    help='grid.vit._coffee_mlp_!pose_full_vae1_time10.0_epochs30_embed40_n2_ordering_gmm1_one_!gt_lr0.001_lr_zeros_b0_v1_l0_c1_'
                         'grid.vit._coffee_mlp_!pose_full_vae1_time10.0_epochs60_embed30_n2_ordering_gmm1_one_!gt_lr0.01_lr_zeros_b0_v1_l0_c1_'
                         '10cl.relt.!idx_full_mlp_!pose_full_vae0_time10.0_epochs45_embed35_n1_!ordering_gmm1_one_!gt_lr0.001_lr_zeros_b0_v1_l0_c1_')
parser.add_argument('--gt_training', default=False, type=bool,
                    help='training embedding either with gt labels '
                         'or with labels gotten from the temporal model')
parser.add_argument('--save_feat', default=False,
                    help='save features after embedding trained on gt')
parser.add_argument('--gmm', default=1, type=int,
                    help='number of components for gaussians')
parser.add_argument('--gmms', default='one',
                    help='number of gmm for the video collection: many/one')
parser.add_argument('--ordering', default=True,
                    help='apply Mallow model for incorporate ordering')
parser.add_argument('--shuffle_order', default=False, type=bool,
                    help='shuffle or order wrt relative time cluster labels after clustering')
parser.add_argument('--kmeans_shuffle', default=False, type=bool,
                    help='auto shuffle after kmeans or'
                         'shuffle enforced numpy with given seed')

parser.add_argument('--n_d', default=2, type=int,
                    help='0: kinetics,'
                         '1: data, /norm/'
                         '2: s1, /wo norm/'
                         '3: video,'
                         '4: YTI original')

###########################################
# poses

parser.add_argument('--pose_path', default='/media/data/kukleva/lab/Breakfast/'
                                           'videos/video_posesbla',
                    help='path to folder containing folders for each video'
                         ' with estimated poses')
parser.add_argument('--time_weight', default=10.0, type=float,
                    help='weighted concatenation')

###########################################
# vae

parser.add_argument('--vae_dim', default=0, type=int,
                    help='additional dimensionality for vae')
parser.add_argument('--label', default=False, type=bool,
                    help='features for training embedding is + concat with '
                         'uniform label')
parser.add_argument('--concat', default=1, type=int,
                    help='how much consecutive features to concatenate')

###########################################
# bg

bg_p = 0
parser.add_argument('--bg_trh', default=93, type=int)

###########################################
# viterbi

parser.add_argument('--viterbi', default=False, type=bool)

###########################################
# additional

parser.add_argument('--full', default=True, type=bool,
                    help='check smth using only 15 videos')
parser.add_argument('--tr_type', default='base',
                    help='define training: base, pose: mlp/vae, joined, nothing, rank')
parser.add_argument('--pose_segm', default=False, type=bool,
                    help='relative time assignment:'
                         'base on pose analysis -> segments'
                         'framewise')
parser.add_argument('--save_model', default=True, type=bool,
                    help='save embedding model after training')
parser.add_argument('--grid_search', default=False, type=bool,
                    help='grid search for optimal parameters')
parser.add_argument('--vis', default=False, type=bool,
                    help='save visualisation of embeddings')
parser.add_argument('--prefix', default='rank.',
                    help='prefix for log file')

###########################################
# rest
parser.add_argument('--log_str', default='',
                    help='unify all savings')
opt = parser.parse_args()

# data_pathes = ['/media/data/kukleva/lab/Breakfast/feat/kinetics',
#                '/media/data/kukleva/lab/Breakfast/feat/data',
#                '/media/data/kukleva/lab/Breakfast/feat/s1',
#                '/media/data/kukleva/lab/Breakfast/videos']

opt.gt = join(opt.dataset_root, opt.data, opt.gt)

data_pathes = [join(opt.dataset_root, opt.data, 'kinetics'),
               join(opt.dataset_root, opt.data, 'data'),
               join(opt.dataset_root, opt.data, 's1'),
               join(opt.dataset_root, opt.data, 'videos'),
               join(opt.dataset_root, opt.data)]

opt.data = data_pathes[opt.n_d]
opt.end = ['gz', 'gz', 'txt', 'avi', 'txt'][opt.n_d]
opt.feature_dim = [400, 64, 64, 0, 3000][opt.n_d]


# if opt.prefix == 'd':
#     opt.save_model = False
#     opt.vis = True

log_str = ''
logs_args = ['prefix', 'subaction', 'tr_type', 'pose_segm', 'full',
             'vae_dim', 'time_weight', 'epochs', 'embed_dim', 'n_d', 'ordering',
             'gmm', 'gmms', 'gt_training', 'lr', 'lr_adj', 'zeros']
for arg in logs_args:
    attr = getattr(opt, arg)
    arg = arg.split('_')[0]
    if isinstance(attr, bool):
        if attr:
            attr = arg
        else:
            attr = '!' + arg
    elif isinstance(attr, str):
        attr = attr.split('_')[0]
    else:
        attr = '%s%s' % (arg, str(attr))
    log_str += '%s_' % attr

# additional attributes which appeared later
additional = ['bg', 'viterbi', 'label', 'concat', 'bg_trh']
for arg in additional:
    attr = getattr(opt, arg)
    if isinstance(attr, int):
        attr = '%s%d' % (arg[0], attr)
    else:
        if attr:
            attr = arg
        else:
            attr = ''
    log_str += '%s_' % attr

opt.log_str = log_str


###########################################
# logging

logger = logging.getLogger('basic')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

filename = sys.argv[0]
search = re.search(r'\/*(\w*).py', filename)
filename = search.group(1)
mallow_model = ['-mm', ''][opt.ordering]

# mlp_onhePoseSegm_
pose_segm = ['!pose_', ''][opt.pose_segm]
full = ['sampls', 'full'][opt.full]
adlr = ['', 'adj_'][opt.lr_adj]

# fh = logging.FileHandler('/media/data/kukleva/lab/logs_debug/'
#                          '%s_%s_%s_%s_%s_%s%s_%d_%d_gmm(%d)_%slr%.1e_ep%d(%s)' %
#                          (opt.prefix, opt.tr_type, opt.subaction, filename,
#                           opt.tr_type, pose_segm, full, opt.n_d, opt.embed_dim,
#                           opt.gmm, adlr, opt.lr, opt.epochs,
#                           str(datetime.datetime.now())), mode='w')


path_logging = join(opt.dataset_root, 'logs', '%s%s(%s)' %
                    (opt.log_str, filename,
                     str(datetime.datetime.now())))
fh = logging.FileHandler(path_logging, mode='w')
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelno)s - %(filename)s - '
                              '%(funcName)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)
for arg in vars(opt):
    logger.debug('%s: %s' % (arg, getattr(opt, arg)))


