#!/usr/bin/env python

"""Hyper-parameters and logging set up

opt: include all hyper-parameters
logger: unified logger for the project
"""

__all__ = ['opt']
__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import argparse


parser = argparse.ArgumentParser()

###########################################
# data
parser.add_argument('--subaction', default='all',  # ['changing_tire', 'coffee', 'jump_car', 'cpr', 'repot']
                    help='measure accuracy for different subactivities')
parser.add_argument('--dataset', default='bf',
                    help='Breakfast dataset (bf) or YouTube Instructional (yti)')
parser.add_argument('--data_type', default=4, type=int,
                    help='for this moment valid just for Breakfast dataset'
                         '0: kinetics - features from the stream network'
                         '1: data - normalized features'
                         '2: s1 - features without normalization'
                         '3: videos'
                         '4: new features, not specified earlier')


parser.add_argument('--dataset_root', default='/media/data/kukleva/lab/Breakfast',
                    help='root folder for dataset:'
                         'Breakfast / YTInstructions')
parser.add_argument('--data', default='feat',
                    help='direct path to your data features')
parser.add_argument('--gt', default='groundTruth',
                    help='folder with ground truth labels')
parser.add_argument('--feature_dim', default=64,
                    help='feature dimensionality')
parser.add_argument('--ext', default='gz',
                    help='extension of the feature files')


###########################################
# hyperparams parameters for embeddings
parser.add_argument('--seed', default=0,
                    help='seed for random algorithms, everywhere')
parser.add_argument('--lr', default=1e-10, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_adj', default=False, type=bool,
                    help='will lr be multiplied by 0.1 in the middle')
parser.add_argument('--momentum', default=0.9,
                    help='momentum')
parser.add_argument('--weight_decay', default=1e-4,
                    help='regularization constant for l_2 regularizer of W')
parser.add_argument('--batch_size', default=256,
                    help='batch size for training embedding (default: 40)')
parser.add_argument('--num_workers', default=4,
                    help='number of threads for dataloading')
parser.add_argument('--embed_dim', default=30, type=int,
                    help='number of dimensions in embedded space')
parser.add_argument('--epochs', default=12, type=int,
                    help='number of epochs for training embedding')
parser.add_argument('--gt_training', default=False, type=bool,
                    help='training embedding (rank model) either with gt labels '
                         'or with labels gotten from the temporal model')


###########################################
# probabilistic parameters
parser.add_argument('--gmm', default=1, type=int,
                    help='number of components for gaussians')
parser.add_argument('--gmms', default='one',
                    help='number of gmm for the video collection: many/one')
parser.add_argument('--reg_cov', default=1e-1, type=float,
                    help='gaussian mixture model parameter')
parser.add_argument('--ordering', default=True,
                    help='apply Mallow model to incorporate ordering')


###########################################
# bg
parser.add_argument('--bg', default=False, type=bool,
                    help='if we need to apply part for modeling background')
parser.add_argument('--bg_trh', default=85, type=int)


###########################################
# viterbi
parser.add_argument('--viterbi', default=False, type=bool)

###########################################
# save
parser.add_argument('--save_model', default=False, type=bool,
                    help='save embedding model after training')
parser.add_argument('--resume', default=False, type=bool,
                    help='load model for embeddings, if positive then it is number of '
                         'epoch which should be loaded')
parser.add_argument('--resume_str',
                    default='',
                    help='which model to load')
parser.add_argument('--save_likelihood', default=False, type=bool)
parser.add_argument('--save_embed_feat', default=False,
                    help='save features after embedding trained on gt')

###########################################
# additional
parser.add_argument('--full', default=True, type=bool,
                    help='check smth using only 15 videos')
parser.add_argument('--zeros', default=False, type=bool,
                    help='if True there can be SIL label (beginning and end)'
                         'that is zeros labels relating to non action'
                         'if False then 0 labels are erased from ground truth '
                         'labeling at all (SIL label), only for Breakfast dataset')
parser.add_argument('--vis', default=False, type=bool,
                    help='save visualisation of embeddings')
parser.add_argument('--prefix', default='slim.mallow.',
                    help='prefix for log file')

###########################################
# logs
parser.add_argument('--log', default='DEBUG',
                    help='DEBUG | INFO | WARNING | ERROR | CRITICAL')
parser.add_argument('--log_str', default='',
                    help='unify all savings')

opt = parser.parse_args()



