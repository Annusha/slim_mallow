#!/usr/bin/env python

"""Implementation and improvement of the paper:
Unsupervised learning and segmentation of complex activities from video.
"""

__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import BF_utils.update_argpars as bf_utils
import YTI_utils.update_argpars as yti_utils
from corpus import Corpus
from utils.arg_pars import opt
from utils.logging_setup import logger
from utils.utils import timing, update_opt_str, join_return_stat, parse_return_stat


@timing
def baseline():
    """Implementation of the paper"""

    corpus = Corpus(Q=opt.gmm, subaction=opt.subaction)

    for iteration in range(7):
        logger.debug('Iteration %d' % iteration)
        corpus.iter = iteration
        corpus.accuracy_corpus()
        if (opt.gt_training and iteration == 0) or not opt.gt_training:
            corpus.embedding_training()
        # one version of gaussian mixtures for the entire dataset
        if opt.gmms == 'one':
            corpus.one_gaussian_model()
        # different gmm for different subsets of videos, i.e. leave one out for
        # each video subset
        elif opt.gmms == 'many':
            corpus.many_gaussian_models()
            # with multiprocessing package
            # corpus.gaussians_mp(n_threads=3)
        else:
            raise RuntimeError('define number of gmms for the video collection')

        if opt.viterbi:
            # corpus.viterbi_decoding()
            # corpus.accuracy_corpus(prefix='pure vit ')

            # corpus.viterbi_ordering()
            # take into account Mallow Model
            corpus.ordering_sampler()
            corpus.rho_sampling()
            # corpus.accuracy_corpus(prefix='vit+ord ')

            corpus.viterbi_decoding()
            # corpus.viterbi_alex_decoding()
        else:
            corpus.subactivity_sampler()

            # take into account Mallow Model
            corpus.ordering_sampler()
            corpus.rho_sampling()

    corpus.accuracy_corpus()


@timing
def all_actions():
    return_stat_all = None
    if opt.dataset == 'bf':
        actions = ['coffee', 'cereals', 'tea', 'milk', 'juice', 'sandwich', 'scrambledegg', 'friedegg', 'salat', 'pancake']
    if opt.dataset == 'yti':
        actions = ['changing_tire', 'coffee', 'jump_car', 'cpr', 'repot']
    lr_init = opt.lr
    for action in actions:
        opt.subaction = action
        if not opt.resume:
            opt.lr = lr_init
        update_opt_str()
        if opt.viterbi:
            return_stat_single = baseline(iterations=1)
        else:
            return_stat_single = baseline(iterations=5)
        return_stat_all = join_return_stat(return_stat_all, return_stat_single)
    parse_return_stat(return_stat_all)


def resume_segmentation(iterations=10):
    logger.debug('Resume segmentation')
    corpus = Corpus(Q=opt.gmm,
                    subaction=opt.subaction)

    for iteration in range(iterations):
        logger.debug('Iteration %d' % iteration)
        corpus.iter = iteration
        corpus.resume_segmentation()
        corpus.accuracy_corpus()
    corpus.accuracy_corpus()


if __name__ == '__main__':
    if opt.dataset == 'bf':
        bf_utils.update()
    if opt.dataset == 'yti':
        yti_utils.update()
    if opt.subaction == 'all':
        all_actions()
    else:
        baseline()
