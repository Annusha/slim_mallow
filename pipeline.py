#!/usr/bin/env python

"""Implementation and improvement of the paper:
Unsupervised learning and segmentation of complex activities from video.
"""

__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

from corpus import Corpus
from utils.mapping import define_K
from utils.arg_pars import opt, logger
from utils.utils import timing, update_opt_str, join_return_stat, parse_return_stat


@timing
def baseline():
    """Implementation of the paper"""
    # default subaction is coffee

    # to train gt embedding and test all iterations on it without retraining it
    # on subgt segmentation
    # if opt.save:
    #     if opt.resume == 0:
    #         opt.epochs = 30
    #         training_embed.pipeline(mnist=False)
    #         opt.resume = 29
    #         training_embed.pipeline(mnist=False)

    corpus = Corpus(Q=opt.gmm, K=define_K(opt.subaction), subaction=opt.subaction)

    for iteration in range(5):
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

        # corpus.check_gaussians()s
        corpus.subactivity_sampler()

        # take into account Mallow Model
        corpus.ordering_sampler()
        corpus.rho_sampling()
    corpus.accuracy_corpus()


@timing
def pose_training(iterations=1):
    corpus = Corpus(Q=opt.gmm,
                    K=define_K(opt.subaction),
                    subaction=opt.subaction,
                    poses=True)

    # corpus = Corpus(Q=opt.gmm,
    #                 K=15,
    #                 subaction=opt.subaction,
    #                 poses=True)

    # corpus._K -=1
    logger.debug('Corpus with poses created')
    if opt.tr_type == 'mlp':
        corpus.regression_training()
    if opt.tr_type == 'vae':
        corpus.vae_training()
    if opt.tr_type == 'nothing':
        corpus.without_temp_emed()
    if opt.tr_type == 'rank':
        corpus.embedding_training()

    # corpus._K += 1

    if opt.embed_dim > opt.feature_dim:
        corpus.pca()

    corpus.clustering()

    # corpus.one_gaussian_model()

    for iteration in range(iterations):
        logger.debug('Iteration %d' % iteration)
        corpus.iter = iteration

        corpus.one_gaussian_model()
        corpus.accuracy_corpus()

        # if opt.resume:
        if False:
            corpus.resume_segmentation()
        else:
            if opt.viterbi:
                # corpus.viterbi_decoding()
                # corpus.accuracy_corpus(prefix='pure vit ')

                # corpus.viterbi_ordering()
                corpus.ordering_sampler()
                # corpus.rho_sampling()
                # corpus.accuracy_corpus(prefix='vit+ord ')


                corpus.viterbi_decoding()
                # corpus.viterbi_alex_decoding()
            else:
                corpus.subactivity_sampler()

                corpus.ordering_sampler()
                corpus.rho_sampling()
        # opt.bg_trh += 0.1
    frames = corpus.accuracy_corpus('final')

    return corpus.return_stat


@timing
def joined_training():
    corpus = pose_training(iterations=4)

    logger.debug('Continue with retraining')
    opt.lr = 1e-6
    opt.epochs = 15
    for iteration in range(10):
        logger.debug('Iteration %d' % iteration)
        corpus.iter = iteration
        corpus.accuracy_corpus()
        corpus.embedding_training()
        corpus.one_gaussian_model()
        corpus.subactivity_sampler()

        corpus.ordering_sampler()
        corpus.rho_sampling()

@timing
def all_actions():
    frames = 0
    return_stat_all = None
    actions = ['coffee', 'cereals', 'tea', 'milk', 'juice', 'sandwich', 'scrambledegg', 'friedegg', 'salat', 'pancake']
    # actions = ['changing_tire', 'coffee', 'jump_car', 'cpr', 'repot']
    lr_init = opt.lr
    for action in actions:
        opt.subaction = action
        if not opt.resume:
            opt.lr = lr_init
        for arg in vars(opt):
            logger.debug('%s: %s' % (arg, getattr(opt, arg)))
        update_opt_str()
        if opt.viterbi:
            return_stat_single = pose_training(iterations=1)
        else:
            return_stat_single = pose_training(iterations=4)
        return_stat_all = join_return_stat(return_stat_all, return_stat_single)
    # logger.debug('Frames in total: %d' % frames)
    parse_return_stat(return_stat_all)


@timing
def grid_search():
    epochs = [30, 60, 90]
    dims = [20, 30, 40]
    lrs = [1e-2, 1e-3, 1e-4]

    # for epoch, dim, lr in grid:
    grid = [[30, 20, 1e-3],
            [30, 30, 1e-3],
            [30, 40, 1e-3]]

    radius = [1.0, 1.5, 2.0]
    epochs = [5, 10, 20]
    dims = [20, 50, 100, 200]


    # for r in radius:
    #     for epoch in epochs:
    #         for dim in dims:
    #         opt.bg_trh = r
    # logger.debug('\n\nSET: radius: %.1e  dim: %d  epochs: %d\n' %
    # (r, dim, epoch))
    # weights = [10.0, 20.0]
    # bg_trh = [1, 1.5, 2]

    # concats = [3, 9]
    # dims = [40, 80]
    # epochs = [30, 60]
    #
    # for concat in concats:
    #     for epoch in epochs:
    #         for dim in dims:

    grid = [[40, 90, 1e-2],
            [20, 90, 1e-4],
            [30, 90, 1e-4],
            [40, 90, 1e-4]]

    epochs = [30, 60, 90]
    dims = [20, 30, 40]
    # lrs = [1e-3, 1e-2, 1e-4]
    lrs = [1e-5]

    # resume_template = 'grid.vit._%s_mlp_!pose_full_vae1_time10.0_epochs%d_embed%d_n2_ordering_gmm1_one_!gt_lr%s_lr_zeros_b0_v1_l0_c1_'
    resume_template = 'fixed.order._%s_mlp_!pose_full_vae0_time10.0_epochs%d_embed%d_n1_!ordering_gmm1_one_!gt_lr%s_lr_zeros_b0_v1_l0_c1_'

    # for dim, epoch, lr in grid:
    for epoch in epochs:
        for lr in lrs:
            for dim in dims:

                opt.embed_dim = dim
                opt.epochs = epoch
                # opt.concat = concat

                opt.lr = lr
                # opt.time_weight = w
                # opt.bg_trh = w
                # opt.resume_str = resume_template % (opt.subaction, epoch, dim, str(lr))

                logger.debug('\n\nSET: dim: %d  epochs: %d, lr: %.1e\n' %
                             (dim, epoch, lr))
                update_opt_str()
                for arg in vars(opt):
                    logger.debug('%s: %s' % (arg, getattr(opt, arg)))
                pose_training(iterations=1)


def seed_test():
    for i in range(50, 60):
        all_actions(seed=i)

def resume_segmentation(iterations=10):
    logger.debug('Resume segmentation')
    corpus = Corpus(Q=opt.gmm,
                    K=define_K(opt.subaction),
                    subaction=opt.subaction,
                    poses=True)

    for iteration in range(iterations):
        logger.debug('Iteration %d' % iteration)
        corpus.iter = iteration
        corpus.resume_segmentation()
        corpus.accuracy_corpus()
    corpus.accuracy_corpus()


if __name__ == '__main__':
    if opt.subaction == 'seed':
        seed_test()
    elif opt.subaction == 'all':
        all_actions()
    elif opt.grid_search:
        grid_search()
    else:
        if opt.tr_type == 'base':
            baseline()
        if opt.tr_type in ['mlp', 'vae', 'nothing', 'rank']:
            pose_training()
        if opt.tr_type == 'joined':
            joined_training()
