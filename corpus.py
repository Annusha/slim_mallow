#!/usr/bin/env python

"""Module with Corpus class. There are methods for each step of the alg for the
whole video collection of one complex activity. See pipeline."""

__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import numpy as np
import os
import torch
import re
import queue
import multiprocessing as mp
from collections import Counter
from sklearn.mixture import GaussianMixture
import time

from video import Video
from mallow import Mallow
from slice_sampling import slice_sampling
from models.rank import Embedding
from models.dataset_torch import load_data
from YTI_utils.dataset_torch import load_data as yti_load_data
from models.training_embed import training, load_model
from utils.arg_pars import opt
from utils.logging_setup import logger
from utils.accuracy_class import Accuracy
from utils.mapping import GroundTruth
from utils.utils import join_data, timing, dir_check
from utils.gmm_utils import AuxiliaryGMM, GMM_trh
from utils.f1_score import F1Score


class Corpus(object):
    def __init__(self, Q, subaction='coffee'):
        """
        Args:
            Q: number of Gaussian components in each mixture
            K: number of possible subactions in current dataset
            subaction: current name of complex activity
        """
        self.gt_map = GroundTruth()
        self.gt_map.load_mapping()
        self._K = self.gt_map.define_K(subaction=subaction)
        logger.debug('%s  subactions: %d' % (subaction, self._K))
        self.iter = -1
        self.return_stat = {}

        self._acc_old = 0
        self._videos = []
        self._subaction = subaction
        # init with ones for consistency with first measurement of MoF
        self._subact_counter = np.ones(self._K)
        # number of gaussian in each mixture
        self._Q = 1 if opt.gt_training else Q
        self._gaussians = {}
        self._mallow = Mallow(self._K)
        self._inv_count_stat = np.zeros(self._K)
        self._embedding = None
        self._gt2label = None
        self._label2gt = {}

        self._with_bg = opt.bg
        self._total_fg_mask = None

        # multiprocessing for sampling activities for each video
        self._queue_in = mp.Queue()
        self._queue_out = mp.Queue()
        self._mp_batch = 20
        self._features = None
        self._embedded_feat = None
        self._init_videos()
        logger.debug('min: %f  max: %f  avg: %f' %
                     (np.min(self._features),
                      np.max(self._features),
                      np.mean(self._features)))

        self.rho_sampling()

        # to save segmentation of the videos
        dir_check(os.path.join(opt.data, 'segmentation'))

    def _init_videos(self):
        logger.debug('.')
        gt_stat = Counter()
        for root, dirs, files in os.walk(os.path.join(opt.data, 'ascii')):
            if files:
                for filename in files:
                    # pick only videos with certain complex action
                    # (ex: just concerning coffee)
                    if self._subaction in filename:
                        match = re.match(r'(\w*)\.\w*', filename)
                        gt_name = match.group(1)
                        # use extracted features from pretrained on gt embedding
                        if opt.save_embed_feat:
                            dir_check(os.path.join(opt.data, 'embed'))
                            path = os.path.join(opt.data, 'embed', '%d_%d_%s' %
                                                (opt.embed_dim, opt.data_type, gt_name))
                        else:
                            path = os.path.join(root, filename)
                        start = 0 if self._features is None else self._features.shape[0]
                        video = Video(path, K=self._K,
                                      gt=self.gt_map.gt[gt_name],
                                      gt_with_0=self.gt_map.gt_with_0[gt_name],
                                      name=gt_name,
                                      start=start,
                                      with_bg=self._with_bg)
                        self._features = join_data(self._features, video.features(),
                                                   np.vstack)

                        self._queue_in.put(len(self._videos))
                        video.reset()  # to not store second time loaded features
                        self._videos.append(video)
                        # accumulate statistic for inverse counts vector for each video
                        gt_stat.update(self.gt_map.gt[gt_name])
                        if not opt.full:
                            if len(self._videos) > 10:
                                break

        # update global range within the current collection for each video
        for video in self._videos:
            video.update_indexes(len(self._features))
        logger.debug('gt statistic: ' + str(gt_stat))
        self._update_fg_mask()

    def _update_fg_mask(self):
        logger.debug('.')
        if self._with_bg:
            self._total_fg_mask = np.zeros(len(self._features), dtype=bool)
            for video in self._videos:
                self._total_fg_mask[np.nonzero(video.global_range)[0][video.fg_mask]] = True
        else:
            self._total_fg_mask = np.ones(len(self._features), dtype=bool)

    def _count_subact(self):
        self._subact_counter = np.zeros(self._K)
        for video in self._videos:
            self._subact_counter += video.a

    def _video_likelihood_grid(self, video_idx):
        video = self._videos[video_idx]
        if opt.save_embed_feat:
            features = self._features[video.global_range]
        else:
            features = self._embedded_feat[video.global_range]
        for subact in range(self._K):
            scores = self._gaussians[subact].score_samples(features)
            if opt.bg:
                video.likelihood_update(subact, scores,
                                        trh=self._gaussians[subact].trh)
            else:
                video.likelihood_update(subact, scores)
        if opt.save_likelihood:
            video.save_likelihood()

    @timing
    def _gaussians_fit(self, idx_exclude=-1, save=False):
        """ Fit GMM to video features.

        Define subset of video collection and fit on it gaussian mixture model.
        If idx_exclude = -1, then whole video collection is used for comprising
        the subset, otherwise video collection excluded video with this index.
        Args:
            idx_exclude: video to exclude (-1 or int in range(0, #_of_videos))
            save: in case of mp lib all computed likelihoods are saved on disk
                before the next step
        """
        # logger.debug('Excluded: %d' % idx_exclude)
        for k in range(self._K):
            gmm = GaussianMixture(n_components=self._Q,
                                  covariance_type='full',
                                  max_iter=150,
                                  random_state=opt.seed,
                                  reg_covar=1e-6)
            total_indexes = np.zeros(len(self._features), dtype=np.bool)
            for idx, video in enumerate(self._videos):
                if idx == idx_exclude:
                    continue
                if opt.gt_training:
                    indexes = np.where(np.asarray(video._gt) == self._label2gt[k])[0]
                else:
                    indexes = np.where(np.asarray(video._z) == k)[0]
                if len(indexes) == 0:
                    continue
                temp = np.zeros(video.n_frames, dtype=np.bool)
                temp[indexes] = True
                total_indexes[video.global_range] = temp

            total_indexes = np.array(total_indexes, dtype=np.bool)
            if opt.save_embed_feat:
                feature = self._features[total_indexes, :]
            else:
                feature = self._embedded_feat[total_indexes, :]
            time1 = time.time()
            try:
                gmm.fit(feature)
            except ValueError:
                gmm = AuxiliaryGMM()
            time2 = time.time()
            if idx_exclude % 20 == 0:
                logger.debug('fit gmm %0.6f %d ' % ((time2 - time1), len(feature))
                             + str(gmm.converged_))

            self._gaussians[k] = gmm

        if save:
            self._video_likelihood_grid(idx_exclude)
            self._videos[idx_exclude].save_likelihood()

        if opt.bg:
            # with bg model I assume that I use only one component
            assert self._Q == 1
            for gm_idx, gmm in self._gaussians.items():
                self._gaussians[gm_idx] = GMM_trh(gmm)

    @timing
    def one_gaussian_model(self):
        logger.debug('Fit Gaussian Mixture Model to the whole dataset at once')
        self._gaussians_fit(idx_exclude=-1)
        for video_idx in range(len(self._videos)):
            self._video_likelihood_grid(video_idx)

        if opt.bg:
            scores = None
            for video in self._videos:
                scores = join_data(scores, video.get_likelihood(), np.vstack)
            # ------ max ------
            # bg_trh_score = np.max(scores, axis=0)
            # logger.debug('bg_trh_score: %s' % str(bg_trh_score))
            # ------ max ------

            bg_trh_score = np.sort(scores, axis=0)[int((opt.bg_trh / 100) * scores.shape[0])]

            bg_trh_set = []
            for action_idx in range(self._K):
                new_bg_trh = self._gaussians[action_idx].mean_score - bg_trh_score[action_idx]
                self._gaussians[action_idx].update_trh(new_bg_trh=new_bg_trh)
                bg_trh_set.append(new_bg_trh)

            logger.debug('new bg_trh: %s' % str(bg_trh_set))
            trh_set = []
            for action_idx in range(self._K):
                trh_set.append(self._gaussians[action_idx].trh)
            for video in self._videos:
                video.valid_likelihood_update(trh_set)

    @timing
    def many_gaussian_models(self):
        logger.debug('Learn Gaussian Mixture Model for each video separately '
                     'and compute likelihood grid')
        for idx in range(len(self._videos)):
            self._gaussians_fit(idx)
            self._video_likelihood_grid(idx)
            if idx % 10 == 0:
                logger.debug('%d / %d' % (idx, len(self._videos)))

    def _gaussians_queue(self):
        """Queue for multiprocessing gmm"""
        while not self._queue_in.empty():
            try:
                n_video = self._queue_in.get(timeout=3)
                if n_video % 20 == 0:
                    logger.debug('%d / %d' % (n_video, len(self._videos)))
                self._gaussians_fit(idx_exclude=n_video, save=True)
            except queue.Empty:
                pass

    @timing
    def gaussians_mp(self, n_threads=2):
        """Pseudo multiprocessing for fitting many gmm concurrently"""
        logger.debug('.')
        procs = []
        for i in range(n_threads):
            p = mp.Process(target=self._gaussians_queue)
            procs.append(p)
            p.start()
        for p in procs:
            p.join()
        for video in self._videos:
            video.load_likelihood()

    @timing
    def subactivity_sampler(self):
        """Sampling of subactivities for each video"""
        logger.debug('.')
        self._count_subact()
        for idx, video in enumerate(self._videos):
            if idx % 20 == 0:
                logger.debug('%d / %d' % (idx, len(self._videos)))
                logger.debug(str(self._subact_counter))
            temp_sub_counter = self._subact_counter - video.a
            a, _ = video.subactivity_sampler(self._subact_counter)
            self._subact_counter = temp_sub_counter + a
        logger.debug(str(self._subact_counter))

    def ordering_sampler(self):
        """Sampling ordering for each video via Mallow model"""
        logger.debug('.')
        self._inv_count_stat = np.zeros(self._K - 1)
        bg_total = 0
        for video_idx, video in enumerate(self._videos):
            video.iter = self.iter

            video.ordering_sampler(mallow_model=self._mallow)
            self._inv_count_stat += video.inv_count_v
            # refill queue
            self._queue_in.put(video_idx)
            bg_total += int(video.fg_mask.size - np.sum(video.fg_mask))
        logger.debug('total background: %d' % bg_total)
        logger.debug('inv_count_vec: %s' % str(self._inv_count_stat))

    @timing
    def viterbi_decoding(self):
        logger.debug('.')
        self._count_subact()
        pr_orders = []
        for video_idx, video in enumerate(self._videos):
            if video_idx % 20 == 0:
                logger.debug('%d / %d' % (video_idx, len(self._videos)))
                self._count_subact()
                logger.debug(str(self._subact_counter))
            video.viterbi()
            cur_order = list(video._pi)
            if cur_order not in pr_orders:
                logger.debug(str(cur_order))
                pr_orders.append(cur_order)
        self._count_subact()
        logger.debug(str(self._subact_counter))

    def rho_sampling(self):
        """Sampling of parameters for Mallow Model using slice sampling"""
        logger.debug('rho sampling')
        # self._mallow.rho = []
        mallow_rho = []
        inv_pdf = lambda x: -1. / self._mallow.logpdf(x)
        for k in range(self._K - 1):
            # logger.debug('rho sampling k: %d' % k)
            self._mallow.set_sample_params(sum_inv_vals=self._inv_count_stat[k],
                                           k=k, N=len(self._videos))
            sample = slice_sampling(burnin=10, x_init=self._mallow.rho[k],
                                    logpdf=inv_pdf)
            mallow_rho.append(sample)
        self._mallow.rho = mallow_rho
        logger.debug(str(['%.4f' % i for i in self._mallow.rho]))

    def embedding_training(self):
        """Training of embedding before fitting gmm on it.

        1) Features from embedding could be already saved on disk -> do nothing.
        2) Use already pretrained embedding -> load model.
        3) Train it with current labeling.
        """
        logger.debug('.')
        if opt.save_embed_feat:
            return
        K_train = self._K
        if opt.bg:
            K_train = self._K + 1
        if opt.resume:
            self._embedding = Embedding(embed_dim=opt.embed_dim,
                                        feature_dim=opt.feature_dim,
                                        n_subact=K_train)
            self._embedding.load_state_dict(load_model(epoch=opt.resume * (opt.epochs - 1),
                                                       name='rank'))
        else:
            if opt.dataset == 'yti':
                dataloader, _ = yti_load_data(opt.data, opt.ext,
                                              subaction=self._subaction,
                                              videos=self._videos,
                                              features=self._features)
            if opt.dataset == 'bf':
                dataloader, _ = load_data(opt.data, opt.ext,
                                          subaction=self._subaction,
                                          videos=self._videos,
                                          features=self._features)

            self._embedding = training(dataloader, opt.epochs,
                                       n_subact=K_train,
                                       name='rank',
                                       save=opt.save_model)
            self._embedding = self._embedding.cpu()

        self._embedded_feat = torch.Tensor(self._features)
        # calculate feature representation at once to make usage of them easier
        self._embedded_feat = self._embedding.embedded(self._embedded_feat).detach().numpy()

    @timing
    def accuracy_corpus(self, prefix=''):
        """Calculate metrics as well with previous correspondences between
        gt labels and output labels"""
        accuracy = Accuracy()
        f1_score = F1Score(K=self._K, n_videos=len(self._videos))
        long_gt = []
        long_pr = []
        long_gt_onhe0 = []
        self.return_stat = {}

        for video in self._videos:
            long_gt += list(video._gt_with_0)
            long_gt_onhe0 += list(video._gt)
            long_pr += list(video._z)

        accuracy.gt_labels = long_gt
        accuracy.predicted_labels = long_pr
        if opt.bg:
            # enforce bg class to be bg class
            accuracy.exclude[-1] = [-1]
        if not opt.zeros and 'Breakfast' in opt.dataset_root:
            # enforce to SIL class assign nothing
            accuracy.exclude[0] = [-1]

        old_mof, total_fr = accuracy.mof(old_gt2label=self._gt2label)
        self._gt2label = accuracy._gt2cluster
        self._label2gt = {}
        for key, val in self._gt2label.items():
            try:
                self._label2gt[val[0]] = key
            except IndexError:
                pass
        acc_cur = accuracy.mof_val()
        logger.debug('%sAction: %s' % (prefix, self._subaction))
        logger.debug('%sMoF val: ' % prefix + str(acc_cur))
        logger.debug('%sprevious dic -> MoF val: ' % prefix + str(float(old_mof) / total_fr))

        accuracy.mof_classes()
        accuracy.iou_classes()

        self.return_stat = accuracy.stat()

        f1_score.set_gt(long_gt)
        f1_score.set_pr(long_pr)
        f1_score.set_gt2pr(self._gt2label)
        if opt.bg:
            f1_score.set_exclude(-1)
        f1_score.f1()

        for key, val in f1_score.stat().items():
            self.return_stat[key] = val

        for video in self._videos:
            video.segmentation[video.iter] = (video._z, self._label2gt)

        return accuracy.frames()

    def resume_segmentation(self):
        for video in self._videos:
            video.iter = self.iter
            video.resume()
        self._count_subact()
