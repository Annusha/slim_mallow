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
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.mixture import GaussianMixture
import time
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from scipy.linalg import eigh

from video import Video
from mallow import Mallow
from slice_sampling import slice_sample
from models.lin_embedding import Embedding
from models.dataset_torch import load_data
from models.training_embed import training, load_model
from models import mlp, vae
from utils.arg_pars import opt, logger
from utils.accuracy_class import Accuracy
from utils.mapping import ground_truth, gt_with_0
from utils.utils import join_data, timing, dir_check
from utils.visualization import Visual, plot_segm
from utils.gmm_utils import AuxiliaryGMM, GMM_trh
from pose_utils.pose_collection import PoseCollection
from utils.f1_score import F1Score


class Corpus(object):
    def __init__(self, Q, K=0, subaction='coffee', *, poses=False, with_bg=False):
        """
        Args:
            Q: number of Gaussian components in each mixture
            K: number of possible subactions in current dataset
            subaction: current name of complex activity
        """
        logger.debug('%s  subactions: %d' % (subaction, K))
        self.iter = -1
        self.return_stat = {}

        ##################################
        # for rho sampling testing
        # self._K = K
        # self._inv_count_stat = np.zeros(K)
        # self._mallow = Mallow(K)
        # self.rho_sampling()
        ####################################

        self._acc_old = 0
        self._K = K
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
        if poses:
            self._init_poses()

        new_norm = (self._features - np.mean(self._features, axis=0)) / np.std(self._features, axis=0)
        logger.debug('min: %f  max: %f  avg: %f' %
                     (np.min(new_norm),
                      np.max(new_norm),
                      np.mean(new_norm)))

        self.rho_sampling()

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
                        # if opt.n_d == 4:
                        #     gt_name += '_idt'
                        # use extracted features from pretrained on gt embedding
                        if opt.save_feat:
                            dir_check(os.path.join(opt.data, 'embed'))
                            path = os.path.join(opt.data, 'embed', '%d_%d_%s' %
                                                (opt.embed_dim, opt.n_d, gt_name))
                        else:
                            path = os.path.join(root, filename)
                        start = 0 if self._features is None else self._features.shape[0]
                        video = Video(path, K=self._K,
                                      gt=ground_truth[gt_name],
                                      gt_with_0=gt_with_0[gt_name],
                                      name=gt_name,
                                      start=start,
                                      with_bg=self._with_bg)
                        # if video.features().shape[0] > 10000:
                        #     continue
                        self._features = join_data(self._features, video.features(),
                                                   np.vstack)

                        self._queue_in.put(len(self._videos))
                        video.reset()
                        # accumulate statistic for inverse counts vector for each video
                        self._videos.append(video)
                        gt_stat.update(ground_truth[gt_name])
                        if not opt.full:
                            if len(self._videos) > 30:
                                break

        for video in self._videos:
            video.update_indexes(len(self._features))
        logger.debug('gt statistic: ' + str(gt_stat))
        self._update_fg_mask()

    def _init_poses(self):
        """Define initial segmentation based on extracted poses"""
        video_poses = PoseCollection(K=self._K, videos=self._videos)
        if opt.pose_segm:
            video_poses.segment_collection()
        else:
            video_poses.labels_without_segmentation()
        for video in self._videos:
            video.pose = video_poses[video.name]

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
        if opt.save_feat:
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
    def _video_gaussians(self, idx_exclude=-1, save=False):
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
                                  reg_covar=1e-4)
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
            if opt.save_feat:
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

            ########################################
            # for gaussian analysis in different cases
            # if k == 0:
            #     types = ['one', 'many']
            #     n = '0%d'%idx_exclude
            #     np.savetxt(os.path.join(opt.data, 'gauss_pr',
            #                             'mean_%s_gmm_%d_#n_%s' %
            #                             (types[1], k, n)), gmm.means_)
            #     np.savetxt(os.path.join(opt.data, 'gauss_pr',
            #                             'var_%s_gmm_%d_#n_%s' %
            #                             (types[1], k, n)), gmm.covariances_.squeeze())
            # break
            ########################################

        if save:
            self._video_likelihood_grid(idx_exclude)
            self._videos[idx_exclude].save_likelihood()

        if opt.bg:
            # with bg model I assume that I use only one component
            assert self._Q == 1
            for gm_idx, gmm in self._gaussians.items():
                self._gaussians[gm_idx] = GMM_trh(gmm)

    def _gaussians_queue(self):
        """Queue for multiprocessing gmm"""
        while not self._queue_in.empty():
            try:
                n_video = self._queue_in.get(timeout=3)
                if n_video % 20 == 0:
                    logger.debug('%d / %d' % (n_video, len(self._videos)))
                self._video_gaussians(idx_exclude=n_video, save=True)
            except queue.Empty:
                pass

    @timing
    def one_gaussian_model(self):
        logger.debug('Fit Gaussian Mixture Model to the whole dataset at once')
        self._video_gaussians(idx_exclude=-1)
        for video_idx in range(len(self._videos)):
            self._video_likelihood_grid(video_idx)

        if opt.bg:
            scores = None
            for video in self._videos:
                scores = join_data(scores, video.get_likelihood(), np.vstack)
            # ------
            # bg_trh_score = np.max(scores, axis=0)
            # bg_trh_score = np.sort(bg_trh_score)[int(0.5 * bg_trh_score.shape[0])]
            # logger.debug('bg_trh_score: %s' % str(bg_trh_score))
            # ------

            smth = np.sort(scores, axis=0)
            smth2 = int(0.5 * scores.shape[0])
            bg_trh_score = np.sort(scores, axis=0)[int((opt.bg_trh / 100) * scores.shape[0])]
            # bg_trh_score = np.max(scores, axis=0)

            bg_trh_set = []
            for action_idx in range(self._K):
                # ------
                # new_bg_trh = self._gaussians[action_idx].mean_score - bg_trh_score
                # opt.bg_trh = max(opt.bg_trh, new_bg_trh)
                # ------

                new_bg_trh = self._gaussians[action_idx].mean_score - bg_trh_score[action_idx]
                self._gaussians[action_idx].update_trh(new_bg_trh=new_bg_trh)
                bg_trh_set.append(new_bg_trh)
            # ------
            # logger.debug('new bg_trh: %f' % opt.bg_trh)
            # ------
            logger.debug('new bg_trh: %s' % str(bg_trh_set))
            trh_set = []
            for action_idx in range(self._K):
                # self._gaussians[action_idx].update_trh()
                trh_set.append(self._gaussians[action_idx].trh)
            for video in self._videos:
                video.valid_likelihood_update(trh_set)

    @timing
    def many_gaussian_models(self):
        logger.debug('Learn Gaussian Mixture Model for each video separately '
                     'and compute likelihood grid')
        for idx in range(len(self._videos)):
            self._video_gaussians(idx)
            self._video_likelihood_grid(idx)
            if idx % 10 == 0:
                logger.debug('%d / %d' % (idx, len(self._videos)))

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
        """Sampling ordering for each video"""
        logger.debug('.')
        self._inv_count_stat = np.zeros(self._K - 1)
        bg_total = 0
        pr_orders = []
        for video_idx, video in enumerate(self._videos):
            video.iter = self.iter

            cur_order = video.ordering_sampler(mallow_model=self._mallow)
            # cur_order = video.viterbi_top_perm()

            # if cur_order not in pr_orders:
            #     logger.debug(str(cur_order))
            #     pr_orders.append(cur_order)
            self._inv_count_stat += video.inv_count_v
            # refill queue
            self._queue_in.put(video_idx)
            # logger.debug('background: %d' % int(video.fg_mask.size - np.sum(video.fg_mask)))
            bg_total += int(video.fg_mask.size - np.sum(video.fg_mask))
        logger.debug('total background: %d' % bg_total)
        logger.debug('inv_count_vec: %s' % str(self._inv_count_stat))

    @timing
    def viterbi_ordering(self):
        logger.debug('.')
        self._inv_count_stat = np.zeros(self._K - 1)
        bg_total = 0
        for video_idx, video in enumerate(self._videos):
            if video_idx % 20 == 0:
                logger.debug('%d / %d' % (video_idx, len(self._videos)))
            video.iter = self.iter
            video.viterbi_ordering(mallow_model=self._mallow)
            self._inv_count_stat += video.inv_count_v
            # logger.debug('background: %d' % int(video.fg_mask.size - np.sum(video.fg_mask)))
            bg_total += int(video.fg_mask.size - np.sum(video.fg_mask))
        logger.debug('total background: %d' % bg_total)
        logger.debug(str(self._inv_count_stat))


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

    def _action_presence_counter(self):
        """Count how many times each action occurs within video collection. """
        presence = np.zeros(self._K)
        for video in self._videos:
            presence += np.asarray(video.a) > 1
        return presence

    def viterbi_alex_decoding(self):
        logger.debug('.')
        self._count_subact()
        len_model = np.asarray(self._subact_counter) / self._action_presence_counter()
        # len_model = len_model.reshape((-1, 1))
        for video_idx, video in enumerate(self._videos):
            if video_idx % 20 == 0:
                logger.debug('%d / %d' % (video_idx, len(self._videos)))
                self._count_subact()
                logger.debug(str(self._subact_counter))
            video.viterbi_alex(len_model)
        self._count_subact()
        logger.debug(str(self._subact_counter))

    def rho_sampling(self):
        """Sampling of parameters for Mallow Model using slice sampling"""
        self._mallow.rho = []
        for k in range(self._K - 1):
            self._mallow.set_sample_params(sum_inv_vals=self._inv_count_stat[k],
                                           k=k, N=len(self._videos))
            sample = slice_sample(init=1, burn_in=100,
                                  logpdf=self._mallow.logpdf)
            self._mallow.rho.append(sample)
        logger.debug(str(['%.4f' % i for i in self._mallow.rho]))

    def embedding_training(self):
        """Training of embedding before fitting gmm on it.

        1) Features from embedding could be already saved on disk -> do nothing.
        2) Use already pretrained embedding -> load model.
        3) Train it with current labeling.
        """
        logger.debug('.')
        if opt.save_feat:
            return
        if opt.resume:
            self._embedding = Embedding(embed_dim=opt.embed_dim,
                                        feature_dim=opt.feature_dim,
                                        n_subact=self._K)
            self._embedding.load_state_dict(load_model(epoch=opt.resume * (opt.epochs - 1),
                                                       name='rank'))
        else:
            if opt.gt_training and opt.full:
                # training with gt labels
                dataloader, _ = load_data(opt.data, opt.end,
                                          subaction=self._subaction)
            else:
                # training with labels from th algorithm
                dataloader, _ = load_data(opt.data, opt.end,
                                          subaction=self._subaction,
                                          videos=self._videos,
                                          features=self._features)
            self._embedding = training(dataloader, opt.epochs,
                                       n_subact=self._K,
                                       name='rank',
                                       save=opt.save_model)
            self._embedding = self._embedding.cpu()

        self._embedded_feat = torch.Tensor(self._features)
        # calculate feature representation at once to make usage of them easier
        self._embedded_feat = self._embedding.embedded(self._embedded_feat).detach().numpy()

    def regression_training(self):
        logger.debug('.')

        if opt.vae_dim == 0:
            dataloader, _ = load_data(opt.data, opt.end,
                                      subaction=self._subaction,
                                      videos=self._videos,
                                      features=self._features,
                                      regression=True)
        else:

            ### concat
            dataloader, _ = load_data(opt.data, opt.end,
                                      subaction=self._subaction,
                                      videos=self._videos,
                                      features=self._features,
                                      vae=True)

        model, loss, optimizer = mlp.create_model()
        if opt.resume:
            model.load_state_dict(load_model(epoch=opt.resume * (opt.epochs - 1),
                                             name='mlp'))
            self._embedding = model
        else:
            self._embedding = training(dataloader, opt.epochs,
                                       save=opt.save_model,
                                       model=model,
                                       loss=loss,
                                       optimizer=optimizer,
                                       name='mlp')

        features = self._features
        # concat consecutive features
        if opt.concat > 1:
            features = None
            for video in self._videos:
                video_features = self._features[video.global_range]
                video_feature_concat = video_features[:]
                last_frame = video_features[-1]
                for i in range(opt.concat - 1):
                    video_feature_concat = np.roll(video_feature_concat, -1, axis=0)
                    video_feature_concat[-1] = last_frame
                    video_features = join_data(video_features,
                                               video_feature_concat,
                                               np.hstack)
                features = join_data(features,
                                     video_features,
                                     np.vstack)

        ### concat with respective relative time label
        if opt.vae_dim == 1:
            long_rel_time = []
            for video in self._videos:
                long_rel_time += list(video.pose.frame_labels)

            # logger.debug('Zero time step')
            # long_rel_time = np.asarray(long_rel_time).reshape((-1, 1)) * 0
            long_rel_time = np.asarray(long_rel_time).reshape((-1, 1))

            features = join_data(features, long_rel_time, np.hstack)

        ### concat label
        if opt.label:
            long_uniform_labels = []
            for video in self._videos:
                long_uniform_labels += list(video._z)
            long_uniform_labels = np.asarray(long_uniform_labels).reshape((-1, 1))
            features = join_data(features, long_uniform_labels, np.hstack)


        # will crash in case of vae_dim != 1
        self._embedded_feat = torch.Tensor(features)

        self._embedding = self._embedding.cpu()
        # self._embedded_feat = torch.Tensor(self._features)
        relative_time = self._embedding(self._embedded_feat).detach().numpy()
        self._embedded_feat = self._embedding.embedded(self._embedded_feat).detach().numpy()
        long_gt_time = []
        for video in self._videos:
            if -1 in video.pose.frame_labels:
                print('wtf')
            long_gt_time += list(video.pose.frame_labels)

        mse = np.sum((long_gt_time - relative_time)**2)
        long_gt_time = np.array(long_gt_time).reshape(-1, 1)
        relative_time = relative_time.reshape(-1, 1)
        long_gt_time = np.concatenate((long_gt_time, relative_time), axis=1)
        mse = mse / len(relative_time)
        # break point: opt.full == False
        logger.debug('MLP training: MSE: %f' % mse)

    def without_temp_emed(self):
        logger.debug('No temporal embedding')
        self._embedded_feat = self._features.copy()

        # ### concat with respective relative time label
        # if opt.vae_dim == 1:
        #     long_rel_time = []
        #     for video in self._videos:
        #         long_rel_time += list(video.pose.frame_labels)
        #     long_rel_time = np.asarray(long_rel_time).reshape((-1, 1))
        #     self._embedded_feat = join_data(self._embedded_feat,
        #                                     long_rel_time,
        #                                     np.hstack)


    def clustering(self):
        logger.debug('.')
        np.random.seed(opt.seed)
        # kmean = KMeans(n_clusters=self._K, random_state=opt.seed)

        kmean = MiniBatchKMeans(n_clusters=self._K,
                                 random_state=opt.seed,
                                 batch_size=50)

        ### concat with respective relative time label
        # long_rel_time = []
        # for video in self._videos:
        #     long_rel_time += list(video.pose.frame_labels)
        # long_rel_time = np.asarray(long_rel_time).reshape((-1, 1))
        # self._embedded_feat = join_data(self._embedded_feat, long_rel_time, np.hstack)
        # assert self._embedded_feat.shape[1] == (opt.embed_dim + 1)

        kmean.fit(self._embedded_feat[self._total_fg_mask])

        accuracy = Accuracy()
        long_gt = []
        long_rt = []
        for video in self._videos:
            long_gt += list(video._gt_with_0)
            long_rt += list(video.pose.frame_labels)
        long_rt = np.array(long_rt)

        kmeans_labels = np.asarray(kmean.labels_).copy()
        uniq = np.unique(kmeans_labels)
        time2label = {}
        for label in np.unique(kmeans_labels):
            cluster_mask = kmeans_labels == label
            r_time = np.mean(long_rt[self._total_fg_mask][cluster_mask])
            time2label[r_time] = label

        np.random.seed(opt.seed)
        shuffle_labels = np.arange(len(time2label))
        np.random.shuffle(shuffle_labels)
        for time_idx, sorted_time in enumerate(sorted(time2label)):
            label = time2label[sorted_time]
            if opt.shuffle_order:
                logger.debug('shuffled labels')
                kmeans_labels[kmean.labels_ == label] = shuffle_labels[time_idx]
            else:
                logger.debug('time ordered labels')
                kmeans_labels[kmean.labels_ == label] = time_idx
                shuffle_labels = np.arange(len(time2label))

        labels_with_bg = np.ones(len(self._total_fg_mask)) * -1

        if opt.shuffle_order and opt.kmeans_shuffle:
            # use pure output of kmeans algorithm
            logger.debug('kmeans random labels')
            labels_with_bg[self._total_fg_mask] = kmean.labels_
            shuffle_labels = [value for (key, value) in sorted(time2label.items(), key=lambda x: x[0])]
        else:
            # use predefined by time order or numpy shuffling labels for kmeans clustering
            logger.debug('assignment: %s' % ['ordered', 'shuffled'][opt.shuffle_order])
            labels_with_bg[self._total_fg_mask] = kmeans_labels

        logger.debug('Shuffled labels: %s' % str(shuffle_labels))
        accuracy.predicted_labels = labels_with_bg
        accuracy.gt_labels = long_gt
        old_mof, total_fr = accuracy.mof()
        self._gt2label = accuracy._gt2cluster
        for key, val in self._gt2label.items():
            try:
                self._label2gt[val[0]] = key
            except IndexError:
                pass


        logger.debug('MoF val: ' + str(accuracy.mof_val()))
        logger.debug('old MoF val: ' + str(float(old_mof) / total_fr))

        ########################################################################
        # VISUALISATION
        # vis = Visual(mode='pca', full=True, save=True)
        # vis.data = self._embedded_feat
        # vis.labels = kmean.labels_
        # vis.fit_data()
        # vis.plot(show=False, prefix='kmean')
        # vis.reset()
        ########################################################################

        logger.debug('Update video z for videos before GMM fitting')
        labels_with_bg[labels_with_bg == self._K] = -1
        for video in self._videos:
            video.update_z(labels_with_bg[video.global_range])
            # video._z = kmean.labels_[video.global_range]

        for video in self._videos:
            video.segmentation['cl'] = (video._z, self._label2gt)

    def pca(self):
        logger.debug('.')
        _pca = PCA(n_components=30, random_state=opt.seed)
        _pca.fit(self._embedded_feat)
        self._embedded_feat = _pca.transform(self._embedded_feat)

    def vae_training(self):
        logger.debug('.')

        dataloader, _ = load_data(opt.data, opt.end,
                                  subaction=self._subaction,
                                  videos=self._videos,
                                  features=self._features,
                                  vae=True)
        model, loss, optimizer = vae.create_model()
        if opt.resume:
            model.load_state_dict(load_model(epoch=opt.resume * (opt.epochs - 1),
                                             name='vae'))
            self._embedding = model
        else:
            self._embedding = training(dataloader, opt.epochs,
                                       save=opt.save_model,
                                       model=model,
                                       loss=loss,
                                       optimizer=optimizer,
                                       name='vae',
                                       vae=True)
        if opt.vae_dim == 1:
            long_rel_time = []
            for video in self._videos:
                long_rel_time += list(video.pose.frame_labels)
            long_rel_time = np.asarray(long_rel_time).reshape((-1, 1))
        else:  # opt.vae_dim > 1
            long_rel_time = None
            for video in self._videos:
                long_rel_time = join_data(long_rel_time,
                                          video.pose.relative_segments(),
                                          np.vstack)

        self._embedded_feat = torch.Tensor(join_data(None,
                                                     (self._features, long_rel_time),
                                                     np.hstack))
        self._embedding = self._embedding.cpu()
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
        long_rel_time = []
        self.return_stat = {}

        for video in self._videos:
            long_gt += list(video._gt_with_0)
            long_gt_onhe0 += list(video._gt)
            long_pr += list(video._z)
            try:
                long_rel_time += list(video.pose.frame_labels)
            except AttributeError:
                pass
                # logger.debug('no poses')
        accuracy.gt_labels = long_gt
        accuracy.predicted_labels = long_pr
        if opt.bg:
            # enforce bg class to be bg class
            accuracy.exclude[-1] = [-1]
        if not opt.zeros:
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
        logger.debug('%sold MoF val: ' % prefix + str(float(old_mof) / total_fr))

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

        if opt.vis:
            # VISUALISATION
            mode = 'pca'
            vis = Visual(mode=mode, full=True, save=True)
            gt_plot_iter = [[0, 1, 2], [0]][self.iter != 0]
            long_pr = [self._label2gt[i] for i in long_pr]
            for gt_plot in gt_plot_iter:
                vis.data = self._embedded_feat
                vis.labels = [long_pr, long_gt_onhe0, long_rel_time][gt_plot]
                if mode == 'pca':
                    vis.fit_data()
                else:  # mode == 'tsne'
                    vis.fit_data(reduce=int(0.3 * self._features.shape[0]))
                vis.plot(iter=self.iter, show=False, gt_plot=gt_plot)
                vis.reset()

            ####################################################################
            # segmentation visualisation
            if prefix == 'final':
                colors = {}
                cmap = plt.get_cmap('tab20')
                for label_idx, label in enumerate(np.unique(long_gt)):
                    if label == -1:
                        colors[label] = (0, 0, 0)
                    else:
                        # colors[label] = (np.random.rand(), np.random.rand(), np.random.rand())
                        colors[label] = cmap(label_idx / len(np.unique(long_gt)))

                dir_check(os.path.join(opt.dataset_root, 'plots'))
                dir_check(os.path.join(opt.dataset_root, 'plots', opt.subaction))
                fold_path = os.path.join(opt.dataset_root, 'plots', opt.subaction, 'segmentation')
                dir_check(fold_path)
                for video in self._videos:
                    path = os.path.join(fold_path, video.name + '.png')
                    name = video.name.split('_')
                    name = '_'.join(name[-2:])
                    plot_segm(path, video.segmentation, colors, name=name)
            ####################################################################

        return accuracy.frames()



        # if np.abs(acc_cur - self._acc_old) < -1:
        #     return -1
        # else:
        #     self._acc_old = acc_cur
        #     return 1

        # self._collect_stat_order()

    def resume_segmentation(self):
        for video in self._videos:
            video.iter = self.iter
            video.resume()
        self._count_subact()

    ###########################################################################
    # additional methods for one time usage

    def _collect_stat_order(self):
        """Collect statistic about ordering in the entire data collection."""
        logger.debug('Collect statistic about ordering')
        without_strict_order = np.zeros(len(self._videos), dtype=bool)
        inv_count_stat = np.zeros(self._K)
        for idx, video in enumerate(self._videos):
            position = self._gt2label[video._gt[0]][0]
            video_order = [position]
            for l in video._gt:
                if self._gt2label[l][0] < position:
                    without_strict_order[idx] = True
                position = self._gt2label[l][0]
                if video_order[-1] != position:
                    if position not in video_order:
                        video_order.append(position)
            if len(video_order) < self._K:
                update_video_order = np.arange(self._K)
                not_in_order = []
                for k in range(self._K):
                    if k not in video_order:
                        not_in_order.append(k)
                shift = 0
                for i, v in enumerate(video_order):
                    j = i + shift
                    while j in not_in_order:
                        shift += 1
                        j += 1
                    update_video_order[i + shift] = v
                video_order = update_video_order
            vec = self._mallow.inversion_counts(video_order)
            inv_count_stat += vec

        logger.debug(str(np.sum(without_strict_order)) + ' / ' + str(len(self._videos)))
        logger.debug('gt inverse count vector stat for entire collection: ' +
                     str(inv_count_stat))

    def check_gaussians(self):
        """Check how well gmms fit on embedding"""
        logger.debug('.')
        anchors = np.loadtxt(os.path.join(opt.data, 'embed',
                                          'anchors_%s_%d_%d' % (opt.subaction,
                                                                opt.embed_dim, opt.n_d)))
        denominator = 0.
        anchor_acc = 0.
        gauss_acc = 0.
        merged_acc = 0.

        subactions = np.sort(list(self._label2gt.values()))
        if len(subactions) > self._K:
            subactions = list(subactions)
            subactions.remove(0)

        embed_gt2idx = {}
        embed_idx2gt = {}
        for idx, old in enumerate(subactions):
            embed_gt2idx[int(old)] = idx
            embed_idx2gt[idx] = int(old)

        for video in self._videos:
            features = video.features()
            dist = -2 * np.dot(features, anchors.T) + np.sum(anchors ** 2, axis=1) \
                   + np.sum(features ** 2, axis=1)[:, np.newaxis]

            embedded_labels = np.argmin(dist, axis=1)
            embedded_labels = np.array(
                list(map(lambda x: embed_idx2gt[x], embedded_labels)))

            table = None
            for k in range(self._K):
                # score = self._gaussians[k].score_samples(features)
                score = video._likelihood_grid[:, k]
                table = join_data(table, score, np.vstack)
            gauss_labels = np.argmax(table, axis=0)
            gauss_labels = np.array(list(map(lambda x: self._label2gt[x], gauss_labels)))

            video_gt = video._gt

            for gt_label in np.unique(video_gt):
                mask = video_gt == gt_label
                anchor_acc += np.sum(embedded_labels[mask] == gt_label, dtype=float)
                gauss_acc += np.sum(gauss_labels[mask] == gt_label, dtype=float)
                merged_acc += np.sum(
                    embedded_labels[gauss_labels == gt_label] == gt_label, dtype=float)

            denominator += video.n_frames

        anchor_acc = anchor_acc / denominator
        gauss_acc = gauss_acc / denominator
        merged_acc = merged_acc / denominator

        logger.debug(
            'anchor: %f\ngauss: %f\nmerged: %f' % (anchor_acc, gauss_acc, merged_acc))


