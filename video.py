#!/usr/bin/env python

"""Module with class for single video"""

__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

from collections import Counter
import numpy as np
import math as m
import os
from os.path import join
import time

from utils.arg_pars import opt
from utils.logging_setup import logger
from utils.utils import timing, dir_check
from viterbi_utils.viterbi import Viterbi
from viterbi_utils.grammar import Grammar


class Video(object):
    """Single video with respective for the algorithm parameters"""
    def __init__(self, path, K, *, reset=False, gt=[],
                 gt_with_0=None, name='', start=0, with_bg=False):
        """
        Args:
            path (str): path to video representation
            K (int): number of subactivities in current video collection
            reset (bool): necessity of holding features in each instance
            gt (arr): ground truth labels
            gt_with_0 (arr): ground truth labels with SIL (0) label
            name (str): short name without any extension
            start (int): start index in mask for the whole video collection
        """
        self.iter = 0
        self.path = path
        self._K = K
        self.name = name

        self._likelihood_grid = None
        self._valid_likelihood = None
        self._theta_0 = 0.1
        self._subact_i_mask = np.eye(self._K)
        self.n_frames = 0
        self._features = None
        self.global_start = start
        self.global_range = None

        self._gt = gt
        self._gt_unique = np.unique(self._gt)
        self._gt_with_0 = gt_with_0

        self.features()
        self._check_gt()

        # counting of subactivities np.array
        self.a = np.zeros(self._K)
        # ordering, init with canonical ordering
        self._pi = list(range(self._K))
        self.inv_count_v = np.zeros(self._K - 1)
        # subactivity per frame
        self._z = []
        self._z_idx = []
        self._init_z_framewise()

        # background
        self._with_bg = with_bg
        self.fg_mask = np.ones(self.n_frames, dtype=bool)
        if self._with_bg:
            self._init_fg_mask()

        self._subact_count_update()

        self.segmentation = {'gt': (self._gt, None)}

    def features(self):
        """Load features given path if haven't do it before"""
        if self._features is None:
            self._features = np.loadtxt(self.path)
            if opt.data_type == 2 and opt.dataset == 'bf':
                self._features = self._features[1:, 1:]
            zeros = 0
            # for i in range(10):
            #     if np.sum(self._features[:1]) == 0:
            #         self._features = self._features[1:]
            #         zeros += 1
            #     else:
            #         break
            self.n_frames = self._features.shape[0]
            self._likelihood_grid = np.zeros((self.n_frames, self._K))
            self._valid_likelihood = np.zeros((self.n_frames, self._K), dtype=bool)
            self._gt = self._gt[zeros:]
            self._gt_with_0 = self._gt_with_0[zeros:]
        return self._features

    def _check_gt(self):
        try:
            assert len(self._gt) == self.n_frames
        except AssertionError:
            print(self.path, '# of gt and # of frames does not match %d / %d' %
                  (len(self._gt), self.n_frames))
            if len(self._gt) - self.n_frames > 50:
                raise AssertionError
            else:
                min_n = min(len(self._gt), self.n_frames)
                self._gt = self._gt[:min_n]
                self._gt_with_0 = self._gt_with_0[:min_n]
                self.n_frames = min_n
                self._features = self._features[:min_n]
                self._likelihood_grid = np.zeros((self.n_frames, self._K))
                self._valid_likelihood = np.zeros((self.n_frames, self._K), dtype=bool)

    def _init_z_framewise(self):
        """Init subactivities uniformly among video frames"""
        # number of frames per activity
        step = m.ceil(self.n_frames / self._K)
        modulo = self.n_frames % self._K
        for action in range(self._K):
            # uniformly distribute remainder per actions if n_frames % K != 0
            self._z += [action] * (step - 1 * (modulo <= action) * (modulo != 0))
        self._z = np.asarray(self._z, dtype=int)
        try:
            assert len(self._z) == self.n_frames
        except AssertionError:
            logger.error('Wrong initialization for video %s', self.path)

    def _init_fg_mask(self):
        indexes = [i for i in range(self.n_frames) if i % 2]
        self.fg_mask[indexes] = False
        # todo: have to check if it works correctly
        # since it just after initialization
        self._z[self.fg_mask == False] = -1

    def _subact_count_update(self):
        c = Counter(self._z)
        # logger.debug('%s: %s' % (self.name, str(c)))
        self.a = []
        for subaction in range(self._K):
            self.a += [c[subaction]]

    def update_indexes(self, total):
        self.global_range = np.zeros(total, dtype=bool)
        self.global_range[self.global_start: self.global_start + self.n_frames] = True

    def reset(self):
        """If features from here won't be in use anymore"""
        self._features = None

    def z(self, pi=None):
        """Construct z (framewise label assignments) from ordering and counting.
        Args:
            pi: order, if not given the current one is used
        Returns:
            constructed z out of indexes instead of actual subactivity labels
        """
        self._z = []
        self._z_idx = []
        if pi is None:
            pi = self._pi
        for idx, activity in enumerate(pi):
            self._z += [int(activity)] * self.a[int(activity)]
            self._z_idx += [idx] * self.a[int(activity)]
        if opt.bg:
            z = np.ones(self.n_frames, dtype=int) * -1
            z[self.fg_mask] = self._z
            self._z = z[:]
            z[self.fg_mask] = self._z_idx
            self._z_idx = z[:]
        assert len(self._z) == self.n_frames
        return np.asarray(self._z_idx)

    def update_z(self, z):
        self._z = np.asarray(z, dtype=int)
        self._subact_count_update()

    def _likelihood_z_a(self, frame_idx):
        """ Likelihoods for different variants of subactivity for given frame.

        Compute all variations of z as for current frame number were sampled
        different activities and compute likelihood for each sequence.
        Args:
            frame_idx: frame number for which different subactivities are tested
        Returns:
            likelihoods for each sequence with the same ordering but different
            number of frames per activity, matrix size of n_subact X n_frames
        """
        if opt.bg:
            if np.sum(self._valid_likelihood[frame_idx]) == 0:
                return np.ones(self._K) * -np.inf
        var_z = np.zeros((self._K, self.n_frames))
        # fill table in with init subactivities
        var_z[:] = self.z()
        # change one label of current subactivity to one of the possible ones
        var_z[:, frame_idx] = np.arange(self._K)
        # rearrange
        var_z[:, self.fg_mask] = np.sort(var_z[:, self.fg_mask], axis=1)
        likelihood = self._helper_likelihood_z(var_z)
        reodered_lk = np.zeros(self._K)
        for idx, act in enumerate(self._pi):
            reodered_lk[act] = likelihood[idx]
        return reodered_lk

    def _likelihood_z_v(self, k, mallow_model):
        """  Likelihoods for different variants of ordering for given position
        in inverse count vector

        Compute all variations of z where current inverse count vector was
        changed in one position
        Args:
            k: number of subactivity which value in inverse count vector should
            mallow_model: obj of Mallow class
        Returns:
            likelihoods for each valid ordering
        """
        inv_count_v = self.inv_count_v.copy()
        possible_vals = list(range(self._K - k))
        var_z = np.zeros((len(possible_vals), self.n_frames))
        pi = []
        for val in possible_vals:
            inv_count_v[k] = val
            pi.append(mallow_model.ordering(inv_count_v))
            self.z(pi=pi[-1])
            var_z[val, :] = self._z
        return self._helper_likelihood_z(var_z, k_i=k, pi=pi)

    def _likelihood_vit(self, k, mallow_model):
        """Likelihoods for different variants of ordering for given position
        in inverse count vector computed with viterbi decoding
        Returns:
            likelihoods for each valid ordering
        """
        inv_count_v = self.inv_count_v.copy()
        possible_vals = list(range(self._K - k))
        likelihoods = []
        for val in possible_vals:
            inv_count_v[k] = val
            pi = mallow_model.ordering(inv_count_v)
            likelihoods.append(self.viterbi(pi=pi))

        return np.asarray(likelihoods)

    def _helper_likelihood_z(self, var_z, k_i=0, pi=None):
        """ Helper computation for a and v likelihoods together.

        Compute likelihood for different possible z sequences, e.i. we need only
        max, than multiply only parts which are differ
        Args:
            var_z: various sequences of z (they are not differ much because of
                only one value of counts (a) was changed in each if to compare \
                with original)
            k_i: number of subaction from which count likelihood
            pi: order to use, if None -> current is used
        Returns:
            likelihoods for each sequence
        """
        uniq_mask = var_z - var_z[0]
        uniq_mask = np.abs(np.sum(uniq_mask, axis=0)) > 0
        # compute part of likelihoods for various z which differ from each other
        likelihoods = np.zeros(self._K - k_i)
        for k in range(self._K - k_i):
            # ee = var_z[k, uniq_mask]
            if pi is None:
                actions = list(map(lambda idx: self._pi[int(idx)], var_z[k, uniq_mask]))
            else:
                # actions = list(map(lambda idx: pi[k][int(idx)], var_z[k, uniq_mask]))
                actions = var_z[k, uniq_mask]
            actions = np.array(actions, dtype=int)
            # since log space => sum instead of product
            t = self._likelihood_grid[uniq_mask, actions]
            likelihoods[k] = np.sum(self._likelihood_grid[uniq_mask, actions])
        return likelihoods

    def likelihood_update(self, subact, scores, trh=None):
        # for all frames
        self._likelihood_grid[:, subact] = scores[:]
        if trh is not None:
            self._valid_likelihood[:, subact] = False
            self._valid_likelihood[:, subact] = scores > trh

    def valid_likelihood_update(self, trhs):
        for trh_idx, trh in enumerate(trhs):
            self._valid_likelihood[:, trh_idx] = False
            self._valid_likelihood[:, trh_idx] = self._likelihood_grid[:, trh_idx] > trh

    def save_likelihood(self):
        """Used for multiprocessing"""
        dir_check(os.path.join(opt.data, 'likelihood'))
        np.savetxt(os.path.join(opt.data, 'likelihood', self.name), self._likelihood_grid)
        # print(os.path.join(opt.data, 'likelihood', self.name))

    def load_likelihood(self):
        """Used for multiprocessing"""
        path_join = os.path.join(opt.data, 'likelihood', self.name)
        self._likelihood_grid = np.loadtxt(path_join)

    def get_likelihood(self):
        return self._likelihood_grid

    def subactivity_sampler(self, subact_counter):
        """Sample most probable subactivity label for each frame

        In the end of computation the histogram of occurrences of subactivities
        in the video will be gotten.
        Args:
             subact_counter: statistic of subactivity occurrences in the entire
                video collection
        """
        # self._logger.debug(self.path)
        self.z()
        subact_counter -= self.a
        for frame_idx in range(self.n_frames):
            # one hot encoding for current subaction class
            # sum instead of product as in log space
            likelihoods = self._likelihood_z_a(frame_idx)
            subact_counter_video = self.a - self._subact_i_mask[self._z[frame_idx]]
            multinomial_probs = (np.log(subact_counter + subact_counter_video
                                        + self._theta_0))
            # in this context just sampling
            # it isn't assignment subact to the frame, but for the entire video
            self._z[frame_idx] = np.argmax(likelihoods + multinomial_probs)
            self.fg_mask[frame_idx] = True

            self._subact_count_update()

        if opt.bg:
            self.fg_mask = np.sum(self._valid_likelihood, axis=1) > 0
            self._z[self.fg_mask == False] = -1  # bg frames
        self._subact_count_update()
        return self.a, subact_counter

    def ordering_sampler(self, mallow_model):
        # logger.debug('Video: %s' % self.name)
        if opt.ordering:
            for k in range(self._K - 1):
                possible_vals = np.array(range(self._K - k))
                # sum instead of product as in log space
                probs = mallow_model.single_term_prob(possible_vals, k)
                likelihood = self._likelihood_z_v(k, mallow_model)
                probs += likelihood
                self.inv_count_v[k] = np.argmax(probs)

            self._pi = mallow_model.ordering(self.inv_count_v)
        self.z()
        # if opt.bg:
        #     self.fg_mask = np.sum(self._valid_likelihood, axis=1) > 0

        # save current segmentation
        name = str(self.name) + '_' + opt.log_str + 'iter%d' % self.iter + '.txt'
        np.savetxt(join(opt.data, 'segmentation', name),
                   np.asarray(self._z), fmt='%d')
        return list(self._pi)

    def _viterbi_inner(self, pi, save=False):
        grammar = Grammar(pi)
        if np.sum(self.fg_mask):
            viterbi = Viterbi(grammar=grammar, probs=(-1 * self._likelihood_grid[self.fg_mask]))
            viterbi.inference()
            viterbi.backward(strict=True)
            z = np.ones(self.n_frames, dtype=int) * -1
            z[self.fg_mask] = viterbi.alignment()
            score = viterbi.loglikelyhood()
        else:
            z = np.ones(self.n_frames, dtype=int) * -1
            score = -np.inf
        # viterbi.calc(z)
        if save:
            name = '%s_%s' % (str(pi), self.name)
            save_path = join(opt.data, 'likelihood', name)
            with open(save_path, 'w') as f:
                # for pi_elem in pi:
                #     f.write('%d ' % pi_elem)
                # f.write('\n')
                f.write('%s\n' % str(score))
                for z_elem in z:
                    f.write('%d ' % z_elem)
        return z, score

    # @timing
    def viterbi(self):
        """Decode segmentation out of likelihood values and save output segmentation"""

        pi = self._pi
        log_probs = self._likelihood_grid
        if np.max(log_probs) > 0:
            self._likelihood_grid = log_probs - (2 * np.max(log_probs))
        alignment, return_score = self._viterbi_inner(pi, save=True)
        self._z = np.asarray(alignment).copy()

        self._subact_count_update()

        name = str(self.name) + '_' + opt.log_str + 'iter%d' % self.iter + '.txt'
        np.savetxt(join(opt.data, 'segmentation', name),
                   np.asarray(self._z), fmt='%d')
        # print('path alignment:', join(opt.data, 'segmentation', name))

        return return_score

    def resume(self):
        name = str(self.name) + '_' + opt.log_str + 'iter%d' % self.iter + '.txt'
        self._z = np.loadtxt(join(opt.data, 'segmentation', name))
        self._subact_count_update()




