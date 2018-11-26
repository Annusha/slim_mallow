#!/usr/bin/env python

"""Smth conserning kmeans. I do not remember what is here  exactly xD
Looks like testing pure kmeans alg for each video separately and for the entire
 video collection at once. Depends on how much clusters we'll define and
 sampling frame frequency. """

__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

from sklearn.cluster import KMeans
import numpy as np
# from sklearn.cluster.tests.test_dbscan import n_clusters

from utils.frames_sampling import FramesSampling
from utils.arg_pars import opt
from utils.accuracy_class import Accuracy


class KmeansClass(object):
    def __init__(self, n_frames):
        self._corpus = False
        self._n_frames = n_frames

        self._accuracy = Accuracy(self._n_frames)
        self._sampling = FramesSampling(root_folder=opt.data, n_frames=self._n_frames)

    def videowise_kmean(self):
        print('{: <5} {: <30} {: <7}\t{:<10}\t{:<10}'
              .format('idx', 'filename', 'current', 'average', 'n_clusters'))

        for idx, (features, gt_set, indices, boundaries, filename) in \
                enumerate(self._sampling.sampling_gen()):
            gt_subset = gt_set[indices]

            # for each video separately (number of clusters is known from ground truth)
            n_clusters = len(np.unique(gt_subset))
            kmean = KMeans(n_clusters=n_clusters, random_state=0)
            kmean.fit(features)

            self._accuracy.predicted_labels = kmean.labels_
            self._accuracy.gt_labels = gt_set
            self._accuracy.params = (boundaries, indices)
            self._accuracy.mof()

            current_accuracy, average_accuracy = self._accuracy()
            print('{: <5} {: <30} : {:01.3f} \t av: {:01.3f} \t n_cl:{:d}'
                  .format(idx, filename, current_accuracy, average_accuracy, n_clusters))

    def corpus_kmean(self):
        # key: filename, val: [start_index, end_index]
        files = {}
        # all - comprised all videos (only sampled frames)
        all_features = None
        all_indices = None
        # full - comprised labels not only for sampled frames (for each one)
        full_gt = None
        full_boundaries = None

        for idx, (features, gt_set, indices, boundaries, filename) in \
                enumerate(self._sampling.sampling_gen()):
            gt_subset = gt_set[indices]
            boundaries += len(full_gt) if full_gt is not None else 0

            all_features = self._join_helper(all_features, features, np.vstack)
            all_indices = self._join_helper(all_indices, indices, np.hstack)
            full_gt = self._join_helper(full_gt, gt_set, np.hstack)
            full_boundaries = self._join_helper(full_boundaries, boundaries, np.vstack)

            files[filename] = [all_features.shape[0] - features.shape[0], all_features.shape[0]]

        print('Number of features together: %d' % (all_features.shape[0]))
        print('Number frames per segment: %d' % self._n_frames)
        for n_clusters in [48, 100, 500, 1000, 5000, 10000]:
            kmean = KMeans(n_clusters=n_clusters, random_state=0)
            kmean.fit(all_features)

            self._accuracy.predicted_labels = kmean.labels_
            self._accuracy.gt_labels = full_gt
            self._accuracy.params = (full_boundaries, all_indices)
            self._accuracy.mof()
            current_accuracy, average_accuracy = self._accuracy()
            print('accuracy MoF all videos: {:01.3f} \t n_clusters:{:d}'
                  .format(current_accuracy, n_clusters))

    @staticmethod
    def _join_helper(conjunction, sample, join_f):
        # join_f either np.hstack, or np.vstack
        if conjunction is None:
            conjunction = sample
        else:
            conjunction = join_f((conjunction, sample))
        return conjunction


def kmean_wrapper(corpus=False):
    sampling = FramesSampling(root_folder=opt.data, n_frames=2)
    accuracy = Accuracy()
    if not corpus:
        print('{: <5} {: <30} {: <7}\t{:<10}\t{:<10}'
              .format('idx', 'filename', 'current', 'average', 'n_clusters'))
    # key: filename, val: [start_index, end_index]
    files = {}
    # all - comprised all videos (only sampled frames)
    all_features = None
    all_labels = None
    # full - comprised labels not only for sampled frames (for each one)
    full_gt = None
    full_boundaries = None

    for idx, (features, gt_set, indices, boundaries, filename) in enumerate(sampling.sampling_gen()):
        gt_subset = gt_set[indices]
        if corpus:
            if all_features is None:
                all_features = features
                all_labels = gt_subset
                full_gt = gt_set
                full_boundaries = boundaries

            else:
                all_features = np.vstack((all_features, features))
                all_labels = np.hstack((all_labels, gt_subset))
                # take into account bias for zero frame gotten from previous videos
                boundaries += len(full_gt)
                full_gt = np.vstack((full_gt, gt_set))
                full_boundaries = np.vstack((full_boundaries, boundaries))

            files[filename] = [all_features.shape[0] - features.shape[0], all_features.shape[0]]

        else:
            # for each video separately (number of clusters is known from ground truth)
            n_clusters = len(np.unique(gt_subset))
            kmean = KMeans(n_clusters=n_clusters, random_state=0)
            kmean.fit(features)

            accuracy.predicted_labels = kmean.labels_
            accuracy.gt_labels = gt_subset
            accuracy.mof()

            current_accuracy, average_accuracy = accuracy()
            print('{: <5} {: <30} : {:01.3f} \t av: {:01.3f} \t n_cl:{:d}'
                  .format(idx, filename, current_accuracy, average_accuracy, n_clusters))
    if corpus:
        # n_clusters = len(np.unique(all_labels))
        print('Number of features together: %d'%(all_features.shape[0]))
        for n_clusters in [48, 100, 500, 1000, 5000, 10000]:
            kmean = KMeans(n_clusters=n_clusters, random_state=0)
            kmean.fit(all_features)

            accuracy.predicted_labels = kmean.labels_
            accuracy.gt_labels = all_labels
            accuracy.mof()
            current_accuracy, average_accuracy = accuracy()
            print('accuracy MoF all videos: {:01.3f} \t n_clusters:{:d}'
                  .format(current_accuracy, n_clusters))


if __name__=='__main__':
    # kmean_wrapper(corpus=True)
    kmean_instance = KmeansClass(n_frames=1)
    # kmean_instance.videowise_kmean()
    kmean_instance.corpus_kmean()


