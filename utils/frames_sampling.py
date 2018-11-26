#!/usr/bin/env python

"""To reduce computations there is possibility to process not each frame but
with some frequency. Here implementation of uniform sampling, bth I haven't use
it at all. Just leave it here because I've already written it."""

__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import os
import numpy as np
import re

from utils.mapping import ground_truth


class Sampling(object):
    def __init__(self, root_folder):
        self._root_folder = root_folder
        self._segments_folder = 'segments'
        np.random.seed(0)


class FramesSampling(Sampling):
    # for each file with predefined segmentation sample by random n frames from each segment
    def __init__(self, root_folder, n_frames=1):
        super(FramesSampling, self).__init__(root_folder)

        self._n_frames = n_frames

    def sampled_features(self, filename):
        boundaries = []
        # open file with predefined segments for the current file
        with open(os.path.join(self._root_folder, 'segments', filename), 'r') as f:
            frame_idxs_file = []
            for line in f:
                start, end = [int(i) for i in line.split()]
                boundaries.append([start, end])
                try:
                    frame_idxs_segment = np.random.choice(range(start, end),
                                                          size=self._n_frames,
                                                          replace=False)
                except ValueError:
                    frame_idxs_segment = np.random.choice(range(start, end),
                                                          size=self._n_frames,
                                                          replace=True)
                frame_idxs_file += list(frame_idxs_segment)
        # load features from the corresponding file and return subset of these features
        # indices predefined in frame_idxs_file variable
        extended_filename = ''
        for root, folder, filenames in os.walk(os.path.join(self._root_folder, 'ascii')):
            if not filenames:
                continue
            for fn in filenames:
                if re.match(filename, fn) is not None:
                    extended_filename = os.path.join(root, fn)
                    break
        features = np.loadtxt(extended_filename)
        features = features[frame_idxs_file]

        # extract gt for chosen features
        gt_set = ground_truth[filename]
        boundaries = np.asarray(boundaries)
        return features, gt_set, frame_idxs_file, boundaries

    def sampling_gen(self):
        for filename in ground_truth.keys():
            features, gt_set, indices, boundaries = self.sampled_features(filename)
            yield features, gt_set, indices, boundaries, filename

