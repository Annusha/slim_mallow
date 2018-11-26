#!/usr/bin/env python

"""Segmentation of video for further sampling for example.
Implemented only uniform one."""

__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import os
import numpy as np

from utils.arg_pars import opt

# n_clusters = opt.n_clusters
# n_segments = opt.n_segments


class Segmentation(object):
    def __init__(self):
        self._save_folder_name = ''

    def n_segments(self):
        return 0

    def max_seg_len(self):
        return 0

    def save_folder(self, folder_name):
        self._save_folder_name = folder_name
        if not os.path.isdir(self._save_folder_name):
            os.mkdir(self._save_folder_name)


class Uniform(Segmentation):
    def __init__(self, n_segments, max_seg_len=0):
        super(Segmentation, self).__init__()
        self._n_segments = n_segments
        self._max_seg_len = max_seg_len
        self._length = []

    def length_file(self, filename_len, save_folder_name=None):
        if save_folder_name is not None:
            self.save_folder(save_folder_name)

        # string format: relative_filename length
        with open(filename_len, 'r') as f:
            for line in f:
                line = line.split()
                filename, length = line[0], int(line[1])
                filename = filename.split('.')[0]
                n_segments = self._n_segments
                # segments' length should be restricted by maximum length if non zero
                if float(length) / self._n_segments > self._max_seg_len and self._max_seg_len:
                    n_segments = length / self._max_seg_len
                # number of segments should be less or equal then number of frames
                while float(length) / n_segments < 1:
                    n_segments -= 1
                segmentation = []
                for stop in np.linspace(0, length, num=n_segments, endpoint=False, dtype=int):
                    if stop == 0:
                        start = stop
                        continue
                    segmentation += [[start, stop - 1]]
                    start = stop
                # last segment
                if length - start > 1:
                    segmentation += [[start, length - 1]]

                # write segmentation in file
                with open(os.path.join(self._save_folder_name, filename), 'w') as g:
                    for segment in segmentation:
                        g.write('%d %d\n' % (segment[0], segment[1]))


if __name__ == '__main__':
    segment_obj = Uniform(n_segments=20)
    postfix = 'segments/lens.txt'
    prefixs = [opt.kinetics, opt.data, opt.s1, opt.video]
    for prefix in prefixs:
        segment_obj.length_file(filename_len=os.path.join(prefix, postfix),
                                save_folder_name=os.path.join(prefix, 'segments'))













