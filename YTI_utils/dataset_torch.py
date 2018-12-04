#!/usr/bin/env python

"""Implementation of dataset structure for dataloader from pytorch.
Two ways of training: with ground truth labels and with labels from current
segmentation given the algorithm"""

__all__ = ['load_data']
__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

from torch.utils.data import Dataset
import torch
from torchvision import datasets, transforms
import numpy as np
from os.path import join
import re
import logging

from utils.mapping import GroundTruth
from utils.arg_pars import opt
from utils.logging_setup import logger
from utils.utils import join_data


class FeatureDataset(Dataset):
    def __init__(self, root_dir, end, subaction='coffee', videos=None,
                 features=None, regression=False, vae=False):
        """
        Filling out the _feature_list parameter. This is a list of [video name,
        index frame in video, ground truth, feature vector] for each frame of each video
        :param root_dir: root directory where exists folder 'ascii' with features or pure
         video files
        :param end: extension of files for current video representation (could be 'gz',
        'txt', 'avi')
        """
        self._logger = logging.getLogger('basic')
        self._logger.debug('FeatureDataset')

        self._root_dir = root_dir
        self._feature_list = None
        self._end = end
        self._old2new = {}
        self._videoname2idx = {}
        self._idx2videoname = {}
        self._videos = videos
        self._subaction = subaction
        self._features = features
        self._regression = regression
        self._vae = vae
        self.gt_map = GroundTruth()

        if self._videos is None:
            self._with_gt()
        elif not vae:
            self._with_predictions()
        else:
            self._for_vae()

        subactions = np.unique(self._feature_list[..., 2])
        for idx, old in enumerate(subactions):
            self._old2new[int(old)] = idx

    def index2name(self):
        return self._idx2videoname

    def _with_gt(self):
        self._logger.debug('__init__')
        fileindex = 0
        len_file = join(self._root_dir, 'segments', 'lens.txt')
        with open(len_file, 'r') as f:
            for line in f:
                if self._subaction not in line:
                    continue
                match = re.match(r'(\w*)\.\w*\s*(\d*)', line)
                filename = match.group(1)
                filepath = filename
                if opt.data_type == 2:
                    filepath = self._subaction + '/' + filename
                self._videoname2idx[filename] = fileindex
                self._idx2videoname[fileindex] = filename
                fileindex += 1

                n_frames = int(match.group(2))
                # because of there can be inconsistency between number of gt labels and
                # corresponding number of frames for current representation
                if len(self.gt_map.gt[filename]) == n_frames:
                    names = np.asarray([self._videoname2idx[filename]] * n_frames)\
                        .reshape((-1, 1))
                    idxs = np.asarray(list(range(0, n_frames))).reshape((-1, 1))
                    gt_file = np.asarray(self.gt_map.gt[filename]).reshape((-1, 1))
                    features = np.loadtxt(join(self._root_dir, 'ascii',
                                               filepath + '.%s' % self._end))
                    if opt.data_type == 2:
                        features = features[:, 1:]
                    temp_feature_list = join_data(None,
                                                  (names, idxs, gt_file, features),
                                                  np.hstack)
                else:
                    min_len = np.min((len(self.gt_map.gt[filename]), n_frames))
                    names = np.asarray([self._videoname2idx[filename]] * min_len)\
                        .reshape((-1, 1))
                    idxs = np.asarray(list(range(0, min_len))).reshape((-1, 1))
                    gt_file = np.asarray(self.gt_map.gt[filename][:min_len]).reshape((-1, 1))
                    features = np.loadtxt(join(self._root_dir, 'ascii',
                                               filepath + '.%s' % self._end))[:min_len]
                    if opt.data_type == 2:
                        features = features[:, 1:]
                    temp_feature_list = join_data(None,
                                                  (names, idxs, gt_file, features),
                                                  np.hstack)
                self._feature_list = join_data(self._feature_list,
                                               temp_feature_list,
                                               np.vstack)

    def _with_predictions(self):
        self._logger.debug('__init__')
        for video_idx, video in enumerate(self._videos):
            filename = re.match(r'[\.\/\w]*\/(\w+).\w+', video.path)
            if filename is None:
                logging.ERROR('Check paths videos, template to extract video name'
                              ' does not match')
            filename = filename.group(1)
            self._videoname2idx[filename] = video_idx
            self._idx2videoname[video_idx] = filename

            names = np.asarray([video_idx] * video.n_frames).reshape((-1, 1))
            idxs = np.asarray(list(range(0, video.n_frames))).reshape((-1, 1))
            if self._regression:
                gt_file = np.asarray(video.pose.frame_labels).reshape((-1, 1))
            else:
                if opt.gt_training:
                    gt_file = np.asarray(video._gt).reshape((-1, 1))
                else:
                    gt_file = np.asarray(video._z).reshape((-1, 1))
            if self._features is None:
                features = video.features()
            else:
                features = self._features[video.global_range]
            temp_feature_list = join_data(None,
                                          (names, idxs, gt_file, features),
                                          np.hstack)
            self._feature_list = join_data(self._feature_list,
                                           temp_feature_list,
                                           np.vstack)
        self._features = None

    def _for_vae(self):
        # todo: different types of including time domain
        self._logger.debug('__init__')
        for video_idx, video in enumerate(self._videos):
            self._videoname2idx[video.name] = video_idx
            self._idx2videoname[video_idx] = video.name

            names = np.asarray([video_idx] * video.n_frames).reshape((-1, 1))
            idxs = np.asarray(list(range(0, video.n_frames))).reshape((-1, 1))
            gt_file = np.zeros(video.n_frames).reshape((-1, 1))
            if opt.vae_dim == 1:
                relative_time = np.asarray(video.pose.frame_labels).reshape((-1, 1))
                gt_file = relative_time.copy()
            else:
                relative_time = video.pose.relative_segments()
            if self._features is None:
                features = video.features()
            else:
                features = self._features[video.global_range]

            if opt.concat > 1:
                video_feature_concat = features[:]
                last_frame = features[-1]
                for i in range(opt.concat - 1):
                    video_feature_concat = np.roll(video_feature_concat, -1, axis=0)
                    video_feature_concat[-1] = last_frame
                    features = join_data(features,
                                         video_feature_concat,
                                         np.hstack)

            relative_time *= opt.time_weight
            if not opt.label:
                temp_feature_list = join_data(None,
                                              (names, idxs, gt_file,
                                               features, relative_time),
                                              np.hstack)
            else:
                labels = np.asarray(video._z).reshape((-1, 1))
                temp_feature_list = join_data(None,
                                              (names, idxs, gt_file,
                                               features, relative_time, labels),
                                              np.hstack)

            self._feature_list = join_data(self._feature_list,
                                           temp_feature_list,
                                           np.vstack)

    def __len__(self):
        return len(self._feature_list)

    def __getitem__(self, idx):
        name, frame_idx, gt_file, *features = self._feature_list[idx]
        # features = np.loadtxt(join(self._root_dir, 'ascii',
        #                            name + '.%s' % self._end))

        gt_out = None
        if self._regression or self._vae:
            gt_out = gt_file
        else:
            one_hot = np.zeros(self.n_subact())
            one_hot[self._old2new[int(gt_file)]] = 1
            gt_out = one_hot
        # features = torch.from_numpy(np.asarray(features))
        return np.asarray(features), gt_out, name
        # return features, one_hot, 1.0
        # return features[int(frame_idx)], one_hot, name

    def n_subact(self):
        return len(self._old2new)


def load_data(root_dir, end, subaction, videos=None, names=None, features=None,
              regression=False, vae=False):
    """Create dataloader within given conditions
    Args:
        root_dir: path to root directory with features
        end: extension of files
        subaction: complex activity
        videos: collection of object of class Video
        names: empty list as input to have opportunity to return dictionary with
            correspondences between names and indices
        features: features for the whole video collection
        regression: regression training
        vae: dataset for vae with incorporated relative time in features
    Returns:
        iterative dataloader
        number of subactions in current complex activity
    """
    logger.debug('create DataLoader')
    dataset = FeatureDataset(root_dir, end, subaction,
                             videos=videos,
                             features=features,
                             regression=regression,
                             vae=vae)
    if names is not None:
        names[0] = dataset.index2name()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=(not opt.save_embed_feat),
                                             num_workers=opt.num_workers)

    return dataloader, dataset.n_subact()


def load_mnist(train=True):
    """Just to test how embedding works on simple and known dataset"""
    if train:
        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=opt.batch_size, shuffle=True)
    else:
        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=opt.batch_size, shuffle=True)

    return dataloader, 10
