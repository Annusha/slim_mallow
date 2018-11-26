#!/usr/bin/env python

"""Rewrite files for each video for each frame into one.
Get rid of Pose class and use pure numpy matrices for the storing pose
parameters.
"""

__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import re
import os
from os.path import join
import numpy as np

from utils.arg_pars import opt, logger
from pose_utils.Pose import PoseExtended
from utils.utils import dir_check
from utils.mapping import ground_truth


def create_one_pose_file(self):
    """Composing of extracted poses into one file for each video

    Save one file per video with all poses related to this video into
    folder: opt.pose_path/poses
    Name of them is vid_name.avi.npy.
    If compare with features then there are also stereo videos: ch0, ch1.
    For consistency with features only ch0 is in use.
    """
    filename_template = 'frame_%d-OriginalCMUMultiPose.npy'
    dir_check(join(opt.data, 'video_poses_POSE'))

    init_pose_path = '/media/data/kukleva/lab/Breakfast/videos/POSES/BreakfastII/poses'

    for idx_folder, folder in enumerate(os.listdir(init_pose_path)):
        match = re.match(r'(\w*).avi', folder)
        video_name = match.group(1)
        poses_video = []
        n_files = len(os.listdir(os.path.join(init_pose_path, folder)))
        frame_counter = 0
        while n_files:
            try:
                poses = np.load(join(init_pose_path, folder,
                                     filename_template % frame_counter))
                n_files -= 1
                if len(poses):
                    max_conf = -np.inf
                    pose_to_save = None
                    # choose pose with max confidence
                    for idx, pose in enumerate(poses):
                        if max_conf < pose.subset_score:
                            max_conf = pose.subset_score
                            pose_to_save = pose
                    if pose_to_save is not None:
                        poses_video.append(PoseExtended(pose_to_save,
                                                        frame_counter - 1))
            except FileNotFoundError:
                pass
            frame_counter += 1
        result = np.array(poses_video)
        np.save(join(opt.data, 'video_poses_POSE', folder + '.npy'), result)
        logger.debug('%d Save pose for %s' % (idx_folder, video_name))


def rewrite_pose_files():
    """Rewrite dummy class Pose to usual matrices"""
    dir_check(os.path.join(opt.data, 'video_poses'))
    for idx, filename in enumerate(os.listdir(os.path.join(opt.data, 'video_poses_POSE'))):
        try:
            poses = np.load(os.path.join(os.path.join(opt.data,
                                                      'video_poses_POSE',
                                                      filename)))
        except OSError:
            print('Passed %s' % filename)
            continue
        search = re.search(r'(\w*).avi.npy', filename)
        video_name = search.group(1)
        if video_name.endswith('ch0'):
            video_name = video_name.replace('_ch0', '')
            video_name = list(video_name.partition('stereo'))
            video_name.insert(2, '01')
            video_name = ''.join(video_name)
        if video_name.endswith('ch1'):
            continue
        # x, y, score each has dim 15
        pose_mat = np.zeros((15*3, len(ground_truth[video_name])))
        for pose in poses:
            x = pose.joints[:, 0]
            y = pose.joints[:, 1]
            try:
                pose_mat[:15, pose.frame] = x
                pose_mat[15:30, pose.frame] = y
                pose_mat[30:, pose.frame] = pose.score
            except IndexError:
                # number of processed frames are more than in gt labels
                break
        np.save(os.path.join(opt.data, 'video_poses',
                             '%s.npy' % video_name),
                pose_mat)
        logger.debug('%d Poses for %s saved' % (idx, video_name))


if __name__ == '__main__':
    print('Pose smth')
    # create_one_pose_file()
    # rewrite_pose_files()
