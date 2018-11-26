#!/usr/bin/env python

"""Load poses for video collection and analysis them. Reliable information can
be extracted only from video with defined poses in more than 60% of duration.
"""

__author__ = 'Anna Kukleva'
__date__ = 'September 2018'

import os

from utils.arg_pars import opt, logger
from pose_utils.pose_video_instance import PoseVideoInstance
from utils.utils import AverageLength, timing


class PoseCollection(object):
    def __init__(self, K=6, videos=None) -> None:
        logger.debug('.')
        self._K = K
        self._videos = videos
        self._video_poses = {}
        self._init_videos()

        self._rel_vidnames = []
        self._avg_len = AverageLength()

    def __getitem__(self, key):
        return self._video_poses[key]

    @timing
    def _init_videos(self) -> None:
        """Init collection for certain subaction"""
        logger.debug('.')
        if 'YTI' in opt.dataset_root:
            logger.debug('YTI video collections')
            assert self._videos is not None
            for video in self._videos:
                pose_video_inst = PoseVideoInstance(path=video.name,
                                                    length=video.n_frames,
                                                    K=self._K,
                                                    yti=True)
                self._video_poses[video.name] = pose_video_inst
        else:
            if os.path.exists(opt.pose_path):
                for filename in os.listdir(opt.pose_path):
                    if opt.subaction not in filename:
                        continue
                    for video in self._videos:
                        if video.name in filename:
                            pose_video_inst = PoseVideoInstance(path=os.path.join(opt.pose_path,
                                                                                  filename),
                                                                K=self._K,
                                                                length=video.n_frames)
                            break
                    pose_video_inst.joint_der(mag_use=True)
                    self._video_poses[pose_video_inst.name] = pose_video_inst
            else:
                for video in self._videos:
                    pose_video_inst = PoseVideoInstance(path=video.name,
                                                        K=self._K,
                                                        length=video.n_frames)
                    pose_video_inst.joint_der(mag_use=True)
                    self._video_poses[pose_video_inst.name] = pose_video_inst

                # if len(self._video_poses) == 15:
                #     break

        logger.debug('In video collection %d videos' % len(self._video_poses))

    @timing
    def _reliable_videos(self) -> None:
        """List videos where feasible pose estimation"""
        logger.debug('.')
        for video_unit in self._video_poses.values():
            if video_unit.get_reliable():
                self._rel_vidnames.append(video_unit.name)

    @timing
    def _pose_segmentation(self) -> None:
        """Define segmentation for reliable videos (video parts)"""
        logger.debug('.')
        for vidname in self._rel_vidnames:
            video_unit = self._video_poses[vidname]
            self._avg_len.add_segments(video_unit.get_rel_segs())

    @timing
    def _uniform_segmentation(self) -> None:
        """Define uniform segmentation for the rest of the videos

        Based on average segment length which was computed from estimated poses,
        define uniform segmentation for unreliable videos and reliable videos
        but with incomplete segmentation
        """
        logger.debug('.')
        for video_unit in self._video_poses.values():
            if video_unit.full_segm():
                video_unit.update_framewise_labels()
                continue
            break_points = []
            for seg in list(video_unit.unseg_parts()):
                if len(seg):
                    start, end = seg
                    break_points.append(start)
                    n_seg = (end - start) // self._avg_len()
                    try:
                        modulo = ((end - start) % self._avg_len()) // n_seg
                    except ZeroDivisionError:
                        pass
                    for i in range(n_seg - 1):
                        # uniformer = (modulo <= i) * (modulo != 0)
                        new_pnt = break_points[-1] + self._avg_len() + modulo
                        break_points.append(new_pnt)
                    break_points.append(end)
            # update break points for video
            video_unit.update_rel_segs(break_points)
            video_unit.update_framewise_labels()

    @timing
    def save_new_segm_img(self):
        for video_unit in self._video_poses.values():
            video_unit.joint_der(mag_use=True, save=True, visual=True)

    @timing
    def segment_videos(self):
        """Segment and save plots"""
        self._reliable_videos()
        self._pose_segmentation()
        self._uniform_segmentation()
        self.save_new_segm_img()
        assert 1

    def segment_collection(self):
        logger.debug('.')
        self._reliable_videos()
        self._pose_segmentation()
        self._uniform_segmentation()

    def labels_without_segmentation(self):
        logger.debug('.')
        for video_unit in self._video_poses.values():
            video_unit.baseline_frame_labels()


if __name__ == '__main__':
    test = PoseCollection()
    test.segment_videos()





