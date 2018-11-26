#!/usr/bin/env python

"""Pose analysis module. The idea is to extract first (second) derivatives of
hand position to understand where they stopped or change direction of
movement.
"""

__author__ = 'Anna Kukleva'
__date__ = 'September 2018'

import matplotlib.pyplot as plt
import numpy as np
import re
import cv2
import os
from scipy import signal

from pose_utils.Pose import pose2idx
from utils.arg_pars import opt, logger
from utils.mapping import ground_truth, order, index2label
from utils.utils import dir_check, merge


class PoseVideoInstance:
    """Analysis of hand positions for one video"""
    def __init__(self, path, thr_rel=0.6, K=6, length=0, yti=False):
        """
        Args:
            path: full path to the pose file for particular video
            thr_rel: threshold for defining video reliability
        """
        # self._rattr = ['RWrist', 'RElbow', 'RShoulder']
        # self._lattr = ['LWrist', 'LElbow', 'LShoulder']
        self._rattr = ['RWrist', 'RElbow']
        self._lattr = ['LWrist', 'LElbow']
        # self._rattr = ['RWrist']
        # self._lattr = ['LWrist']
        self._path = path
        self._K = K
        if not yti:
            try:
                self._poses = np.load(path)
                if length:
                    self._poses = self._poses[-length:]
            except FileNotFoundError:
                if length:
                    # video name if path is not valid
                    self._name = path
                    self._poses = np.zeros((45, length))
                else:
                    raise FileNotFoundError
            else:
                # video name
                search = re.search(r'(\w*).npy', self._path)
                self._name = search.group(1)
                if 'ch0' in self._name:
                    self._name = '_'.join(self._name.split('_')[:-1])
                if 'ch1' in self._name:
                    raise NameError('Do not consider processing of the second channel '
                                    'of two stream camera')
        else:
            self._name = path
            self._poses = np.zeros((45, length))

        # should be more than 60% of the video length where poses were detected
        self._reliable = False
        self._thr_rel = thr_rel

        self._total_mag = None
        self._filt_total_mag = None
        self._frames = None

        self._local_mins = None
        self._frame_bounds = None
        # init framewise assignments for relative time labeling
        self.frame_labels = np.ones(self._poses.shape[1]) * -1

        self._order_comparator = 12

    def __len__(self) -> int:
        return self._poses.shape[1]

    def set_len(self, length):
        self._poses = self._poses[:, -length:]
        self.frame_labels = np.ones(self._poses.shape[1]) * -1

    @property
    def name(self):
        return self._name

    def get_reliable(self):
        if self._total_mag is None:
            self.joint_der(mag_use=True)
        self._reliability()
        return self._reliable

    def get_rel_segs(self):
        """Return segment boundaries only for reliable parts"""
        # if self._frame_bounds is None:
        self._frame_bounds = [self._frames[0]]
        if self._pose_segmentation():
            if self.__len__() - self._frames[-1] < self._order_comparator:
                self._frame_bounds += [self.__len__()]
            else:
                self._frame_bounds += [self._frames[-1]]
        return self._frame_bounds

    def update_rel_segs(self, update_bounds):
        """Merge already existed bounds and new ones"""
        self._frame_bounds = merge(self._frame_bounds, update_bounds)

    def _compute_total(self, derivatives):
        """Compute sum of all magnitudes of different joints, filter it

        Args:
            derivatives: sequence of magnitudes and angles for different joints
        Returns:
        """

        self._total_mag = np.zeros(len(self._frames))
        for idx, mag_local in enumerate(derivatives):
            if idx % 2:
                # take every second array (with magnitudes but without angles)
                continue
            mag_local = np.array(mag_local)
            mag_local[np.argwhere(np.isnan(mag_local))] = 0
            self._total_mag += mag_local

        b, a = signal.butter(3, 0.05)
        try:
            self._filt_total_mag = signal.filtfilt(b, a, self._total_mag)
        except ValueError:
            # there were no detected poses
            try:
                self._filt_total_mag = self._total_mag
            except Warning:
                logger.debug('Future warning inside module')

    def _reliability(self):
        # multiplication by 2 because of for poses was extracted every second frame
        non_zeros = np.nonzero(self._total_mag)[0].shape[0] * 2
        test_val = self._thr_rel * self._poses.shape[1]
        self._reliable = non_zeros > self._thr_rel * self._poses.shape[1]

    def _local_min(self):
        # todo: define proper @order, should depend on video length
        # todo: thr for to high minimums
        self._local_mins = signal.argrelextrema(data=self._filt_total_mag,
                                                comparator=np.less,
                                                order=self._order_comparator,
                                                mode='wrap')[0]
        ratio = len(self._total_mag) / self.__len__()
        if len(self._local_mins) < self._K * ratio:
            self._reliable = False
            self._local_mins = None
            return False
        return True

    def _pose_segmentation(self) -> bool:
        """Calculate extremes.

        Set self._frame_bounds as a frame numbers where local extremes were
        found
        """
        if self._local_min():
            if self._local_mins[0] == 0:
                self._local_mins = self._local_mins[1:]
            self._frame_bounds += list(np.asarray(self._frames)[self._local_mins])

            upp_bound = self.__len__() - self._frames[self._local_mins[-1]]
            if upp_bound < self._order_comparator:
                self._frame_bounds[-1] = self.__len__()
                # self._local_mins[-1] = self.__len__()
            return True
        else:
            self._frame_bounds = None
            return False

    def update_framewise_labels(self):
        """Assign to each frame relative time label"""
        for bound_idx, start in enumerate(self._frame_bounds[:-1]):
            end = self._frame_bounds[bound_idx + 1]
            self.frame_labels[start:end] = (end + start) / (2 * self.__len__())

    def baseline_frame_labels(self):
        """Assign to each frame its own relative time label without any additional
        segmentation which relies on hand position"""
        for frame_idx in range(len(self.frame_labels)):
            self.frame_labels[frame_idx] = frame_idx / self.__len__()

    def joint_der(self, mag_use=True, save=False, visual=False):
        """Compute derivatives for given joints and save/plot it

        Args:
            mag_use: separate derivatives for x and y directions or combined
                magnitude and angle
            save (bool): save plots with derivatives
            visual: if false it returns computed derivatives otherwise
                continue processing with either visualization or saving plots

        Returns:
            todo: specify what it returns
        """

        derivatives = []
        joint_names = self._rattr + self._lattr
        for joint_name in joint_names:
            joint_idx = pose2idx[joint_name]
            x_cur = self._poses[joint_idx, 0]
            y_cur = self._poses[15 + joint_idx, 0]
            dx, dy = [], []
            mag, angl = [], []

            frames_loc = [0]
            for frame_idx in range(self._poses.shape[1]):
                if np.sum(self._poses[:, frame_idx]) == 0:
                    continue
                x_next = self._poses[joint_idx, frame_idx]
                y_next = self._poses[15 + joint_idx, frame_idx]
                # x_next, y_next = pose.joints[pose2idx[joint_name]]
                frames_loc.append(frame_idx)
                dx.append((x_next - x_cur) / (frames_loc[-1] - frames_loc[-2]))
                dy.append((y_next - y_cur) / (frames_loc[-1] - frames_loc[-2]))
                mag.append(np.sqrt(dx[-1] ** 2 + dy[-1] ** 2))
                angl.append(np.arctan2(dy[-1], dx[-1]))
                x_cur, y_cur = x_next, y_next
            if mag_use:
                derivatives.append(mag)
                derivatives.append(angl)
            else:
                derivatives.append(dx)
                derivatives.append(dy)
        self._frames = frames_loc[1:]
        if visual:
            self.plot(derivatives, mag=mag_use, save=save)
        else:
            self._compute_total(derivatives)

    def full_segm(self) -> bool:
        """Check if the current segmentation covers all video frames or not

        Returns:
            True: if video completely segmented
            False: otherwise
        """
        if self._frame_bounds is None:
            return False
        complete_segm = True
        complete_segm = complete_segm & self._frame_bounds[0] == 0
        complete_segm = complete_segm & self._frame_bounds[-1] == self.__len__()
        return complete_segm

    def unseg_parts(self):
        """Return unsegmented parts of video

        In case of video comprises reliable parts: unsegmented part could be in
        the beginning and in the end of the video.
        Otherwise range includes all video frames.
        """
        first_seg, second_seg = (), ()
        if self._reliable:
            if self._frame_bounds[0] != 0:
                first_seg = (0, self._frame_bounds[0])
            if self._frame_bounds[-1] != self.__len__():
                second_seg = (self._frame_bounds[-1], self.__len__())
        else:
            first_seg = (0, self.__len__())
        return first_seg, second_seg

    def relative_segments(self):
        """For each frame define respective vector with 1 covers its segment

        Like:
        1 -> 11000
        2 -> 11000
        3 -> 00111
        4 -> 00111
        5 -> 00111
        """

        # define mask for each segment
        masks = {}
        start_idx = 0
        for bound_idx, start in enumerate(self._frame_bounds[:-1]):
            end = self._frame_bounds[bound_idx + 1]
            key = (end + start) / (2 * self.__len__())
            range_segm = 100
            value = np.zeros(range_segm)
            if end != self.__len__():
                segment_len = int(((end - start) / self.__len__()) * range_segm)
                value[start_idx: start_idx + segment_len] = 1
                start_idx += segment_len
            else:
                value[start_idx:] = 1
            masks[key] = value
        relative_bin_labels = np.zeros((self.__len__(), 100))
        for frame_idx, frame_time in enumerate(self.frame_labels):
            relative_bin_labels[frame_idx] = masks[frame_time]
        return relative_bin_labels

    ###########################################################################
    # additional functions for visualization statistics from pose estimation

    def plot(self, coords, mag=False, save=False):
        plots_f = 'plots_f'
        dir_check(os.path.join(opt.data, plots_f))

        joint_names = self._rattr + self._lattr
        colors = {}
        for label, _, _ in order[self._name]:
            colors[label] = (np.random.rand(), np.random.rand(), np.random.rand())

        # total_mag = np.zeros(len(coords[0]))

        add = mag * 2 + (self._frame_bounds is not None)

        fig = plt.figure(figsize=(15, 15))
        plt.subplots_adjust(top=0.9, hspace=0.6)
        for idx, coord in enumerate(coords):
            if idx % 2 == 0:
                ax = fig.add_subplot(len(coords) // 2 + add, 1, idx // 2 + 1)
                plt.title(joint_names[idx // 2])
                for label, start, end in order[self._name]:
                    ax.axvspan(start, end, facecolor=colors[label], alpha=0.3)
                ax.grid(True)
                # if mag:
                #     temp_coord = np.array(coord)
                #     temp_coord[np.argwhere(np.isnan(temp_coord))] = 0
                #     total_mag += temp_coord

            ax.plot(self._frames, coord, ['r', 'g'][idx % 2])

        title = ['x & y', 'magnitude and angle']
        fig.suptitle(title[mag], fontsize=20)
        fig.legend([['x', 'y'], ['mag', 'angle']][mag])

        if mag:
            self._compute_total(coords)
            ax = fig.add_subplot(len(coords) // 2 + add, 1, len(coords) // 2 + 1)
            plt.title('total magnitude')
            for label, start, end in order[self._name]:
                ax.axvspan(start, end, facecolor=colors[label], alpha=0.3)
            ax.grid(True)

            ax.plot(self._frames, self._total_mag)

            # try:
            #     b, a = signal.butter(3, 0.05)
            #     filtered_total = signal.filtfilt(b, a, self._total_mag)
            # except ValueError:
            #     pass
            # else:
            ax = fig.add_subplot(len(coords) // 2 + add, 1, len(coords) // 2 + 2)
            plt.title('total magnitude filtered')
            for label, start, end in order[self._name]:
                ax.axvspan(start, end, facecolor=colors[label], alpha=0.3)
            ax.grid(True)
            ax.plot(self._frames, self._filt_total_mag)

            if self._frame_bounds is not None:
                ax = fig.add_subplot(len(coords) // 2 + add, 1, len(coords) // 2 + 3)
                plt.title('new segmentation')
                start = self._frame_bounds[0]
                for end in self._frame_bounds[1:]:
                    clr = colors[np.random.choice(list(colors.keys()))]
                    ax.axvspan(start, end, facecolor=clr, alpha=0.3)
                    ax.axvline(x=start, color='k')
                    ax.axvline(x=end, color='k')
                ax.grid(True)
                ax.plot(self._frames, self._filt_total_mag)

        if not save:
            # plt.tight_layout()
            # plt.show()
            pass
        else:
            if mag:
                dir_check(os.path.join(opt.data, plots_f, 'mag_%s' % opt.subaction))
                fig.savefig(os.path.join(opt.data, plots_f, 'mag_%s' % opt.subaction,
                                         '%s.png' % self._name))
            else:
                dir_check(os.path.join(opt.data, plots_f, 'xy_%s' % opt.subaction))
                fig.savefig(os.path.join(opt.data, plots_f, 'xy_%s' % opt.subaction,
                                         '%s.png' % self._name))
        plt.close('all')

    def video_vis(self):
        """Video visualisation

        Video visualization framewise with corrsponding notation about
        subaction, frame number and joint positions
        """
        video_path = os.path.join(opt.data, 'ascii',
                                  '%s.%s' % (self._name, opt.end))
        cap = cv2.VideoCapture(video_path)
        try:
            assert cap.isOpened()
        except AssertionError:
            raise AssertionError('Cannot open video %s: ' % video_path)
        # text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner_of_text = (10, 20)
        font_scale = 0.5
        font_color = (255, 255, 255)
        line_type = 1

        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        gt_count = len(ground_truth[self._name])
        frame_n = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        output_path = os.path.join('/media/data/kukleva/lab/presentation', self._name + '.avi')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (320, 240))
        while ret and frame_n < gt_count:
            # put text
            subaction = index2label[ground_truth[self._name][frame_n]]
            cv2.putText(frame,
                        'frame # %d %s %d/%d' %
                        (frame_n, subaction, gt_count, frame_count),
                        bottom_left_corner_of_text,
                        font,
                        font_scale,
                        font_color,
                        line_type)
            # visualize joints
            if np.sum(self._poses[:, frame_n]):
                vis_frame_n = frame_n
            else:
                vis_frame_n = frame_n - 1

            for joint_name in self._lattr + self._rattr:
                joint_idx = pose2idx[joint_name]
                try:
                    x = int(self._poses[joint_idx, vis_frame_n])
                    y = int(self._poses[15 + joint_idx, vis_frame_n])
                except ValueError:
                    continue
                cv2.circle(frame, (x, y), 3, [0, 0, 255], thickness=3)

            out.write(frame)

            cv2.imshow('Video', frame)
            frame_n = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.waitKey(1)
            # time.sleep(0.5)
            ret, frame = cap.read()

        out.release()
        cap.release()
        cv2.destroyAllWindows()


def save_plots():
    for filename in os.listdir(opt.pose_path):
        if opt.subaction not in filename:
            continue
        logger.debug('%s' % filename)
        path = os.path.join(opt.pose_path, filename)
        f_der = PoseVideoInstance(path)
        f_der.joint_der(mag_use=False, save=True, visual=True)
        f_der.joint_der(mag_use=True, save=True, visual=True)


if __name__ == '__main__':
    # test_path = '/media/data/kukleva/lab/Breakfast/videos/video_poses/P03_cam01_P03_salat.npy'

    test_path = '/media/data/kukleva/lab/Breakfast/videos/video_poses/P23_cam01_P23_coffee.npy'  # useless poses
    # test_path = '/media/data/kukleva/lab/Breakfast/videos/video_poses/P25_cam01_P25_coffee.npy'  # okeyish
    # test_path = '/media/data/kukleva/lab/Breakfast/videos/video_poses/P17_cam01_P17_coffee.npy'  # okeyish
    test_path = '/media/data/kukleva/lab/Breakfast/videos/video_poses/P15_webcam01_P15_coffee.npy'  # okeyish
    test_path = '/media/data/kukleva/lab/Breakfast/videos/video_poses/P17_webcam02_P17_coffee.npy'  # okeyish
    inst = PoseVideoInstance(test_path)
    inst.joint_der(mag_use=True, visual=True)

    inst.video_vis()
    # inst.joint_der(mag_vis=False)

    # save_plots()
