"""
 * Pose
 * Created on 5/29/18
 * Author: doering
"""


class Pose(object):
    """This class represents a single pose in an image."""
    def __init__(self, joints=None, score=None, feature=None, id=None,
                 subset_score=None):
        self.joints = joints
        self.score = score
        self.feature = feature
        self.id = id
        self.subset_score = subset_score


class PoseExtended(Pose):
    """Include frame number where pose was detected"""
    def __init__(self, pose, frame):
        super().__init__(**pose.__dict__)
        self.frame = frame


pose2idx = {}
idx2pose = {}


def pose_mapping():
    joint_names = ['RAnkle', 'RKnee', 'RHip', 'LHip', 'LKnee', 'LAnkle', 'RWrist', 'RElbow', 'RShoulder', 'LShoulder',
                   'LElbow', 'LWrist', 'Neck', 'Nose', 'Head_Top']
    for idx, joint_name in enumerate(joint_names):
        pose2idx[joint_name] = idx
        idx2pose[idx] = joint_name


pose_mapping()

