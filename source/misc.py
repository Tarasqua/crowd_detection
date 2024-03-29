import os
import yaml
from pathlib import Path

import numpy as np


class ConfigNotFoundException(Exception):
    """Ошибка, возникающая при отсутствии config файла."""

    def __str__(self):
        return "\nCouldn't find config file"


class UnknownSceneSizeException(Exception):
    """Ошибка, возникающая при некорректном вводе размера сцены."""

    def __str__(self):
        return "\nThere is no such scene size"


class PlotPoseData:

    def __init__(self):
        key_points = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                      'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                      'right_knee', 'left_ankle', 'right_ankle']
        self.limbs = [
            [key_points.index('right_eye'), key_points.index('nose')],
            [key_points.index('right_eye'), key_points.index('right_ear')],
            [key_points.index('left_eye'), key_points.index('nose')],
            [key_points.index('left_eye'), key_points.index('left_ear')],
            [key_points.index('right_shoulder'), key_points.index('right_elbow')],
            [key_points.index('right_elbow'), key_points.index('right_wrist')],
            [key_points.index('left_shoulder'), key_points.index('left_elbow')],
            [key_points.index('left_elbow'), key_points.index('left_wrist')],
            [key_points.index('right_hip'), key_points.index('right_knee')],
            [key_points.index('right_knee'), key_points.index('right_ankle')],
            [key_points.index('left_hip'), key_points.index('left_knee')],
            [key_points.index('left_knee'), key_points.index('left_ankle')],
            [key_points.index('right_shoulder'), key_points.index('left_shoulder')],
            [key_points.index('right_hip'), key_points.index('left_hip')],
            [key_points.index('right_shoulder'), key_points.index('right_hip')],
            [key_points.index('left_shoulder'), key_points.index('left_hip')]
        ]
        palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                            [255, 255, 255]])
        self.pose_limb_color = palette[[16, 16, 16, 16, 9, 9, 9, 9, 0, 0, 0, 0, 7, 7, 7, 7]]
        self.pose_kpt_color = palette[[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]]

    def __getattr__(self, item):
        return self.limbs, self.pose_limb_color, self.pose_kpt_color


class MainConfigurationsData:
    """Данные из config по MAIN_CONFIGURATIONS_DATA"""
    def __init__(self):
        config_path = os.path.join(Path(__file__).resolve().parents[1], 'config', 'config.yaml')
        if not os.path.exists(config_path):
            raise ConfigNotFoundException
        with open(config_path, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.yolo_confidence: float = self.config['MAIN_CONFIGURATIONS_DATA']['YOLO_CONFIDENCE']
        self.bbox_confidence: float = self.config['MAIN_CONFIGURATIONS_DATA']['BBOX_CONFIDENCE']
        self.key_points_confidence: float = self.config['MAIN_CONFIGURATIONS_DATA']['KEY_POINTS_CONFIDENCE']

    def __getattr__(self, item):
        return self.yolo_confidence, self.bbox_confidence, self.key_points_confidence


class CrowdDetectorData:
    """Данные из config по CROWD_DETECTOR_DATA"""
    def __init__(self):
        config_path = os.path.join(Path(__file__).resolve().parents[1], 'config', 'config.yaml')
        if not os.path.exists(config_path):
            raise ConfigNotFoundException
        with open(config_path, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.min_distance: float = self.config['CROWD_DETECTOR_DATA']['MIN_DISTANCE']
        self.min_crowd_num_of_people: int = self.config['CROWD_DETECTOR_DATA']['MIN_CROWD_NUM_OF_PEOPLE']
        self.max_crowd_num_of_people: int | None = self.config['CROWD_DETECTOR_DATA'][
            'MAX_CROWD_NUM_OF_PEOPLE']
        self.kmeans_n_clusters: int = self.select_n_clusters()
        self.iou_threshold: float = self.config['CROWD_DETECTOR_DATA']['IOU_THRESHOLD']
        self.iou_crowd_threshold: float = self.config['CROWD_DETECTOR_DATA']['IOU_CROWD_THRESHOLD']

    def select_n_clusters(self):
        match self.config['CROWD_DETECTOR_DATA']['SCENE_SIZE']:
            case 'small':
                return 2
            case 'medium':
                return 5
            case 'large':
                return 10
            case _:
                raise UnknownSceneSizeException

    def __getattr__(self, item):
        return self.min_distance, self.min_crowd_num_of_people, self.max_crowd_num_of_people, \
               self.kmeans_n_clusters, self.iou_threshold, self.iou_crowd_threshold


class ActiveGesturesDetectorData:
    """Данные из config по ACTIVE_GESTURES_DETECTOR_DATA"""
    def __init__(self):
        config_path = os.path.join(Path(__file__).resolve().parents[1], 'config', 'config.yaml')
        if not os.path.exists(config_path):
            raise ConfigNotFoundException
        with open(config_path, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.delta_angle_threshold: float | int = self.config['ACTIVE_GESTURES_DETECTOR_DATA']['DELTA_ANGLE_THRESHOLD']
        self.max_active_gestures: int = self.config['ACTIVE_GESTURES_DETECTOR_DATA']['MAX_ACTIVE_GESTURES']

    def __getattr__(self, item):
        return self.delta_angle_threshold, self.max_active_gestures


class RaisedHandsDetectorData:
    """Данные из config по RAISED_HANDS_DETECTOR_DATA"""
    def __init__(self):
        config_path = os.path.join(Path(__file__).resolve().parents[1], 'config', 'config.yaml')
        if not os.path.exists(config_path):
            raise ConfigNotFoundException
        with open(config_path, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.shoulders_angle_threshold: int = self.config['RAISED_HANDS_DETECTOR_DATA']['SHOULDERS_ANGLE_THRESHOLD']
        self.shoulders_bent_angle_threshold: int = self.config['RAISED_HANDS_DETECTOR_DATA'][
            'SHOULDERS_BENT_ANGLE_THRESHOLD']
        self.elbow_bent_angle_threshold: int = self.config['RAISED_HANDS_DETECTOR_DATA']['ELBOW_BENT_ANGLE_THRESHOLD']

    def __getattr__(self, item):
        return self.shoulders_angle_threshold, self.shoulders_bent_angle_threshold, self.elbow_bent_angle_threshold


class SquatsDetectorData:
    """Данные из config по SQUAT_DETECTOR_DATA"""
    def __init__(self):
        config_path = os.path.join(Path(__file__).resolve().parents[1], 'config', 'config.yaml')
        if not os.path.exists(config_path):
            raise ConfigNotFoundException
        with open(config_path, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.angles_threshold: int = self.config['SQUAT_DETECTOR_DATA']['ANGLES_THRESHOLD']

    def __getattr__(self, item):
        return self.angles_threshold


class PlotData:
    """Данные из config по PLOT_DATA"""
    def __init__(self):
        config_path = os.path.join(Path(__file__).resolve().parents[1], 'config', 'config.yaml')
        if not os.path.exists(config_path):
            raise ConfigNotFoundException
        with open(config_path, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.main_color: tuple = tuple(self.config['PLOT_DATA']['MAIN_COLOR'])
        self.additional_color: tuple = tuple(self.config['PLOT_DATA']['ADDITIONAL_COLOR'])
        self.additional_color_visibility: float = self.config['PLOT_DATA']['ADDITIONAL_COLOR_VISIBILITY']

    def __getattr__(self, item):
        return self.main_color, self.additional_color, self.additional_color_visibility


if __name__ == '__main__':
    c = CrowdDetectorData()
    print('a')
