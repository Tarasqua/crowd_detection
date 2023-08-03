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


class CrowdDetectorData:

    def __init__(self):
        config_path = os.path.join(Path(__file__).resolve().parents[1], 'config', 'config.yaml')
        if not os.path.exists(config_path):
            raise ConfigNotFoundException
        with open(config_path, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.min_distance: float = self.config['CROWD_DETECTOR_DATA']['MAIN_DATA']['MIN_DISTANCE']
        self.yolo_confidence: float = self.config['CROWD_DETECTOR_DATA']['MAIN_DATA']['YOLO_CONFIDENCE']
        self.bbox_confidence: float = self.config['CROWD_DETECTOR_DATA']['MAIN_DATA']['BBOX_CONFIDENCE']
        self.key_points_confidence: float = self.config['CROWD_DETECTOR_DATA']['MAIN_DATA']['KEY_POINTS_CONFIDENCE']
        self.min_crowd_num_of_people: int = self.config['CROWD_DETECTOR_DATA']['MAIN_DATA']['MIN_CROWD_NUM_OF_PEOPLE']
        self.max_crowd_num_of_people: int | None = self.config['CROWD_DETECTOR_DATA']['MAIN_DATA'][
            'MAX_CROWD_NUM_OF_PEOPLE']
        self.kmeans_n_clusters: int = self.select_n_clusters()
        self.main_color: tuple = tuple(self.config['CROWD_DETECTOR_DATA']['PLOT_DATA']['MAIN_COLOR'])
        self.additional_color: tuple = tuple(self.config['CROWD_DETECTOR_DATA']['PLOT_DATA']['ADDITIONAL_COLOR'])
        self.additional_color_visibility: float = self.config['CROWD_DETECTOR_DATA']['PLOT_DATA'][
            'ADDITIONAL_COLOR_VISIBILITY']

    def select_n_clusters(self):
        match self.config['CROWD_DETECTOR_DATA']['MAIN_DATA']['SCENE_SIZE']:
            case 'small':
                return 2
            case 'medium':
                return 5
            case 'large':
                return 10
            case _:
                raise UnknownSceneSizeException

    def __getattr__(self, item):
        return self.min_distance, self.yolo_confidence, self.bbox_confidence, self.key_points_confidence, \
               self.min_crowd_num_of_people, self.max_crowd_num_of_people, self.kmeans_n_clusters, \
               self.main_color, self.additional_color, self.additional_color_visibility


if __name__ == '__main__':
    c = CrowdDetectorData()
    print('a')
