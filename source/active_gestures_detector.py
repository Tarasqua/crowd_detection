import os
from pathlib import Path
import datetime
import asyncio

import cv2
from ultralytics import YOLO
import numpy as np

from misk import PlotPoseData, PlotData, ActiveGesturesDetectorData, MainConfigurationsData


class ActiveGesturesDetector:

    def __init__(self, stream_source: str | int, yolo_model: str = 'n',
                 show_angles: bool = False, save_record: bool = False):
        self.stream_source = stream_source
        self.show_angles = show_angles
        self.save_record = save_record
        models_path = os.path.join(Path(__file__).resolve().parents[1], 'models')
        self.yolo_detector = YOLO(os.path.join(models_path, 'yolo_models', f'yolov8{yolo_model}-pose.onnx'))
        self.detector_data = ActiveGesturesDetectorData()
        self.plot_pose_data = PlotPoseData()
        self.plot_data = PlotData()
        self.main_config_data = MainConfigurationsData()

        self.frame = None
        self.people_angles = {}

    @staticmethod
    def get_video_writer(cap):
        records_folder_path = os.path.join(Path(__file__).resolve().parents[1], 'records')
        active_gestures_folder = os.path.join(records_folder_path, 'active_gestures_records')
        if not os.path.exists(records_folder_path):
            os.mkdir(records_folder_path)
        if not os.path.exists(active_gestures_folder):
            os.mkdir(active_gestures_folder)
        out = cv2.VideoWriter(
            os.path.join(active_gestures_folder, f'active_gestures_{len(os.listdir(active_gestures_folder)) + 1}.mp4'),
            -1, 20.0, (int(cap.get(3)), int(cap.get(4))))
        return out

    @staticmethod
    def get_angle_degrees(p1: np.array, p2: np.array, p3: np.array) -> float | None:
        """Возвращает угол между тремя точками на плоскости в градусах"""
        if np.array([None, None]) in np.array([p1, p2, p3]):
            return 0
        v1, v2 = p1 - p2, p3 - p2
        return float(np.degrees(np.arccos(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        )))

    def get_hands_angles(self, detection) -> np.array:
        """Возвращает углы в порядке: левый локоть, левое плечо, правое плечо, правый локоть"""
        key_points = [kp[:-1] if kp[-1] > self.main_config_data.key_points_confidence else np.array([None, None])
                      for kp in detection.keypoints.cpu().data.numpy()[0]]  # ключевые точки с порогом
        left_elbow = self.get_angle_degrees(key_points[5], key_points[7], key_points[9])  # левый локоть
        right_elbow = self.get_angle_degrees(key_points[6], key_points[8], key_points[10])  # правый локоть
        left_shoulder = self.get_angle_degrees(key_points[11], key_points[5], key_points[7])  # левое плечо
        right_shoulder = self.get_angle_degrees(key_points[12], key_points[6], key_points[8])  # правое плечо
        # отображение углов в локтях и плечах
        if self.show_angles:
            if left_elbow != 0 and left_elbow is not None:
                cv2.putText(self.frame, str(int(np.round(left_elbow))), tuple(key_points[7].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, 3)
            if right_elbow != 0 and right_elbow is not None:
                cv2.putText(self.frame, str(int(np.round(right_elbow))), tuple(key_points[8].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, 3)
            if left_shoulder != 0 and left_shoulder is not None:
                cv2.putText(self.frame, str(int(np.round(left_shoulder))), tuple(key_points[5].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, 3)
            if right_shoulder != 0 and right_shoulder is not None:
                cv2.putText(self.frame, str(int(np.round(right_shoulder))), tuple(key_points[6].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, 3)
        return np.array([left_elbow, left_shoulder, right_shoulder, right_elbow])

    def check_angles(self, angles, human_id: int) -> None:
        """Подсчет резкого изменения средних углов левой и правой руки.
        Считается дельта между углами текущего и предыдущего кадра и, если дельта больше порогового, то
        количество активных действий увеличивается"""
        if len(self.people_angles[human_id]) == 0:
            # среднее по левой, правой руке; количество активных жестов; время слежения в кадрах
            self.people_angles[human_id] = np.array([np.mean(angles[2:]), np.mean(angles[:2]), 0, 1])
        else:
            mean_left, mean_right, active_gestures_counter, observation_time_frames = self.people_angles[human_id]
            if observation_time_frames >= self.detector_data.max_active_gestures * 12:  # сбрасываем счетчик
                self.people_angles[human_id] = np.array([])
                return
            new_mean_left, new_mean_right = np.mean(angles[2:]), np.mean(angles[:2])
            if np.abs(mean_left - new_mean_left) > self.detector_data.delta_angle_threshold \
                    or np.abs(mean_right - new_mean_right) > self.detector_data.delta_angle_threshold:
                active_gestures_counter += 1
            self.people_angles[human_id] = np.array(
                [new_mean_left, new_mean_right, active_gestures_counter, observation_time_frames + 1])

    def people_ids_update(self, people_ids) -> None:
        """Добавляет новых людей в список и удаляет тех, кого не обнаружили на новом кадре"""
        for person_id in people_ids:
            if person_id not in self.people_angles:
                self.people_angles[person_id] = np.array([])
        self.people_angles = {person_id: angles_data for person_id, angles_data
                              in self.people_angles.items() if person_id in people_ids}

    async def plot_bbox(self, detection) -> None:
        """Отрисовка bbox'а и центроида человека"""
        x1, y1, x2, y2 = detection.boxes.xyxy.data.numpy()[0]
        if detection.boxes.id is None:
            return
        human_id = detection.boxes.id.cpu().numpy()[0].astype(int)
        if len(self.people_angles[human_id]) != 0 and self.people_angles[human_id][-2] \
                >= self.detector_data.max_active_gestures:
            cv2.rectangle(self.frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          self.plot_data.additional_color, 2)
        else:
            cv2.rectangle(self.frame, (int(x1), int(y1)), (int(x2), int(y2)), self.plot_data.main_color, 2)

    async def plot_skeleton_kpts(self, kpts, limbs, pose_limb_color, pose_kpt_color) -> None:
        """Строит ключевые точки и суставы скелета человека"""
        for p_id, point in enumerate(kpts):
            x_coord, y_coord, conf = point
            if conf < self.main_config_data.key_points_confidence:
                continue
            r, g, b = pose_kpt_color[p_id]
            cv2.circle(self.frame, (int(x_coord), int(y_coord)), 5, (int(r), int(g), int(b)), -1)
        for sk_id, sk in enumerate(limbs):
            r, g, b = pose_limb_color[sk_id]
            if kpts[sk[0]][2] < self.main_config_data.key_points_confidence or \
                    kpts[sk[1]][2] < self.main_config_data.key_points_confidence:
                continue
            pos1 = int(kpts[sk[0]][0]), int(kpts[sk[0]][1])
            pos2 = int(kpts[sk[1]][0]), int(kpts[sk[1]][1])
            cv2.line(self.frame, pos1, pos2, (int(r), int(g), int(b)), thickness=2)

    async def plot_bboxes_poses(self, detections: list) -> None:
        """Отрисовка bbox'ов и скелетов"""
        skeleton_tasks, bboxes_tasks = [], []
        for detection in detections:
            if detection.boxes.conf < self.main_config_data.bbox_confidence:
                continue
            bboxes_tasks.append(asyncio.create_task(self.plot_bbox(detection)))
            skeleton_tasks.append(asyncio.create_task(
                self.plot_skeleton_kpts(detection.keypoints.data.numpy()[0], self.plot_pose_data.limbs,
                                        self.plot_pose_data.pose_limb_color, self.plot_pose_data.pose_kpt_color)
            ))
        for bbox_task in bboxes_tasks:
            await bbox_task
        for skeleton_task in skeleton_tasks:
            await skeleton_task

    async def detect_(self):
        if self.save_record:
            cap = cv2.VideoCapture(self.stream_source)
            out = self.get_video_writer(cap)
        for detections in self.yolo_detector.track(
                self.stream_source, classes=[0], stream=True,
                conf=self.main_config_data.yolo_confidence, verbose=False):
            self.frame = detections.orig_img
            people_ids = np.array([detection.boxes.id.cpu().numpy()[0].astype(int) for detection in detections
                                   if detection.boxes.id is not None])
            self.people_ids_update(people_ids)
            if len(detections) != 0:
                for detection, human_id in zip(detections, people_ids):
                    angles = self.get_hands_angles(detection)
                    if any(angles > 0):
                        self.check_angles(angles, human_id)
                await self.plot_bboxes_poses(detections)
            if self.frame is not None:
                if self.save_record:
                    out.write(self.frame)
                cv2.imshow('main', self.frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        if self.save_record:
            out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # active_gestures = ActiveGesturesDetector('../demos/nervous/4.mp4', show_angles=True)
    # active_gestures = ActiveGesturesDetector('../demos/nervous/3.mp4', show_angles=False, save_record=False)
    active_gestures = ActiveGesturesDetector(0, show_angles=True)
    asyncio.run(active_gestures.detect_())
