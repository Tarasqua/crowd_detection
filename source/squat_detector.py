import asyncio
import os
import itertools
from pathlib import Path

import cv2
import numpy as np

from misc import PlotPoseData, SquatsDetectorData, MainConfigurationsData, PlotData


class SquatDetector:
    """Детектор приседаний"""
    def __init__(self, show_angles: bool = False):
        self.show_angles = show_angles
        self.detector_data = SquatsDetectorData()
        self.plot_pose_data = PlotPoseData()
        self.plot_data = PlotData()
        self.main_config_data = MainConfigurationsData()

        self.frame = None
        self.people_flags, self.trigger = {}, False

    @staticmethod
    def get_angle_degrees(p1: np.array, p2: np.array, p3: np.array) -> float | None:
        """Возвращает угол между тремя точками на плоскости в градусах"""
        if np.array([None, None]) in np.array([p1, p2, p3]):
            return 0
        v1, v2 = p1 - p2, p3 - p2
        return float(np.degrees(np.arccos(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        )))

    def get_legs_angles(self, detection) -> np.array:
        """Возвращает углы в порядке: левое колено, левое бедро, правое бедро, правое колено"""
        key_points = [kp[:-1] if kp[-1] > self.main_config_data.key_points_confidence else np.array([None, None])
                      for kp in detection.keypoints.cpu().data.numpy()[0]]  # ключевые точки с порогом
        left_knee = self.get_angle_degrees(key_points[11], key_points[13], key_points[15])  # левое колено
        right_knee = self.get_angle_degrees(key_points[12], key_points[14], key_points[16])  # правое колено
        left_hip = self.get_angle_degrees(key_points[5], key_points[11], key_points[13])  # левое бедро
        right_hip = self.get_angle_degrees(key_points[6], key_points[12], key_points[14])  # правое бедро
        # отображение углов в локтях и плечах
        if self.show_angles:
            if left_knee != 0 and left_knee is not None:
                cv2.putText(self.frame, str(int(np.round(left_knee))), tuple(key_points[7].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, 3)
            if right_knee != 0 and right_knee is not None:
                cv2.putText(self.frame, str(int(np.round(right_knee))), tuple(key_points[8].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, 3)
            if left_hip != 0 and left_hip is not None:
                cv2.putText(self.frame, str(int(np.round(left_hip))), tuple(key_points[8].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, 3)
            if right_hip != 0 and right_hip is not None:
                cv2.putText(self.frame, str(int(np.round(right_hip))), tuple(key_points[8].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, 3)
        return np.array([left_knee, left_hip, right_hip, right_knee])

    def people_ids_update(self, people_ids) -> None:
        """Добавляет новых людей в список и удаляет тех, кого не обнаружили на новом кадре"""
        for person_id in people_ids:
            if person_id not in self.people_flags:
                self.people_flags[person_id] = False
        self.people_flags = {person_id: raised_flag for person_id, raised_flag
                             in self.people_flags.items() if person_id in people_ids}

    def get_squatted_ids(self) -> dict.keys:
        """Возвращает id людей, поднявших руки"""
        return {person_id: squatted_flag for person_id, squatted_flag in self.people_flags.items()
                if squatted_flag}.keys()

    def check_angles(self, angles, human_id) -> None:
        """Проверка на то, что человек присел"""
        if all(angles <= self.detector_data.angles_threshold) and all(angles != 0):
            self.people_flags[human_id] = True
        else:
            self.people_flags[human_id] = False

    async def plot_bbox(self, detection) -> None:
        """Отрисовка bbox'а"""
        x1, y1, x2, y2 = detection.boxes.xyxy.data.numpy()[0]
        if detection.boxes.id is None:
            return
        human_id = detection.boxes.id.cpu().numpy()[0].astype(int)
        if self.people_flags[human_id]:
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

    async def detect_(self, detections):
        """Обработка YOLO-детекций детектором приседаний"""
        self.frame = detections.orig_img
        people_ids = np.array([detection.boxes.id.cpu().numpy()[0].astype(int) for detection in detections
                               if detection.boxes.id is not None])
        self.people_ids_update(people_ids)
        # чтобы отследить новых людей, которые присели
        squatted_ids_before = self.get_squatted_ids()
        if len(detections) != 0:
            for detection, human_id in zip(detections, people_ids):
                angles = self.get_legs_angles(detection)
                self.check_angles(angles, human_id)
            await self.plot_bboxes_poses(detections)
        squatted_ids_after = self.get_squatted_ids()
        if len(squatted_ids_before) <= len(squatted_ids_after) and \
                squatted_ids_before != squatted_ids_after:
            return self.frame, True
        else:
            return self.frame, False
