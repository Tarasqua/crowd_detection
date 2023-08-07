import os
from pathlib import Path
import datetime
import asyncio

import cv2
from ultralytics import YOLO
import numpy as np


class IdleDetector:

    def __init__(self, stream_source: str | int, yolo_model: str = 'n'):
        self.stream_source = stream_source
        models_path = os.path.join(Path(__file__).resolve().parents[1], 'models')
        self.yolo_detector = YOLO(os.path.join(models_path, 'yolo_models', f'yolov8{yolo_model}-pose.onnx'))
        self.frame = None

        self.yolo_conf = 0.2
        self.key_points_conf = 0.5
        self.data_accumulation_counter = 100

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
        key_points = [kp[:-1] if kp[-1] > self.key_points_conf else np.array([None, None])  # ключевые точки с порогом
                      for kp in detection.keypoints.cpu().data.numpy()[0]]

        left_elbow = self.get_angle_degrees(key_points[5], key_points[7], key_points[9])  # левый локоть
        if left_elbow != 0:
            cv2.putText(self.frame, str(np.round(left_elbow)), tuple(key_points[7].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, 3)

        right_elbow = self.get_angle_degrees(key_points[6], key_points[8], key_points[10])  # правый локоть
        if right_elbow != 0:
            cv2.putText(self.frame, str(np.round(right_elbow)), tuple(key_points[8].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, 3)

        left_shoulder = self.get_angle_degrees(key_points[11], key_points[5], key_points[7])  # левое плечо
        if left_shoulder != 0:
            cv2.putText(self.frame, str(np.round(left_shoulder)), tuple(key_points[5].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, 3)

        right_shoulder = self.get_angle_degrees(key_points[12], key_points[6], key_points[8])  # правое плечо
        if right_shoulder != 0:
            cv2.putText(self.frame, str(np.round(right_shoulder)), tuple(key_points[6].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, 3)

        return np.array([left_elbow, left_shoulder, right_shoulder, right_elbow])

    @staticmethod
    def get_ema_deviations(frame_counter, previous_calculations: np.array, new_angles: np.array) -> tuple:
        """Возвращает exponential moving average (EMA) и дисперсию"""
        alpha = 2 / (frame_counter + 1)
        emas = np.array([alpha * new_angles[i] + (1 - alpha) * angle  # EMAt от t-1
                         for i, angle in enumerate(previous_calculations[:4])])
        deviations = np.array([alpha * np.square(angle - ema) + (1 - alpha) * deviation  # Dt от t-1
                               for ema, angle, deviation in
                               zip(emas, previous_calculations[:4], previous_calculations[4:-1])])
        return emas, deviations

    def get_angles_statistics(self, detection, human_id: int, people_angles_statistics: dict):
        """Возвращает статистические данные по углам в локтях и плечах - ema, дисперсию"""
        hands_angles = self.get_hands_angles(detection)
        # первые 4 - EMA, вторые - дисперсия, крайний - какой кадр
        if human_id not in people_angles_statistics.keys():
            people_angles_statistics[human_id] = np.append(hands_angles, [0, 0, 0, 0, 1])
        # elif people_angle_statistics[human_id][-1] < self.data_accumulation_counter:
        else:
            frames_counter = people_angles_statistics[human_id][-1] + 1
            emas, deviations = self.get_ema_deviations(
                frames_counter, people_angles_statistics[human_id], hands_angles)
            people_angles_statistics[human_id] = np.concatenate(np.array([emas, deviations, [frames_counter]]))
        return people_angles_statistics

    def plot_bboxes(self, detection, violate: list) -> None:
        """Отрисовка bbox'а и центроида человека"""
        x1, y1, x2, y2 = detection.boxes.xyxy.data.numpy()[0]
        color = (0, 255, 0)
        if detection.boxes.id.cpu().numpy()[0].astype(int) in violate:
            color = (0, 0, 255)
        cv2.rectangle(self.frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    async def detect_(self) -> None:
        """Детекция большого скопления людей в кадре"""
        cap = cv2.VideoCapture(self.stream_source)
        people_angles_statistics = {}
        while cap.isOpened():
            start = datetime.datetime.now()
            ret, self.frame = cap.read()
            if not ret:
                break
            detections = self.yolo_detector.track(
                self.frame, classes=[0], stream=False, conf=self.yolo_conf, verbose=False)
            if len(detections[0]) != 0:
                for detection in detections[0]:
                    human_id = detection.boxes.id.cpu().numpy()[0].astype(int)
                    people_angles_statistics = self.get_angles_statistics(
                        detection, human_id, people_angles_statistics)
            fps = (1 / (datetime.datetime.now() - start).microseconds) / 10 ** -6
            cv2.putText(self.frame, f'fps: {str(int(np.round(fps)))}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, 3)
            if self.frame is not None:
                cv2.imshow('main', self.frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    idle = IdleDetector(0)
    asyncio.run(idle.detect_())
