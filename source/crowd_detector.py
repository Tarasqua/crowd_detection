import asyncio
import datetime
import os
import itertools
from pathlib import Path

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance as dist
from sklearn.cluster import KMeans

from misk import PlotPoseData, CrowdDetectorData


class CrowdDetector:

    def __init__(self, stream_source: str | int, yolo_model: str = 'n', save_record: bool = False):
        self.stream_source = stream_source
        self.save_record = save_record
        self.detector_data = CrowdDetectorData()
        self.plot_pose_data = PlotPoseData()
        self.yolo_detector = YOLO(os.path.join('yolo_models', f'yolov8{yolo_model}-pose.onnx'))
        self.kmeans_model = KMeans(n_clusters=self.detector_data.kmeans_n_clusters, init='k-means++',
                                   max_iter=300, n_init=10, random_state=42)
        self.frame = None

    def kmeans_fit(self, data: np.ndarray):
        """Предобучение KMeans модели"""
        self.kmeans_model.fit(data)

    @staticmethod
    def get_video_writer(cap):
        folder_path = os.path.join(Path(__file__).resolve().parents[1], 'records')
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        out = cv2.VideoWriter(os.path.join(folder_path, f'record_{len(os.listdir(folder_path)) + 1}.mp4'),
                              -1, 20.0, (int(cap.get(3)), int(cap.get(4))))
        return out

    async def plot_skeleton_kpts(self, kpts, limbs, pose_limb_color, pose_kpt_color) -> None:
        """Строит ключевые точки и суставы скелета человека"""
        for p_id, point in enumerate(kpts):
            x_coord, y_coord, conf = point
            if conf < self.detector_data.key_points_confidence:
                continue
            r, g, b = pose_kpt_color[p_id]
            cv2.circle(self.frame, (int(x_coord), int(y_coord)), 5, (int(r), int(g), int(b)), -1)
        for sk_id, sk in enumerate(limbs):
            r, g, b = pose_limb_color[sk_id]
            if kpts[sk[0]][2] < self.detector_data.key_points_confidence or \
                    kpts[sk[1]][2] < self.detector_data.key_points_confidence:
                continue
            pos1 = int(kpts[sk[0]][0]), int(kpts[sk[0]][1])
            pos2 = int(kpts[sk[1]][0]), int(kpts[sk[1]][1])
            cv2.line(self.frame, pos1, pos2, (int(r), int(g), int(b)), thickness=2)

    def get_centroids(self, detections) -> np.array:
        """Возвращает центроиды bbox'ов в относительных координатах"""
        xyxy = [d.boxes.xyxy.data.numpy()[0] for d in detections[0]
                if float(d.boxes.conf.data) > self.detector_data.bbox_confidence]
        width, height = self.frame.shape[:-1][::-1]
        return np.array([((x1 / width + x2 / width) / 2, (y1 / height + y2 / height) / 2) for x1, y1, x2, y2 in xyxy])

    def get_kpts_centroid(self, detection) -> np.array:
        """
        Возвращает центроиды по ключевым точкам, где:
        5 - левое плечо, 6 - правое плечо, 11 - левое бедро, 12 - правое бедро.
        Если видны все 4 точки, то вычисляется центроид относительно них;
        если видны только верхние две, то центроидом считается середина отрезка между плечами;
        если ни одни из точек не видна, то центроид находится, исходя из bbox'а
        """
        kpts = detection.keypoints.data.numpy()[0]
        points = np.concatenate(  # координаты точек
            [[kpts[:, :-1][5]], [kpts[:, :-1][6]], [kpts[:, :-1][11]], [kpts[:, :-1][12]]]
        ) / self.frame.shape[:-1][::-1]  # в относительные координаты
        confs = np.array([kpts[:, -1][5], kpts[:, -1][6], kpts[:, -1][11], kpts[:, -1][12]])  # их confidence
        if all(confs > self.detector_data.key_points_confidence):  # если видно все тело
            return points.sum(axis=0) / len(points)
        elif any(confs[-2:] < self.detector_data.key_points_confidence) and \
                all(confs[:-2] > self.detector_data.key_points_confidence):  # видно только плечи
            return points[:-2].sum(axis=0) / 2
        else:  # не видно тела
            xyxy = np.array(
                [detection.boxes.xyxy.data.numpy()[0][:-2], detection.boxes.xyxy.data.numpy()[0][-2:]]
            ) / self.frame.shape[:-1][::-1]  # в относительные координаты
            return xyxy.sum(axis=0) / 2

    def get_grouped_ids(self, group_id: dict, detections) -> dict:
        """Возвращает группы с id, расстояние между которыми меньше порогового"""
        # centroids = self.get_centroids(detections)
        centroids = np.array(
            [self.get_kpts_centroid(detection) for detection in detections[0]]
        ).astype('float64')
        if centroids.size == 0:
            return group_id
        # формируем словарь с id: группа, исходя из разделения по группам kmeans
        id_group = {person_id: group for person_id, group in enumerate(self.kmeans_model.predict(centroids))}
        # раскидываем id по группам
        for k, v in id_group.items():
            group_id.setdefault(v, []).append(k)
        # делаем условие на минимальное количество человек в группе
        if self.detector_data.max_crowd_num_of_people is not None:
            group_id = {
                g: i for g, i in group_id.items()
                if self.detector_data.min_crowd_num_of_people <= len(i) <= self.detector_data.max_crowd_num_of_people}
        else:
            group_id = {g: i for g, i in group_id.items() if len(i) >= self.detector_data.min_crowd_num_of_people}
        if group_id:
            return self.get_violate_group_id(centroids, group_id)
        else:
            return group_id

    def get_violate_group_id(self, centroids, group_id):
        """Возвращает группы с id, расстояния между которыми меньше порогового"""
        # расстояния каждый с каждым
        distances = dist.cdist(centroids, centroids, metric='euclidean')
        violate_group_id = {g: [] for g in group_id.keys()}
        # смотрим расстояния между центроидами
        for group, person_ids in group_id.items():
            for id1, id2 in itertools.combinations(person_ids, 2):  # каждый с каждым
                if distances[id1, id2] < self.detector_data.min_distance:
                    violate_group_id[group].append([id1, id2])
            if violate_group_id[group]:
                violate_group_id[group] = np.concatenate(violate_group_id[group])
        violate_group_id = {g: ids for g, ids in violate_group_id.items() if ids != []}  # убираем пустые
        return violate_group_id

    async def plot_bboxes(self, detection):
        """Отрисовка bbox'а и центроида человека"""
        x1, y1, x2, y2 = detection.boxes.xyxy.data.numpy()[0]
        # centroid = int((x1 + x2) / 2), int((y1 + y2) / 2)
        centroid = (self.get_kpts_centroid(detection) * self.frame.shape[:-1][::-1]).astype(int)
        cv2.rectangle(self.frame, (int(x1), int(y1)), (int(x2), int(y2)), self.detector_data.main_color, 2)
        cv2.circle(self.frame, tuple(centroid), 5, self.detector_data.main_color, -1)
        # cv2.circle(self.frame, centroid, 5, self.detector_data.main_color, -1)

    async def plot_crowd_bbox(self, detections, group_id: dict):
        """Отрисовка всей толпы"""
        overlay = self.frame.copy()
        for ids in group_id.values():
            bboxes = np.concatenate(  # все bbox'ы данной группы
                [d.boxes.xyxy.data.cpu().numpy().astype(int) for i, d in enumerate(detections[0]) if i in ids])
            x1, y1, x2, y2 = min(bboxes[:, 0]), min(bboxes[:, 1]), max(bboxes[:, 2]), max(bboxes[:, 3])
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), self.detector_data.additional_color, -1)
        self.frame = cv2.addWeighted(
            overlay, self.detector_data.additional_color_visibility, self.frame,
            1 - self.detector_data.additional_color_visibility, 0)

    async def plot_bboxes_poses(self, detections) -> None:
        """Отрисовка bbox'ов и скелетов"""
        skeleton_tasks, bboxes_tasks = [], []
        for detection in detections[0]:
            if detection.boxes.conf < self.detector_data.bbox_confidence:
                continue
            bboxes_tasks.append(asyncio.create_task(self.plot_bboxes(detection)))
            skeleton_tasks.append(asyncio.create_task(
                self.plot_skeleton_kpts(detection.keypoints.data.numpy()[0], self.plot_pose_data.limbs,
                                        self.plot_pose_data.pose_limb_color, self.plot_pose_data.pose_kpt_color)
            ))
        for bbox_task in bboxes_tasks:
            await bbox_task
        for skeleton_task in skeleton_tasks:
            await skeleton_task

    async def detect_(self):
        """Детекция толпы"""
        cap = cv2.VideoCapture(self.stream_source)
        out = None
        if self.save_record:
            out = self.get_video_writer(cap)
        while cap.isOpened():
            start = datetime.datetime.now()
            ret, self.frame = cap.read()
            if not ret:
                break
            group_id = dict()
            detections = self.yolo_detector.predict(
                self.frame, classes=[0], stream=False, conf=self.detector_data.yolo_confidence)
            if len(detections[0]) != 0:
                if len(detections[0]) >= 2:
                    group_id = self.get_grouped_ids(group_id, detections)
                await self.plot_crowd_bbox(detections, group_id)  # отрисовка толпы
                await self.plot_bboxes_poses(detections)  # асинхронная отрисовка bbox'ов и скелетов
            fps = (1 / (datetime.datetime.now() - start).microseconds) / 10 ** -6
            cv2.putText(self.frame, f'fps: {str(int(np.round(fps)))}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, self.detector_data.main_color, 1, 3)
            if self.frame is not None:
                if out is not None:
                    out.write(self.frame)
                cv2.imshow('main', self.frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # main = CrowdDetector('rtsp://admin:Qwer123@192.168.9.126/cam/realmonitor?channel=1&subtype=0', 'n')
    # main = CrowdDetector('pedestrians.mp4', 'n')
    main = CrowdDetector(0, 'n')
    x = torch.load('centroids_kpts.pt')
    main.kmeans_fit(x)
    asyncio.run(main.detect_())
