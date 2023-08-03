import asyncio
import datetime
import os

import cv2
import numpy as np
import av
from ultralytics import YOLO
from scipy.spatial import distance as dist

from misk import PlotPoseData


class Processing:

    def __init__(self):
        self.yolo_tracker = YOLO(os.path.join('../models/yolo_models', f'yolov8n-pose.onnx'))
        self.bbox_confidence = 0.7
        self.kpts_confidence = 0.5
        self.min_distance = 300
        self.plot_data = PlotPoseData()
        self.frame = None

        self.det_trigger = False
        self.violate_num = 0

    async def plot_skeleton_kpts(self, kpts, limbs, pose_limb_color, pose_kpt_color):
        for p_id, point in enumerate(kpts):
            x_coord, y_coord, conf = point
            if conf < self.kpts_confidence:
                continue
            r, g, b = pose_kpt_color[p_id]
            cv2.circle(self.frame, (int(x_coord), int(y_coord)), 5, (int(r), int(g), int(b)), -1)
        for sk_id, sk in enumerate(limbs):
            r, g, b = pose_limb_color[sk_id]
            if kpts[sk[0]][2] < self.kpts_confidence or kpts[sk[1]][2] < self.kpts_confidence:
                continue
            pos1 = int(kpts[sk[0]][0]), int(kpts[sk[0]][1])
            pos2 = int(kpts[sk[1]][0]), int(kpts[sk[1]][1])
            cv2.line(self.frame, pos1, pos2, (int(r), int(g), int(b)), thickness=2)

    @staticmethod
    def get_centroids(detections, conf: float):
        xyxy = [d.boxes.xyxy.data.numpy()[0] for d in detections[0] if float(d.boxes.conf.data) > conf]
        centroids = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2 in xyxy]
        return centroids

    def get_violate_set(self, violate_set: set, detections, min_distance):
        centroids = np.array(self.get_centroids(detections, self.bbox_confidence))
        if centroids.size == 0:
            return violate_set
        distances = dist.cdist(centroids, centroids, metric='euclidean')
        for i in range(0, distances.shape[0]):
            for j in range(i + 1, distances.shape[1]):
                if distances[i, j] < min_distance:
                    violate_set.add(i)
                    violate_set.add(j)
        return violate_set

    async def plot_bboxes(self, i, detection, violate):
        x1, y1, x2, y2 = detection.boxes.xyxy.data.numpy()[0]
        centroid = int((x1 + x2) / 2), int((y1 + y2) / 2)
        color = (0, 255, 0)
        if i in violate:
            color = (0, 0, 255)
        cv2.rectangle(self.frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.circle(self.frame, centroid, 7, color, 2)

    async def processing(self, video_frame):
        self.frame = video_frame.to_ndarray(format='bgr24')
        violate = set()
        detections = self.yolo_tracker.predict(self.frame, classes=[0], stream=False)
        if len(detections[0]) != 0:
            if len(detections[0]) >= 2:
                violate = self.get_violate_set(violate, detections, self.min_distance)
                if len(violate) != self.violate_num:
                    self.violate_num = len(violate)
                    self.det_trigger = True
            skeleton_tasks = []
            bboxes_tasks = []
            for i, detection in enumerate(detections[0]):
                if detection.boxes.conf < self.bbox_confidence:
                    continue
                # отрисовка людей, находящихся рядом
                bboxes_tasks.append(asyncio.create_task(self.plot_bboxes(i, detection, violate)))
                # отрисовка скелетов
                skeleton_tasks.append(asyncio.create_task(
                    self.plot_skeleton_kpts(detection.keypoints.data.numpy()[0], self.plot_data.limbs,
                                            self.plot_data.pose_limb_color, self.plot_data.pose_kpt_color)
                ))
            for bbox_task in bboxes_tasks:
                await bbox_task
            for skeleton_task in skeleton_tasks:
                await skeleton_task

        if self.det_trigger:
            # success, encoded_image = cv2.imencode('.png', self.frame)
            # bytes_img = encoded_image.tobytes()
            cv2.imwrite(os.path.join('triggers', f'{len(os.listdir("triggers")) + 1}.png'), self.frame)
            self.det_trigger = False
        new_frame = av.VideoFrame.from_ndarray(self.frame, format="bgr24")
        new_frame.pts = video_frame.pts
        new_frame.time_base = video_frame.time_base
        return new_frame
