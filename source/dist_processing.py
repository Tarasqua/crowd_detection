import os
import asyncio

import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm

from misc import PlotPoseData


class DistProcessing:

    def __init__(self, yolo_model, folder_path: str, save_results: bool = True):
        self.folder_path = folder_path
        self.save_results = save_results
        self.yolo_detector = YOLO(os.path.join('../models/yolo_models', f'yolov8{yolo_model}-pose.onnx'))
        self.kpts_confidence = 0.5
        self.frame = None
        self.plot_data = PlotPoseData()

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
    def get_frame_centroids(detections, conf: float, frame_shape: tuple):
        width, height = frame_shape
        xyxy = [d.boxes.xyxy.data.numpy()[0] for d in detections[0] if float(d.boxes.conf.data) > conf]
        centroids = [((x1 / width + x2 / width) / 2, (y1 / height + y2 / height) / 2) for x1, y1, x2, y2 in xyxy]
        return centroids

    @staticmethod
    def get_kpts_centroid(detection, kpts, conf: float, frame_shape: tuple):
        """
        Возвращает центроид по ключевым точкам, где:
        5 - левое плечо, 6 - правое плечо, 11 - левое бедро, 12 - правое бедро.
        Если видны все 4 точки, то вычисляется центроид относительно них;
        если видны только верхние две, то центроидом считается середина отрезка между плечами;
        если ни одни из точек не видна, то центроид находится, исходя из bbox'а
        """
        points = np.concatenate([[kpts[:, :-1][5]], [kpts[:, :-1][6]], [kpts[:, :-1][11]], [kpts[:, :-1][12]]])
        points = points / frame_shape  # в относительные координаты
        confs = np.array([kpts[:, -1][5], kpts[:, -1][6], kpts[:, -1][11], kpts[:, -1][12]])
        if all(confs > conf):
            return points.sum(axis=0) / len(points)
        elif any(confs[-2:] < conf) and all(confs[:-2] > conf):
            return points[:-2].sum(axis=0) / 2
        else:
            xyxy = np.array(
                [detection.boxes.xyxy.data.numpy()[0][:-2], detection.boxes.xyxy.data.numpy()[0][-2:]]
            ) / frame_shape
            return xyxy.sum(axis=0) / 2

    async def make_centroids(self):
        X = []
        for im_num, image_name in enumerate(
                tqdm(os.listdir(self.folder_path), desc="Processing files", unit="files", ncols=75, colour='#37B6BD')
        ):
            self.frame = cv2.imread(os.path.join(self.folder_path, image_name))
            detections = self.yolo_detector.predict(self.frame, classes=[0], stream=False,
                                                    conf=self.kpts_confidence, verbose=False)
            if len(detections[0]) == 0:
                continue
            frame_shape = self.frame.shape[:-1][::-1]
            tasks = []
            centroids = []
            for detection in detections[0]:
                kpts = detection.keypoints.data.numpy()[0]
                centroid = self.get_kpts_centroid(detection, kpts, self.kpts_confidence, frame_shape)
                if any(centroid >= 1):
                    continue
                centroids.append(centroid)
                cv2.circle(self.frame, tuple((centroid * frame_shape).astype(int)), 7, (0, 255, 0), 2)
                tasks.append(asyncio.create_task(
                    self.plot_skeleton_kpts(detection.keypoints.data.numpy()[0], self.plot_data.limbs,
                                            self.plot_data.pose_limb_color, self.plot_data.pose_kpt_color)
                ))
            for task in tasks:
                await task
            if centroids:
                X.append(centroids)
            if self.save_results:
                cv2.imwrite(os.path.join('../prepared_images', f'{im_num}.png'), self.frame)
        return np.concatenate(X)

    async def clusters_search(self):
        wcss = []
        # X = await self.make_centroids()
        # torch.save(X, 'centroids_kpts.pt')
        X = torch.load('../models/kmeans_train_data/centroids_kpts.pt')
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 11), wcss)
        plt.title('Выбор количества кластеров методом локтя')
        plt.xlabel('Количество кластеров')
        plt.ylabel('WCSS')
        plt.grid()
        plt.show()

    @staticmethod
    def kmeans_test():
        X = torch.load('../models/kmeans_train_data/centroids_kpts.pt')
        kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=42)
        y_pred = kmeans.fit_predict(X)
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='Paired')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=150, c='red', marker='^',
                    label='Centroids')
        plt.legend(loc='upper right')
        plt.show()


if __name__ == '__main__':
    proc = DistProcessing('x', '../images')
    # asyncio.run(proc.clusters_search())
    proc.kmeans_test()
