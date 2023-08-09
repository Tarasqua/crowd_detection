import asyncio
import datetime
import os
import itertools
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance as dist

from kmeans_separator import KMeansSeparator
from misk import PlotPoseData, CrowdDetectorData


class CrowdDetector:

    def __init__(self, stream_source: str | int, kmeans_data_name: str | None,
                 yolo_model: str = 'n', save_record: bool = False, save_triggers: bool = False):
        self.stream_source = stream_source
        self.save_record = save_record
        self.save_triggers = save_triggers
        self.detector_data = CrowdDetectorData()
        self.plot_pose_data = PlotPoseData()
        models_path = os.path.join(Path(__file__).resolve().parents[1], 'models')
        self.yolo_detector = YOLO(os.path.join(models_path, 'yolo_models', f'yolov8{yolo_model}-pose.onnx'))
        # TODO: описать случай, при котором подгружается, а не обучается KMeans
        if kmeans_data_name is None:
            kmeans_ = KMeansSeparator(None)
        else:
            kmeans_ = KMeansSeparator(
                os.path.join(models_path, 'kmeans_train_data', kmeans_data_name))
            self.kmeans_model = kmeans_.kmeans_fit(self.detector_data.kmeans_n_clusters)
        self.frame = None
        self.group_bbox, self.new_group_found = {}, False

    @staticmethod
    def get_video_writer(cap):
        records_folder_path = os.path.join(Path(__file__).resolve().parents[1], 'records')
        crowd_folder = os.path.join(records_folder_path, 'crowd_records')
        if not os.path.exists(records_folder_path):
            os.mkdir(records_folder_path)
        if not os.path.exists(crowd_folder):
            os.mkdir(crowd_folder)
        out = cv2.VideoWriter(
            os.path.join(crowd_folder, f'crowd_{len(os.listdir(crowd_folder)) + 1}.mp4'),
            -1, 20.0, (int(cap.get(3)), int(cap.get(4))))
        return out

    def save_image(self, directory_path: str):
        """Сохраняет картинку в выбранной директории"""
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)
        cv2.imwrite(os.path.join(directory_path, f'{len(os.listdir(directory_path)) + 1}.png'), self.frame)

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
        centroids = np.array(
            [self.get_kpts_centroid(detection) for detection in detections]
        ).astype('float64')
        if centroids.size == 0:
            return group_id
        # формируем словарь с id: группа, исходя из разделения по группам kmeans
        id_group = {person_id: group for person_id, group in enumerate(self.kmeans_model.predict(centroids))}
        # раскидываем id по группам
        for k, v in id_group.items():
            group_id.setdefault(v, []).append(k)
        group_id = self.check_groups(detections, group_id)
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

    def check_groups(self, detections: list, group_id: dict) -> dict:
        """
        Проверка на то, что группы разделены верно, во избежание случаев, когда KMeans разделяет на несколько разных
        групп рядом стоящих людей. Смотрится расстояние между центроидами групп и IOU.
        """
        group_bbox = self.get_group_bbox_centroid(detections, group_id)
        # только те, где количество человек больше порогового
        checked_group_id = {g: ids for g, ids in group_id.items()
                            if len(ids) >= self.detector_data.min_crowd_num_of_people}
        for (group1, ids1), (group2, ids2) in itertools.combinations(group_id.items(), 2):  # каждый с каждым
            # сравниваем расстояния и iou
            iou = self.get_iou(np.concatenate(group_bbox[group1][:-1]), np.concatenate(group_bbox[group2][:-1]))
            if np.linalg.norm(group_bbox[group1][-1] - group_bbox[group2][-1]) < self.detector_data.min_distance or \
                    iou > self.detector_data.iou_threshold:
                if group1 in checked_group_id and group2 not in checked_group_id:
                    checked_group_id[group1] = ids1 + ids2
                elif group1 not in checked_group_id and group2 in checked_group_id:
                    checked_group_id[group2] = ids1 + ids2
                elif group1 in checked_group_id and group2 in checked_group_id:
                    checked_group_id[group1] = ids1 + ids2
                    del checked_group_id[group2]
        return checked_group_id

    @staticmethod
    def get_iou(bbox1: np.array, bbox2: np.array) -> float:
        """Возвращает IOU по входным bbox'ам"""
        # добавляем 1 при вычислении высоты и ширины, чтобы избежать ошибок деления на ноль
        intersection_height = np.maximum(
            np.minimum(bbox1[3], bbox2[3]) - np.maximum(bbox1[1], bbox2[1]) + 1,
            np.array(0.)
        )
        intersection_width = np.maximum(
            np.minimum(bbox1[2], bbox2[2]) - np.maximum(bbox1[0], bbox2[0]) + 1,
            np.array(0.)
        )
        area_of_intersection = intersection_height * intersection_width
        area_of_union = (bbox1[3] - bbox1[1] + 1) * (bbox1[2] - bbox1[0] + 1) + \
                        (bbox2[3] - bbox2[1] + 1) * (bbox2[2] - bbox2[0] + 1) - \
                        (intersection_height * intersection_width)
        return area_of_intersection / area_of_union

    def get_group_bbox_centroid(self, detections: list, group_id: dict) -> np.array:
        """Возвращает целиковые bbox'ы по группам и их центроиды
        в относительных координатах в формате [[x1,y1],[x2,y3],[x_c,y_c]]"""
        group_bbox = {}
        for group, ids in group_id.items():
            bboxes = np.concatenate(  # все bbox'ы данной группы
                [d.boxes.xyxy.data.cpu().numpy().astype(int) for i, d in enumerate(detections) if i in ids])
            xyxy = np.array([[min(bboxes[:, 0]), min(bboxes[:, 1])],
                             [max(bboxes[:, 2]), max(bboxes[:, 3])]]) / self.frame.shape[:-1][::-1]
            centroid = xyxy.sum(axis=0) / 2
            group_bbox[group] = np.concatenate([xyxy, [centroid]])
        return group_bbox

    def get_violate_group_id(self, centroids: np.array, group_id: dict) -> dict:
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
        violate_group_id = {g: ids for g, ids in violate_group_id.items()  # убираем пустые
                            if not np.array_equal(ids, [])}
        return violate_group_id

    async def plot_bbox(self, detection) -> None:
        """Отрисовка bbox'а и центроида человека"""
        x1, y1, x2, y2 = detection.boxes.xyxy.data.numpy()[0]
        centroid = (self.get_kpts_centroid(detection) * self.frame.shape[:-1][::-1]).astype(int)
        cv2.rectangle(self.frame, (int(x1), int(y1)), (int(x2), int(y2)), self.detector_data.main_color, 2)
        cv2.circle(self.frame, tuple(centroid), 5, self.detector_data.main_color, -1)

    async def plot_crowd_bbox(self, detections: list, group_id: dict) -> list:
        """Отрисовка всей толпы"""
        overlay = self.frame.copy()
        crowd_bboxes = []
        for ids in group_id.values():
            bboxes = np.concatenate(  # все bbox'ы данной группы
                [d.boxes.xyxy.data.cpu().numpy().astype(int) for i, d in enumerate(detections) if i in ids])
            crowd_bbox = bboxes[:, :-2].min(axis=0), bboxes[:, 2:].max(axis=0)
            crowd_bboxes.append(crowd_bbox)
            cv2.rectangle(overlay, tuple(crowd_bbox[0]), tuple(crowd_bbox[1]), self.detector_data.additional_color, -1)
        self.frame = cv2.addWeighted(
            overlay, self.detector_data.additional_color_visibility, self.frame,
            1 - self.detector_data.additional_color_visibility, 0)
        return crowd_bboxes

    def check_new_groups(self, crowd_bboxes: list):
        """
        Отлавливает новые скопления людей в кадре
        По IOU тречит рамки скопления людей и оповещает, если появляются новые
        """
        if not self.group_bbox:  # первое вхождение
            for crowd_number, crowd_bbox in enumerate(crowd_bboxes):
                self.group_bbox[crowd_number] = crowd_bbox, True
            self.new_group_found = True
        else:
            for group, bbox in self.group_bbox.items():
                if crowd_bboxes:  # если соответствий больше не останется
                    iou = [self.get_iou(np.concatenate(bbox[0]), np.concatenate(crowd)) for crowd in crowd_bboxes]
                    max_iou = np.argmax(iou)  # индекс
                    if iou[max_iou] >= self.detector_data.iou_crowd_threshold:
                        self.group_bbox[group] = crowd_bboxes[max_iou], True
                        crowd_bboxes.pop(max_iou)
                    else:
                        # флаг False, чтобы следить уже имеющаяся рамка ни с чем не сопоставилась
                        self.group_bbox[group] = bbox, False
                else:
                    self.group_bbox[group] = bbox, False
            for crowd_bbox in crowd_bboxes:  # новые скопления людей
                self.group_bbox[max(self.group_bbox.keys()) + 1] = crowd_bbox, True
                self.new_group_found = True
            # удаляем те рамки, которые ни с чем не сопоставились
            self.group_bbox = {group: bbox for group, bbox in self.group_bbox.items() if bbox[1]}

    async def plot_bboxes_poses(self, detections: list) -> None:
        """Отрисовка bbox'ов и скелетов"""
        skeleton_tasks, bboxes_tasks = [], []
        for detection in detections:
            if detection.boxes.conf < self.detector_data.bbox_confidence:
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
        triggers_path = os.path.join(Path(__file__).resolve().parents[1], 'triggers')
        for detections in self.yolo_detector.track(
                self.stream_source, classes=[0], stream=True, conf=self.detector_data.yolo_confidence, verbose=False):
            self.frame = detections.orig_img
            group_id = dict()
            if len(detections) != 0:
                if len(detections) >= self.detector_data.min_crowd_num_of_people:
                    group_id = self.get_grouped_ids(group_id, detections)
                    crowd_bboxes = await self.plot_crowd_bbox(detections, group_id)  # отрисовка толпы
                    if crowd_bboxes:
                        self.check_new_groups(crowd_bboxes)
                else:
                    self.group_bbox = {}
                await self.plot_bboxes_poses(detections)  # асинхронная отрисовка bbox'ов и скелетов
            else:
                print('a')
            if self.new_group_found and self.save_triggers:
                self.save_image(triggers_path)
                self.new_group_found = False
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
    # main = CrowdDetector('rtsp://admin:Qwer123@192.168.9.126/cam/realmonitor?channel=1&subtype=0', 'n')
    main = CrowdDetector('pedestrians.mp4', 'centroids_kpts.pt', 'n')
    # main = CrowdDetector(0, 'centroids_kpts.pt', 'n')
    asyncio.run(main.detect_())
