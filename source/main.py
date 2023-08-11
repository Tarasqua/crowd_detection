import os
import asyncio
from pathlib import Path
import datetime
import collections

import cv2
from ultralytics import YOLO
import numpy as np

from misk import MainConfigurationsData, PlotPoseData, PlotData
from crowd_detector import CrowdDetector
from active_gestures_detector import ActiveGesturesDetector
from raised_hands_detector import RaisedHandsDetector
from squat_detector import SquatDetector
from plots import Plots


class Main:

    def __init__(self, stream_source: str | int, kmeans_data_name: str | None,
                 yolo_model: str = 'n', show_angles: bool = False,
                 save_record: bool = False, save_trigger: bool = False):
        self.stream_source = stream_source
        self.save_record = save_record
        self.save_trigger = save_trigger
        self.crowd_detector = CrowdDetector(kmeans_data_name)
        self.active_gestures_detector = ActiveGesturesDetector(show_angles)
        self.raised_hands_detector = RaisedHandsDetector(show_angles)
        self.squat_detector = SquatDetector(show_angles)
        self.main_config_data = MainConfigurationsData()

        models_path = os.path.join(Path(__file__).resolve().parents[1], 'models')
        self.yolo_detector = YOLO(os.path.join(models_path, 'yolo_models', f'yolov8{yolo_model}-pose.onnx'))
        self.frame = None
        self.stream_frame = None
        self.frame_click_x, self.frame_click_y = 0, 0
        backgrounds_path = os.path.join(Path(__file__).resolve().parents[1], 'backgrounds')
        self.backgrounds = {
            'crowd': cv2.imread(os.path.join(backgrounds_path, 'crowd_detector.png')),
            'active_gestures': cv2.imread(os.path.join(backgrounds_path, 'active_gestures_detector.png')),
            'raised_hands': cv2.imread(os.path.join(backgrounds_path, 'raised_hands_detector.png')),
            'squat': cv2.imread(os.path.join(backgrounds_path, 'squat_detector.png')),
            '': cv2.imread(os.path.join(backgrounds_path, 'empty_.png'))
        }
        self.main_margin_x, self.main_margin_y = 290, 110
        self.triggers = collections.deque(maxlen=3)
        self.triggers_margins = np.array([[1552, 145], [1552, 455], [1552, 765]])
        self.detectors = {'crowd': (np.array([43, 239, 176, 258]), False),
                          'active_gestures': (np.array([43, 268, 226, 287]), False),
                          'raised_hands': (np.array([43, 296, 145, 315]), False),
                          'squat': (np.array([43, 324, 207, 343]), False)}
        self.enabled_detector = ''

    def click_event(self, event, x, y, flags, params) -> None:
        """Кликер для определения координат"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.frame_click_x, self.frame_click_y = x, y

    @staticmethod
    def get_video_writer(cap):
        records_folder_path = os.path.join(Path(__file__).resolve().parents[1], 'records')
        if not os.path.exists(records_folder_path):
            os.mkdir(records_folder_path)
        out = cv2.VideoWriter(
            os.path.join(records_folder_path, f'{len(os.listdir(records_folder_path)) + 1}.mp4'),
            -1, 20.0, (int(cap.get(3)), int(cap.get(4))))
        return out

    def save_image(self, directory_path: str):
        """Сохраняет картинку в выбранной директории"""
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)
        cv2.imwrite(os.path.join(directory_path, f'{len(os.listdir(directory_path)) + 1}.png'), self.stream_frame)

    def show_triggers(self, window_with_background: np.array, detector_name: str):
        """Отображение сработок на боковой панели"""
        sorted_triggers = sorted(self.triggers, key=lambda x: x[1], reverse=True)  # сортируем по времени
        for (trigger, trigger_time), margins in zip(sorted_triggers, self.triggers_margins):
            resized = cv2.resize(trigger, (np.array(trigger.shape[:-1][::-1]) * 0.5).astype(int))  # 320 х 240
            window_with_background[margins[1]:resized.shape[0] + margins[1],
                                   margins[0]:resized.shape[1] + margins[0], :] = resized
            cv2.putText(window_with_background, f"{detector_name}: {trigger_time.strftime('%H:%M:%S')}",
                        (margins[0], (resized.shape[0] + margins[1]).astype(int) + 16),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.FILLED)
        return window_with_background

    @staticmethod
    def get_trigger_name(enabled_detector):
        """Возвращает название события, исходя из выбранного детектора, на русском"""
        match enabled_detector:
            case 'crowd':
                return 'Скопление людей'
            case 'active_gestures':
                return 'Активная жестикуляция'
            case 'raised_hands':
                return 'Поднятие рук'
            case 'squat':
                return 'Человек на корточках'
            case _:
                pass

    def enable_detector(self):
        """Включает и выключает детекторы и возвращает список включенных"""
        for detector, ((x_min, y_min, x_max, y_max), is_enabled) in self.detectors.items():
            if x_min <= self.frame_click_x <= x_max and y_min <= self.frame_click_y <= y_max:
                self.detectors[detector] = ((x_min, y_min, x_max, y_max), not is_enabled)
                if is_enabled:
                    self.enabled_detector = ''
                else:
                    self.enabled_detector = detector

    def frame_processing(self, trigger, triggers_path):
        """Постобработка изображения со вставкой стрима на макет и сработками"""
        if self.stream_frame is not None:
            # основной стрим
            resized = cv2.resize(self.stream_frame, (np.array(self.stream_frame.shape[:-1][::-1]) * 1.2).astype(int))
            self.frame[self.main_margin_y:resized.shape[0] + self.main_margin_y,
                       self.main_margin_x:resized.shape[1] + self.main_margin_x, :] = resized
            if trigger:
                if self.save_trigger:
                    self.save_image(triggers_path)
                self.triggers.append(np.array([self.stream_frame, datetime.datetime.now()]))
            self.frame = self.show_triggers(self.frame, self.get_trigger_name(self.enabled_detector))

    async def main(self) -> None:
        """Запуск определенных детекторов"""
        cv2.namedWindow('main')
        cv2.setMouseCallback('main', self.click_event)
        # новая сработка
        trigger = False
        triggers_path = os.path.join(Path(__file__).resolve().parents[1], 'triggers')
        # если необходима запись работы детектора
        if self.save_record:
            cap = cv2.VideoCapture(self.stream_source)
            out = self.get_video_writer(cap)
        # получаем детекции из генератора
        for detections in self.yolo_detector.track(
                self.stream_source, classes=[0], stream=True,
                conf=self.main_config_data.yolo_confidence, verbose=False):
            if self.frame_click_x != 0 and self.frame_click_y != 0:
                self.enable_detector()
            self.frame = self.backgrounds[self.enabled_detector].copy()
            # обработка полученных результатов детектора, исходя из выбранного детектора
            match self.enabled_detector:
                case 'crowd':
                    self.stream_frame, trigger = await self.crowd_detector.detect_(detections)
                case 'active_gestures':
                    self.stream_frame, trigger = await self.active_gestures_detector.detect_(detections)
                case 'raised_hands':
                    self.stream_frame, trigger = await self.raised_hands_detector.detect_(detections)
                case 'squat':
                    self.stream_frame, trigger = await self.squat_detector.detect_(detections)
                case _:
                    self.stream_frame, trigger = detections.orig_img, False
            self.frame_processing(trigger, triggers_path)
            self.frame_click_x, self.frame_click_y = 0, 0
            if self.save_record:
                out.write(self.stream_frame)
            cv2.imshow('main', self.frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        if self.save_record:
            out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # main = Main('pedestrians.mp4', 'crowd', 'centroids_kpts.pt')
    main = Main(0, 'centroids_kpts.pt', save_trigger=True, show_angles=False)
    asyncio.run(main.main())
