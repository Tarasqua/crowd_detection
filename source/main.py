import os
import asyncio
from pathlib import Path
import datetime
import collections

import cv2
from ultralytics import YOLO
import numpy as np

from misk import MainConfigurationsData
from crowd_detector import CrowdDetector
from active_gestures_detector import ActiveGesturesDetector
from raised_hands_detector import RaisedHandsDetector
from squat_detector import SquatDetector


class Main:

    def __init__(self, stream_source: str | int, using_detector: str, kmeans_data_name: str | None,
                 yolo_model: str = 'n', show_angles: bool = False,
                 save_record: bool = False, save_trigger: bool = False):
        self.stream_source = stream_source
        self.save_record = save_record
        self.save_trigger = save_trigger
        self.crowd_detector = CrowdDetector(kmeans_data_name)
        self.active_gestures_detector = ActiveGesturesDetector(show_angles)
        self.raised_hands_detector = RaisedHandsDetector(show_angles)
        self.squat_detector = SquatDetector(show_angles)
        self.chosen_detector = using_detector
        self.main_config_data = MainConfigurationsData()

        models_path = os.path.join(Path(__file__).resolve().parents[1], 'models')
        self.yolo_detector = YOLO(os.path.join(models_path, 'yolo_models', f'yolov8{yolo_model}-pose.onnx'))
        self.frame = None
        self.frame_click_x, self.frame_click_y = None, None
        self.backgrounds = {'crowd': cv2.imread('crowd_detector_main.png')}
        self.main_margin_x, self.main_margin_y = 290, 110
        self.triggers = collections.deque(maxlen=3)
        self.triggers_margins = np.array([[1552, 145], [1552, 455], [1552, 765]])

    def click_event(self, event, x, y, flags, params) -> None:
        """Кликер для определения координат"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.frame_click_x, self.frame_click_y = x, y

    @staticmethod
    def get_video_writer(cap, detector_name):
        records_folder_path = os.path.join(Path(__file__).resolve().parents[1], 'records')
        crowd_folder = os.path.join(records_folder_path, f'{detector_name}_records')
        if not os.path.exists(records_folder_path):
            os.mkdir(records_folder_path)
        if not os.path.exists(crowd_folder):
            os.mkdir(crowd_folder)
        out = cv2.VideoWriter(
            os.path.join(crowd_folder, f'{detector_name}_{len(os.listdir(crowd_folder)) + 1}.mp4'),
            -1, 20.0, (int(cap.get(3)), int(cap.get(4))))
        return out

    def save_image(self, directory_path: str):
        """Сохраняет картинку в выбранной директории"""
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)
        cv2.imwrite(os.path.join(directory_path, f'{len(os.listdir(directory_path)) + 1}.png'), self.frame)

    def show_triggers(self, window_with_background: np.array, detector_name: str):
        """Отображение сработок на боковой панели"""
        sorted_triggers = sorted(self.triggers, key=lambda x: x[1], reverse=True)  # сортируем по времени
        for (trigger, trigger_time), margins in zip(sorted_triggers, self.triggers_margins):
            resized = cv2.resize(trigger, (np.array(trigger.shape[:-1][::-1]) * 0.5).astype(int))  # 320 х 240
            window_with_background[margins[1]:resized.shape[0] + margins[1],
                                   margins[0]:resized.shape[1] + margins[0], :] = resized
            cv2.putText(window_with_background, f"{detector_name}: {trigger_time.strftime('%H:%M:%S')}",
                        (margins[0], (resized.shape[0] + margins[1]).astype(int) + 18),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv2.FILLED)
        return window_with_background

    def get_trigger_name(self):
        """Возвращает название события, исходя из выбранного детектора, на русском"""
        match self.chosen_detector:
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

    async def main(self) -> None:
        """Запуск определенных детекторов"""
        cv2.namedWindow('main')
        cv2.setMouseCallback('main', self.click_event)
        # новая сработка
        trigger = False
        triggers_path = os.path.join(Path(__file__).resolve().parents[1], 'triggers')
        trigger_name = self.get_trigger_name()
        # если необходима запись работы детектора
        if self.save_record:
            cap = cv2.VideoCapture(self.stream_source)
            out = self.get_video_writer(cap, self.chosen_detector)
        # получаем детекции из генератора
        for detections in self.yolo_detector.track(
                self.stream_source, classes=[0], stream=True,
                conf=self.main_config_data.yolo_confidence, verbose=False):
            show_frame = self.backgrounds['crowd'].copy()
            # обработка полученных результатов детектора, исходя из выбранного детектора
            match self.chosen_detector:
                case 'crowd':
                    self.frame, trigger = await self.crowd_detector.detect_(detections)
                case 'active_gestures':
                    self.frame, trigger = await self.active_gestures_detector.detect_(detections)
                case 'raised_hands':
                    self.frame, trigger = await self.raised_hands_detector.detect_(detections)
                case 'squat':
                    self.frame, trigger = await self.squat_detector.detect_(detections)
                case _:
                    pass
            if self.frame is not None:
                # основной стрим
                resized = cv2.resize(self.frame, (np.array(self.frame.shape[:-1][::-1]) * 1.2).astype(int))
                show_frame[self.main_margin_y:resized.shape[0] + self.main_margin_y,
                           self.main_margin_x:resized.shape[1] + self.main_margin_x, :] = resized
                if self.save_record:
                    out.write(self.frame)
                if trigger:
                    if self.save_trigger:
                        self.save_image(triggers_path)
                    self.triggers.append(np.array([self.frame, datetime.datetime.now()]))
                show_frame = self.show_triggers(show_frame, trigger_name)
                cv2.imshow('main', show_frame)
            print(self.frame_click_x, self.frame_click_y)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        if self.save_record:
            out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # main = Main('pedestrians.mp4', 'crowd', 'centroids_kpts.pt')
    main = Main(0, 'active_gestures', 'centroids_kpts.pt', save_trigger=True, show_angles=False)
    asyncio.run(main.main())
