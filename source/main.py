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
        self.people_on_frame = cv2.imread(os.path.join(backgrounds_path, 'people_on_frame.png'))
        self.main_margin_x, self.main_margin_y = 290, 110
        self.triggers = collections.deque(maxlen=3)
        self.triggers_margins = np.array([[1552, 145], [1552, 455], [1552, 765]])
        self.triggers_table = collections.deque(maxlen=5)
        self.triggers_table_margins = np.array([])
        self.detectors = {'crowd': (np.array([43, 239, 176, 258]), False),
                          'active_gestures': (np.array([43, 268, 226, 287]), False),
                          'raised_hands': (np.array([43, 296, 145, 315]), False),
                          'squat': (np.array([43, 324, 207, 343]), False)}
        self.enabled_detector = ''
        self.full_screen = False

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

    def show_triggers(self):
        """Отображение сработок на боковой панели"""
        sorted_triggers = sorted(self.triggers, key=lambda x: x[2], reverse=True)  # сортируем по времени
        for (trigger, trigger_name, trigger_time), margins in zip(sorted_triggers, self.triggers_margins):
            resized = cv2.resize(trigger, (np.array(trigger.shape[:-1][::-1]) * 0.5).astype(int))  # 320 х 240
            self.frame[margins[1]:resized.shape[0] + margins[1],
                       margins[0]:resized.shape[1] + margins[0], :] = resized
            cv2.putText(self.frame, f"{trigger_name}: {trigger_time.strftime('%H:%M:%S')}",
                        (margins[0], (resized.shape[0] + margins[1]).astype(int) + 16),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.FILLED)

    def show_table(self):
        line_coords = (923, 1431)
        tl_point = (924, 784)
        x, y = tl_point
        font = cv2.FONT_HERSHEY_COMPLEX
        fontScale = 0.5
        color = (255, 255, 255)
        thickness = 1
        margin_x, margin_y = 50, 60
        sorted_triggers = sorted(self.triggers_table, key=lambda x: x[0], reverse=True)  # сортируем по времени
        for i, (time, camera, detector_name, event) in enumerate(sorted_triggers):
            time = time.strftime('%H:%M:%S')
            last_y = y
            y = tl_point[1] + i * margin_y
            size, height = cv2.getTextSize(str(time), font, fontScale, thickness)
            if i >= 1:
                line_y = last_y + int((y - (last_y + height)) / 2)
                cv2.line(self.frame, (line_coords[0], line_y), (line_coords[1], line_y), color, 2)
            cv2.putText(self.frame, time, (938, y), font,
                        fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(self.frame, str(camera), (1059, y), font,
                        fontScale, color, thickness, cv2.LINE_AA)
            detector_name_split = detector_name.split(' ')
            cv2.putText(self.frame, detector_name_split[0], (1144, y), font,
                        fontScale, color, thickness, cv2.LINE_AA)
            if len(detector_name_split) == 3:
                cv2.putText(self.frame, ' '.join(detector_name_split[1:]), (1144, y+3 * height), font,
                            fontScale, color, thickness, cv2.LINE_AA)
            else:
                cv2.putText(self.frame, detector_name_split[1], (1144, y + 3 * height), font,
                            fontScale, color, thickness, cv2.LINE_AA)
            event_split = event.split(' ')
            cv2.putText(self.frame, event_split[0], (1283, y), font,
                        fontScale, color, thickness, cv2.LINE_AA)
            if len(event_split) == 3:
                cv2.putText(self.frame, ' '.join(event_split[1:]), (1283, y+3 * height), font,
                            fontScale, color, thickness, cv2.LINE_AA)
            else:
                cv2.putText(self.frame, event_split[1], (1283, y + 3 * height), font,
                            fontScale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def get_trigger_name(enabled_detector):
        """Возвращает название детектора, исходя из выбранного детектора, на русском"""
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

    @staticmethod
    def get_event_name(enabled_detector):
        """Возвращает название события, исходя из выбранного детектора, на русском"""
        match enabled_detector:
            case 'crowd':
                return 'Группа людей'
            case 'active_gestures':
                return 'Подозрительные жесты'
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

    def frame_processing(self, trigger, triggers_path, people_on_frame: int):
        """Постобработка изображения со вставкой стрима на макет и сработками"""
        if self.stream_frame is not None:
            trigger_name = self.get_trigger_name(self.enabled_detector)
            event_name = self.get_event_name(self.enabled_detector)
            if trigger:
                if self.save_trigger:
                    self.save_image(triggers_path)
                self.triggers.append(np.array([self.stream_frame, trigger_name, datetime.datetime.now()]))
                self.triggers_table.append(np.array([datetime.datetime.now(), 1, trigger_name, event_name]))
            self.show_triggers()
            self.show_table()
            self.stream_frame[449:449 + self.people_on_frame.shape[0],
                              200:200 + self.people_on_frame.shape[1], :] = self.people_on_frame
            cv2.putText(self.stream_frame, f"Людей в кадре: {people_on_frame}",
                        (245, 466), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.FILLED)
            # основной стрим
            if self.full_screen:
                resized = cv2.resize(self.stream_frame, (1505, 1010))
                self.frame[70:, :1505, :] = resized
            else:
                resized = cv2.resize(self.stream_frame,
                                     (np.array(self.stream_frame.shape[:-1][::-1]) * 1.2).astype(int))
                self.frame[self.main_margin_y:resized.shape[0] + self.main_margin_y,
                           self.main_margin_x:resized.shape[1] + self.main_margin_x, :] = resized

    @staticmethod
    def hide_title() -> None:
        prop_value = cv2.getWindowProperty('main', cv2.WND_PROP_FULLSCREEN)
        if prop_value == 0.0:
            cv2.setWindowProperty('main', cv2.WND_PROP_FULLSCREEN, 1.0)
        elif prop_value == 1.0:
            cv2.setWindowProperty('main', cv2.WND_PROP_FULLSCREEN, 0.0)

    async def main(self) -> None:
        """Запуск определенных детекторов"""
        cv2.namedWindow('main', flags=cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback('main', self.click_event)
        cv2.setWindowProperty('main', cv2.WND_PROP_TOPMOST, 1.0)
        self.hide_title()
        # новая сработка
        triggers_path = os.path.join(Path(__file__).resolve().parents[1], 'triggers')
        # если необходима запись работы детектора
        if self.save_record:
            cap = cv2.VideoCapture(self.stream_source)
            out = self.get_video_writer(cap)
        # получаем детекции из генератора
        for detections in self.yolo_detector.track(
                self.stream_source, classes=[0], stream=True,
                conf=self.main_config_data.yolo_confidence, verbose=False):
            people_on_frame = len(detections)
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
            self.frame_processing(trigger, triggers_path, people_on_frame)
            self.frame_click_x, self.frame_click_y = 0, 0
            if self.save_record:
                out.write(self.stream_frame)
            cv2.imshow('main', self.frame)
            if cv2.waitKey(1) & 0xFF == ord('f'):
                self.full_screen = not self.full_screen
            if cv2.waitKey(1) & 0xFF == 27:
                break
        if self.save_record:
            out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # main = Main('pedestrians.mp4', 'crowd', 'centroids_kpts.pt')
    main = Main(0, 'centroids_kpts.pt', save_trigger=True, show_angles=False)
    asyncio.run(main.main())
