import os
import asyncio
from pathlib import Path

import cv2
from ultralytics import YOLO

from misk import MainConfigurationsData
from crowd_detector import CrowdDetector
from active_gestures_detector import ActiveGesturesDetector
from raised_hands_detector import RaisedHandsDetector


class Main:

    def __init__(self, stream_source: str | int, using_detector: str, kmeans_data_name: str | None,
                 yolo_model: str = 'n', show_hands_angles: bool = False,
                 save_record: bool = False, save_trigger: bool = False):
        self.stream_source = stream_source
        self.save_record = save_record
        self.save_trigger = save_trigger
        self.crowd_detector = CrowdDetector(kmeans_data_name)
        self.active_gestures_detector = ActiveGesturesDetector(show_hands_angles)
        self.raised_hands_detector = RaisedHandsDetector(show_hands_angles)
        self.chosen_detector = using_detector
        self.main_config_data = MainConfigurationsData()

        models_path = os.path.join(Path(__file__).resolve().parents[1], 'models')
        self.yolo_detector = YOLO(os.path.join(models_path, 'yolo_models', f'yolov8{yolo_model}-pose.onnx'))
        self.frame = None

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

    async def main(self) -> None:
        """Запуск определенных детекторов"""
        # новая сработка
        trigger = False
        triggers_path = os.path.join(Path(__file__).resolve().parents[1], 'triggers')
        # если необходима запись работы детектора
        if self.save_record:
            cap = cv2.VideoCapture(self.stream_source)
            out = self.get_video_writer(cap, self.chosen_detector)
        # получаем детекции из генератора
        for detections in self.yolo_detector.track(
                self.stream_source, classes=[0], stream=True,
                conf=self.main_config_data.yolo_confidence, verbose=False):
            # обработка полученных результатов детектора, исходя из выбранного детектора
            match self.chosen_detector:
                case 'crowd':
                    self.frame, trigger = await self.crowd_detector.detect_(detections)
                case 'active_gestures':
                    self.frame, trigger = await self.active_gestures_detector.detect_(detections)
                case 'raised_hands':
                    self.frame, trigger = await self.raised_hands_detector.detect_(detections)
                case _:
                    pass
            if self.frame is not None:
                if self.save_record:
                    out.write(self.frame)
                if trigger and self.save_trigger:
                    self.save_image(triggers_path)
                cv2.imshow('main', self.frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        if self.save_record:
            out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # main = Main('pedestrians.mp4', 'crowd', 'centroids_kpts.pt')
    main = Main(0, 'raised_hands', 'centroids_kpts.pt', save_trigger=True, show_hands_angles=True)
    asyncio.run(main.main())
