import os
from pathlib import Path

import torch
import pickle
from sklearn.cluster import KMeans


class TrainDataNotTransmittedException(Exception):
    """Ошибка, возникающая при отсутствии данных для обучения KMeans."""
    def __str__(self):
        return "\nTraining data has not been transmitted"


class ModelNotFoundException(Exception):
    """Ошибка, возникающая при отсутствии kmeans модели в директории с моделями."""
    def __str__(self):
        return "\nCouldn't find kmeans model file"


class KMeansSeparator:
    """Модель KMeans для разделения людей в кадре по группам"""
    def __init__(self, train_data_path: str | None):
        if train_data_path is not None:
            self.train_data = torch.load(train_data_path)
        else:
            self.train_data = None

    def kmeans_fit(self, n_clusters: int) -> KMeans:
        """Предобучение модели"""
        kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
        if self.train_data is None:
            raise TrainDataNotTransmittedException
        kmeans_model.fit(self.train_data)
        return kmeans_model

    @staticmethod
    def kmeans_save(kmeans_model, model_name: str) -> None:
        """Сохранение модели в директорию с моделями"""
        kmeans_models_folder = os.path.join(Path(__file__).resolve().parents[1], 'models', 'kmeans_models')
        if not os.path.exists(kmeans_models_folder):
            os.mkdir(kmeans_models_folder)
        with open(os.path.join(kmeans_models_folder, model_name), 'wb') as f:
            pickle.dump(kmeans_model, f)

    @staticmethod
    def kmeans_load(model_name: str) -> KMeans:
        """Подгрузка модели из директории с моделями"""
        kmeans_model_path = os.path.join(Path(__file__).resolve().parents[1], 'models', 'kmeans_models', model_name)
        if not os.path.exists(kmeans_model_path):
            raise ModelNotFoundException
        with open(kmeans_model_path, 'rb') as f:
            model = pickle.load(f)
        return model
