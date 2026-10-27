"""
SignBridge - Загрузчики данных для обучения моделей
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, List, Tuple, Callable
from pathlib import Path
import cv2
from tqdm import tqdm

from landmark_extractor import LandmarkExtractor, VideoLandmarkExtractor


class SlovoLandmarksDataset(Dataset):
    """
    Датасет Slovo с готовыми MediaPipe landmarks
    
    Использует предварительно извлечённые ключевые точки из файла
    slovo_mediapipe.json (1.2 GB) — самый быстрый способ начать.
    """
    
    def __init__(
        self,
        landmarks_json_path: str,
        annotations_csv_path: str,
        train: bool = True,
        seq_length: int = 30,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            landmarks_json_path: Путь к slovo_mediapipe.json
            annotations_csv_path: Путь к annotations.csv
            train: True для обучающей выборки, False для тестовой
            seq_length: Фиксированная длина последовательности
            transform: Опциональные аугментации
        """
        self.seq_length = seq_length
        self.transform = transform
        
        # Загрузка аннотаций
        print("Loading annotations...")
        self.annotations = pd.read_csv(annotations_csv_path)
        self.annotations = self.annotations[self.annotations['train'] == train]
        
        # Создание словаря классов
        self.classes = sorted(self.annotations['text'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        print(f"Found {len(self.classes)} classes, {len(self.annotations)} samples")
        
        # Загрузка landmarks
        print("Loading landmarks (this may take a while)...")
        with open(landmarks_json_path, 'r') as f:
            self.landmarks_data = json.load(f)
        print(f"Loaded landmarks for {len(self.landmarks_data)} videos")
        
        # Фильтрация - только те видео, для которых есть landmarks
        valid_ids = set(self.landmarks_data.keys())
        self.annotations = self.annotations[
            self.annotations['attachment_id'].isin(valid_ids)
        ].reset_index(drop=True)
        
        print(f"Final dataset size: {len(self.annotations)} samples")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.annotations.iloc[idx]
        video_id = row['attachment_id']
        label = self.class_to_idx[row['text']]
        
        # Получение landmarks для видео
        landmarks = self.landmarks_data[video_id]
        
        # Преобразование в numpy
        features = self._process_landmarks(landmarks)
        
        # Паддинг/обрезка до фиксированной длины
        features = self._pad_or_truncate(features)
        
        # Аугментации
        if self.transform:
            features = self.transform(features)
        
        return torch.FloatTensor(features), label
    
    def _process_landmarks(self, landmarks: List[Dict]) -> np.ndarray:
        """Преобразование landmarks в матрицу признаков"""
        features = []
        
        for frame_data in landmarks:
            frame_features = []
            
            # Левая рука (21 × 3 = 63)
            if 'left_hand' in frame_data and frame_data['left_hand']:
                frame_features.extend(np.array(frame_data['left_hand']).flatten())
            else:
                frame_features.extend([0.0] * 63)
            
            # Правая рука (21 × 3 = 63)
            if 'right_hand' in frame_data and frame_data['right_hand']:
                frame_features.extend(np.array(frame_data['right_hand']).flatten())
            else:
                frame_features.extend([0.0] * 63)
            
            # Поза - верхняя часть тела (11 × 3 = 33)
            if 'pose' in frame_data and frame_data['pose']:
                pose = np.array(frame_data['pose'])
                # Индексы верхней части тела
                upper_indices = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
                if len(pose) > max(upper_indices):
                    upper_body = pose[upper_indices].flatten()
                    frame_features.extend(upper_body)
                else:
                    frame_features.extend([0.0] * 33)
            else:
                frame_features.extend([0.0] * 33)
            
            features.append(frame_features)
        
        return np.array(features, dtype=np.float32)
    
    def _pad_or_truncate(self, features: np.ndarray) -> np.ndarray:
        """Приведение к фиксированной длине"""
        current_len = len(features)
        
        if current_len > self.seq_length:
            # Равномерная выборка кадров
            indices = np.linspace(0, current_len - 1, self.seq_length, dtype=int)
            return features[indices]
        elif current_len < self.seq_length:
            # Паддинг нулями
            padding = np.zeros((self.seq_length - current_len, features.shape[1]))
            return np.vstack([features, padding])
        else:
            return features
    
    def get_class_name(self, idx: int) -> str:
        """Получение названия класса по индексу"""
        return self.idx_to_class[idx]


class BukvaDataset(Dataset):
    """
    Датасет Bukva для дактильной азбуки (33 буквы)
    
    Извлекает landmarks из видео в реальном времени.
    """
    
    def __init__(
        self,
        data_dir: str,
        annotations_path: str,
        train: bool = True,
        seq_length: int = 30,
        transform: Optional[Callable] = None,
        cache_landmarks: bool = True
    ):
        """
        Args:
            data_dir: Путь к папке с видео (trimmed/)
            annotations_path: Путь к annotations.tsv
            train: True для обучающей выборки
            seq_length: Фиксированная длина последовательности
            transform: Аугментации
            cache_landmarks: Кэшировать извлечённые landmarks
        """
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length
        self.transform = transform
        self.cache_landmarks = cache_landmarks
        self.landmarks_cache = {}
        
        # Загрузка аннотаций
        self.annotations = pd.read_csv(annotations_path, sep='\t')
        self.annotations = self.annotations[self.annotations['train'] == train]
        self.annotations = self.annotations.reset_index(drop=True)
        
        # Словарь классов (буквы А-Я)
        self.classes = sorted(self.annotations['text'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Инициализация экстрактора
        self.extractor = LandmarkExtractor()
        self.video_extractor = VideoLandmarkExtractor(self.extractor)
        
        print(f"Bukva dataset: {len(self.annotations)} samples, {len(self.classes)} classes")
        print(f"Classes: {self.classes}")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.annotations.iloc[idx]
        video_id = row['attachment_id']
        label = self.class_to_idx[row['text']]
        
        # Проверка кэша
        if self.cache_landmarks and video_id in self.landmarks_cache:
            features = self.landmarks_cache[video_id]
        else:
            # Извлечение landmarks из видео
            video_path = self.data_dir / f"{video_id}.mp4"
            
            if not video_path.exists():
                # Пробуем другие расширения
                for ext in ['.avi', '.mov', '.webm']:
                    alt_path = self.data_dir / f"{video_id}{ext}"
                    if alt_path.exists():
                        video_path = alt_path
                        break
            
            landmarks_list = self.video_extractor.extract_from_video(str(video_path))
            features = self.video_extractor.to_feature_sequence(landmarks_list)
            
            if self.cache_landmarks:
                self.landmarks_cache[video_id] = features
        
        # Паддинг/обрезка
        features = self._pad_or_truncate(features)
        
        if self.transform:
            features = self.transform(features)
        
        return torch.FloatTensor(features), label
    
    def _pad_or_truncate(self, features: np.ndarray) -> np.ndarray:
        """Приведение к фиксированной длине"""
        current_len = len(features)
        
        if current_len > self.seq_length:
            indices = np.linspace(0, current_len - 1, self.seq_length, dtype=int)
            return features[indices]
        elif current_len < self.seq_length:
            padding = np.zeros((self.seq_length - current_len, features.shape[1]))
            return np.vstack([features, padding])
        else:
            return features
    
    def get_class_name(self, idx: int) -> str:
        return self.idx_to_class[idx]


class SignLanguageAugmentation:
    """Аугментации для данных жестового языка"""
    
    def __init__(
        self,
        noise_std: float = 0.01,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        time_shift_range: Tuple[int, int] = (-3, 3),
        dropout_prob: float = 0.1
    ):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.time_shift_range = time_shift_range
        self.dropout_prob = dropout_prob
    
    def __call__(self, features: np.ndarray) -> np.ndarray:
        """
        Args:
            features: (seq_len, num_features)
            
        Returns:
            Аугментированные features
        """
        features = features.copy()
        
        # Гауссов шум
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, features.shape)
            features = features + noise
        
        # Масштабирование
        scale = np.random.uniform(*self.scale_range)
        features = features * scale
        
        # Случайный dropout точек
        if self.dropout_prob > 0:
            mask = np.random.random(features.shape) > self.dropout_prob
            features = features * mask
        
        return features.astype(np.float32)


def create_data_loaders(
    dataset_type: str = "slovo",
    data_dir: str = "data/slovo",
    batch_size: int = 32,
    seq_length: int = 30,
    num_workers: int = 4,
    augment: bool = True
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Создание DataLoader'ов для обучения
    
    Args:
        dataset_type: "slovo" или "bukva"
        data_dir: Путь к данным
        batch_size: Размер батча
        seq_length: Длина последовательности
        num_workers: Количество воркеров
        augment: Использовать аугментации
        
    Returns:
        train_loader, test_loader, num_classes
    """
    transform = SignLanguageAugmentation() if augment else None
    
    if dataset_type == "slovo":
        train_dataset = SlovoLandmarksDataset(
            landmarks_json_path=os.path.join(data_dir, "slovo_mediapipe.json"),
            annotations_csv_path=os.path.join(data_dir, "annotations.csv"),
            train=True,
            seq_length=seq_length,
            transform=transform
        )
        test_dataset = SlovoLandmarksDataset(
            landmarks_json_path=os.path.join(data_dir, "slovo_mediapipe.json"),
            annotations_csv_path=os.path.join(data_dir, "annotations.csv"),
            train=False,
            seq_length=seq_length,
            transform=None
        )
    elif dataset_type == "bukva":
        train_dataset = BukvaDataset(
            data_dir=os.path.join(data_dir, "trimmed"),
            annotations_path=os.path.join(data_dir, "annotations.tsv"),
            train=True,
            seq_length=seq_length,
            transform=transform
        )
        test_dataset = BukvaDataset(
            data_dir=os.path.join(data_dir, "trimmed"),
            annotations_path=os.path.join(data_dir, "annotations.tsv"),
            train=False,
            seq_length=seq_length,
            transform=None
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    num_classes = len(train_dataset.classes)
    
    return train_loader, test_loader, num_classes


# === ТЕСТИРОВАНИЕ ===

if __name__ == "__main__":
    print("Testing augmentation...")
    
    aug = SignLanguageAugmentation()
    dummy_features = np.random.randn(30, 159).astype(np.float32)
    
    augmented = aug(dummy_features)
    print(f"Original shape: {dummy_features.shape}")
    print(f"Augmented shape: {augmented.shape}")
    print(f"Original mean: {dummy_features.mean():.4f}")
    print(f"Augmented mean: {augmented.mean():.4f}")
    
    print("\nAugmentation test passed!")
