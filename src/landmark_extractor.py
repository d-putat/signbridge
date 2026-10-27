"""
SignBridge - Модуль извлечения ключевых точек с помощью MediaPipe
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class LandmarkData:
    """Структура для хранения ключевых точек"""
    left_hand: Optional[np.ndarray] = None   # (21, 3)
    right_hand: Optional[np.ndarray] = None  # (21, 3)
    pose: Optional[np.ndarray] = None        # (33, 3)
    face: Optional[np.ndarray] = None        # (468, 3) или None для экономии


class LandmarkExtractor:
    """Извлечение ключевых точек из изображений/видео"""
    
    def __init__(
        self, 
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        include_face: bool = False
    ):
        """
        Args:
            min_detection_confidence: Минимальная уверенность детекции
            min_tracking_confidence: Минимальная уверенность трекинга
            include_face: Включать ли точки лица (увеличивает размер данных)
        """
        self.include_face = include_face
        
        # Инициализация MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def extract_from_frame(self, frame: np.ndarray) -> LandmarkData:
        """
        Извлечение ключевых точек из одного кадра
        
        Args:
            frame: BGR изображение (numpy array)
            
        Returns:
            LandmarkData с извлечёнными точками
        """
        # Конвертация BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Обработка
        results = self.holistic.process(rgb_frame)
        
        # Извлечение данных
        data = LandmarkData()
        
        # Левая рука (21 точка × 3 координаты)
        if results.left_hand_landmarks:
            data.left_hand = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in results.left_hand_landmarks.landmark
            ])
        
        # Правая рука
        if results.right_hand_landmarks:
            data.right_hand = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in results.right_hand_landmarks.landmark
            ])
        
        # Поза (33 точки)
        if results.pose_landmarks:
            data.pose = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in results.pose_landmarks.landmark
            ])
        
        # Лицо (опционально)
        if self.include_face and results.face_landmarks:
            data.face = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in results.face_landmarks.landmark
            ])
        
        return data
    
    def to_feature_vector(self, data: LandmarkData, include_pose: bool = True) -> np.ndarray:
        """
        Преобразование LandmarkData в плоский вектор признаков
        
        Args:
            data: Данные ключевых точек
            include_pose: Включать ли точки позы
            
        Returns:
            Numpy array с признаками (заполнен нулями если точки не найдены)
        """
        features = []
        
        # Руки (21 × 3 × 2 = 126 признаков)
        if data.left_hand is not None:
            features.extend(data.left_hand.flatten())
        else:
            features.extend([0.0] * 63)
        
        if data.right_hand is not None:
            features.extend(data.right_hand.flatten())
        else:
            features.extend([0.0] * 63)
        
        # Поза - только верхняя часть тела (11 точек × 3 = 33 признака)
        if include_pose:
            if data.pose is not None:
                # Берём только верхнюю часть: плечи, локти, запястья, лицо
                upper_body_indices = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
                upper_body = data.pose[upper_body_indices]
                features.extend(upper_body.flatten())
            else:
                features.extend([0.0] * 33)
        
        return np.array(features, dtype=np.float32)
    
    def draw_landmarks(self, frame: np.ndarray, data: LandmarkData) -> np.ndarray:
        """
        Отрисовка ключевых точек на кадре
        
        Args:
            frame: Исходный кадр
            data: Данные ключевых точек
            
        Returns:
            Кадр с нарисованными точками
        """
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Рисуем руки
        for hand_data, color in [(data.left_hand, (255, 0, 0)), (data.right_hand, (0, 0, 255))]:
            if hand_data is not None:
                for point in hand_data:
                    x, y = int(point[0] * w), int(point[1] * h)
                    cv2.circle(annotated_frame, (x, y), 5, color, -1)
        
        # Рисуем позу
        if data.pose is not None:
            for point in data.pose[:17]:  # Верхняя часть тела
                x, y = int(point[0] * w), int(point[1] * h)
                cv2.circle(annotated_frame, (x, y), 4, (0, 255, 0), -1)
        
        return annotated_frame
    
    def __del__(self):
        """Освобождение ресурсов"""
        if hasattr(self, 'holistic'):
            self.holistic.close()


class VideoLandmarkExtractor:
    """Извлечение ключевых точек из видеофайла"""
    
    def __init__(self, extractor: Optional[LandmarkExtractor] = None):
        self.extractor = extractor or LandmarkExtractor()
    
    def extract_from_video(
        self, 
        video_path: str, 
        max_frames: Optional[int] = None,
        sample_rate: int = 1
    ) -> List[LandmarkData]:
        """
        Извлечение ключевых точек из видеофайла
        
        Args:
            video_path: Путь к видеофайлу
            max_frames: Максимальное количество кадров (None = все)
            sample_rate: Брать каждый N-й кадр
            
        Returns:
            Список LandmarkData для каждого кадра
        """
        cap = cv2.VideoCapture(video_path)
        landmarks_list = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                data = self.extractor.extract_from_frame(frame)
                landmarks_list.append(data)
                
                if max_frames and len(landmarks_list) >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        return landmarks_list
    
    def to_feature_sequence(
        self, 
        landmarks_list: List[LandmarkData],
        fixed_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Преобразование списка landmarks в матрицу признаков
        
        Args:
            landmarks_list: Список LandmarkData
            fixed_length: Фиксированная длина последовательности (паддинг/обрезка)
            
        Returns:
            Numpy array shape (seq_len, features)
        """
        features = [self.extractor.to_feature_vector(data) for data in landmarks_list]
        sequence = np.array(features)
        
        if fixed_length:
            if len(sequence) > fixed_length:
                # Обрезаем
                sequence = sequence[:fixed_length]
            elif len(sequence) < fixed_length:
                # Паддинг нулями
                padding = np.zeros((fixed_length - len(sequence), sequence.shape[1]))
                sequence = np.vstack([sequence, padding])
        
        return sequence


# === ТЕСТИРОВАНИЕ ===

def test_webcam():
    """Тест извлечения landmarks с веб-камеры"""
    extractor = LandmarkExtractor()
    cap = cv2.VideoCapture(0)
    
    print("Нажмите 'q' для выхода")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Извлечение точек
        data = extractor.extract_from_frame(frame)
        
        # Получение вектора признаков
        features = extractor.to_feature_vector(data)
        
        # Отрисовка
        annotated = extractor.draw_landmarks(frame, data)
        
        # Информация
        hands_detected = (data.left_hand is not None) + (data.right_hand is not None)
        cv2.putText(
            annotated, 
            f"Hands: {hands_detected}, Features: {len(features)}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        cv2.imshow('SignBridge - Landmark Test', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_webcam()
