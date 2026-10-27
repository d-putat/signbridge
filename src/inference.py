"""
SignBridge - Инференс в реальном времени
"""

import cv2
import torch
import numpy as np
from collections import deque
from typing import Optional, Tuple, List
from pathlib import Path
import onnxruntime as ort

from landmark_extractor import LandmarkExtractor


class SignLanguageRecognizer:
    """Распознавание жестов в реальном времени"""
    
    def __init__(
        self,
        model_path: str,
        class_names: List[str],
        seq_length: int = 30,
        confidence_threshold: float = 0.5,
        use_onnx: bool = True,
        device: str = "auto"
    ):
        """
        Args:
            model_path: Путь к модели (.pt или .onnx)
            class_names: Список названий классов
            seq_length: Длина последовательности
            confidence_threshold: Порог уверенности
            use_onnx: Использовать ONNX Runtime
            device: Устройство для PyTorch модели
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.seq_length = seq_length
        self.confidence_threshold = confidence_threshold
        self.use_onnx = use_onnx
        
        # Буфер для накопления кадров
        self.frame_buffer = deque(maxlen=seq_length)
        
        # Извлечение landmarks
        self.extractor = LandmarkExtractor()
        
        # Загрузка модели
        if use_onnx:
            self._load_onnx_model(model_path)
        else:
            self._load_pytorch_model(model_path, device)
        
        # Сглаживание предсказаний
        self.prediction_history = deque(maxlen=5)
        
        print(f"Model loaded: {model_path}")
        print(f"Classes: {self.num_classes}")
    
    def _load_onnx_model(self, model_path: str):
        """Загрузка ONNX модели"""
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
    
    def _load_pytorch_model(self, model_path: str, device: str):
        """Загрузка PyTorch модели"""
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Импортируем модель
        from model import get_model
        
        config = checkpoint.get('config', {})
        self.model = get_model(
            model_type=config.get('model_type', 'lstm'),
            input_size=config.get('input_size', 159),
            num_classes=self.num_classes,
            hidden_size=config.get('hidden_size', 256)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[str], float, np.ndarray]:
        """
        Обработка одного кадра
        
        Args:
            frame: BGR изображение
            
        Returns:
            (predicted_class, confidence, annotated_frame)
        """
        # Извлечение landmarks
        data = self.extractor.extract_from_frame(frame)
        features = self.extractor.to_feature_vector(data)
        
        # Добавление в буфер
        self.frame_buffer.append(features)
        
        # Отрисовка landmarks
        annotated = self.extractor.draw_landmarks(frame, data)
        
        # Предсказание только если буфер заполнен
        if len(self.frame_buffer) < self.seq_length:
            remaining = self.seq_length - len(self.frame_buffer)
            cv2.putText(
                annotated,
                f"Collecting frames: {len(self.frame_buffer)}/{self.seq_length}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )
            return None, 0.0, annotated
        
        # Предсказание
        predicted_class, confidence = self._predict()
        
        # Сглаживание
        self.prediction_history.append((predicted_class, confidence))
        smoothed_class, smoothed_conf = self._smooth_predictions()
        
        # Отображение результата
        if smoothed_conf >= self.confidence_threshold:
            color = (0, 255, 0)  # Зелёный
            text = f"{smoothed_class}: {smoothed_conf:.1%}"
        else:
            color = (0, 255, 255)  # Жёлтый
            text = f"? ({smoothed_class}: {smoothed_conf:.1%})"
        
        cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return smoothed_class if smoothed_conf >= self.confidence_threshold else None, smoothed_conf, annotated
    
    def _predict(self) -> Tuple[str, float]:
        """Предсказание на основе буфера"""
        # Подготовка данных
        sequence = np.array(list(self.frame_buffer), dtype=np.float32)
        sequence = sequence[np.newaxis, ...]  # (1, seq_len, features)
        
        if self.use_onnx:
            outputs = self.session.run(None, {self.input_name: sequence})
            logits = outputs[0][0]
        else:
            with torch.no_grad():
                x = torch.FloatTensor(sequence).to(self.device)
                logits = self.model(x)[0].cpu().numpy()
        
        # Softmax
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        # Лучший класс
        idx = np.argmax(probs)
        confidence = probs[idx]
        predicted_class = self.class_names[idx]
        
        return predicted_class, confidence
    
    def _smooth_predictions(self) -> Tuple[str, float]:
        """Сглаживание предсказаний"""
        if len(self.prediction_history) < 3:
            return self.prediction_history[-1]
        
        # Голосование
        class_counts = {}
        class_confs = {}
        
        for cls, conf in self.prediction_history:
            if cls not in class_counts:
                class_counts[cls] = 0
                class_confs[cls] = []
            class_counts[cls] += 1
            class_confs[cls].append(conf)
        
        # Выбор наиболее частого класса
        best_class = max(class_counts, key=class_counts.get)
        avg_conf = np.mean(class_confs[best_class])
        
        return best_class, avg_conf
    
    def reset(self):
        """Сброс буфера"""
        self.frame_buffer.clear()
        self.prediction_history.clear()


def run_realtime_recognition(
    model_path: str,
    class_names: List[str],
    camera_id: int = 0,
    seq_length: int = 30,
    confidence_threshold: float = 0.5
):
    """
    Запуск распознавания в реальном времени
    
    Args:
        model_path: Путь к модели
        class_names: Список классов
        camera_id: ID камеры
        seq_length: Длина последовательности
        confidence_threshold: Порог уверенности
    """
    recognizer = SignLanguageRecognizer(
        model_path=model_path,
        class_names=class_names,
        seq_length=seq_length,
        confidence_threshold=confidence_threshold
    )
    
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Press 'q' to quit, 'r' to reset buffer")
    
    recognized_text = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Зеркальное отражение для удобства
        frame = cv2.flip(frame, 1)
        
        # Обработка
        predicted, confidence, annotated = recognizer.process_frame(frame)
        
        # Добавление в текст
        if predicted and confidence >= confidence_threshold:
            if not recognized_text or recognized_text[-1] != predicted:
                recognized_text.append(predicted)
        
        # Отображение распознанного текста
        text = " ".join(recognized_text[-10:])  # Последние 10 символов
        cv2.putText(
            annotated,
            f"Text: {text}",
            (10, annotated.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        cv2.imshow('SignBridge', annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recognizer.reset()
            recognized_text.clear()
            print("Buffer reset")
    
    cap.release()
    cv2.destroyAllWindows()
    
    return " ".join(recognized_text)


# === ТЕСТИРОВАНИЕ ===

if __name__ == "__main__":
    # Дактильная азбука
    DACTYL_CLASSES = list("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")
    
    print("SignBridge Real-time Recognition")
    print("="*40)
    print("Для тестирования нужна обученная модель.")
    print("Запустите: python train.py --dataset bukva")
    print()
    print("Или используйте предобученную модель:")
    print("  data/models/MobileNetV2_TSM.onnx")
    print()
    
    # Пример с предобученной моделью (если есть)
    model_path = "data/models/MobileNetV2_TSM.onnx"
    
    if Path(model_path).exists():
        run_realtime_recognition(
            model_path=model_path,
            class_names=DACTYL_CLASSES,
            confidence_threshold=0.6
        )
    else:
        print(f"Model not found: {model_path}")
        print("Download it first:")
        print(f"  curl -o {model_path} https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/bukva/models/MobileNetV2_TSM.onnx")
