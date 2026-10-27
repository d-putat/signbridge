# SignBridge — Полное техническое руководство для macOS

## 📦 Все ресурсы и технологии

---

## 1. ДАТАСЕТЫ

### 1.1 Slovo — Основной датасет (1000 жестов)
**Что это:** 20,400 видео с 1000 жестами РЖЯ от 194 человек

| Параметр | Значение |
|----------|----------|
| Размер | ~16 GB (основной) / ~13 GB (360p версия) |
| Видео | 20,400 RGB видео |
| Классы | 1000 жестов |
| Формат | HD/FullHD |
| Лицензия | CC BY-SA 4.0 |

**Ссылки для скачивания:**
```
# Основной датасет (16 GB) — обрезанные видео
https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo.zip

# Лёгкая версия 360p (13 GB) — для быстрых экспериментов
https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo_full360.zip

# Готовые MediaPipe landmarks (1.2 GB) — уже извлечённые ключевые точки!
https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo_mediapipe.json

# Kaggle (альтернатива)
https://www.kaggle.com/datasets/kapitanov/slovo
```

**GitHub:** https://github.com/hukenovs/slovo

---

### 1.2 Bukva — Датасет дактильной азбуки (33 буквы)
**Что это:** 3,757 видео с буквами русской дактильной азбуки

| Параметр | Значение |
|----------|----------|
| Размер | ~27 GB |
| Видео | 3,757 RGB видео |
| Классы | 33 буквы (А-Я) |
| Формат | HD/FullHD |
| Лицензия | CC BY-SA 4.0 |

**Ссылка для скачивания:**
```
https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/bukva/bukva.zip
```

**GitHub:** https://github.com/ai-forever/bukva

---

### 1.3 Готовые предобученные модели

**Для Slovo (1000 жестов):**
```
# MViTv2 — лучшая точность (64.09%)
ONNX: https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/mvit/onnx/mvit32-2.onnx
PyTorch: https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/mvit/pt/mvit32-2.pt

# SignFlow-R — 267 классов, протестировано с GigaChat
ONNX: https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/SignFlow-R.onnx
```

**Для Bukva (дактиль):**
```
# MobileNetV2_TSM — 83.6% точность, работает на CPU
ONNX: https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/bukva/models/MobileNetV2_TSM.onnx
```

---

## 2. УСТАНОВКА ОКРУЖЕНИЯ НА macOS

### 2.1 Установка Python и зависимостей

```bash
# Установка Homebrew (если нет)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Установка Python 3.11 (рекомендуемая версия)
brew install python@3.11

# Создание виртуального окружения
cd ~/Documents/big-calls
python3.11 -m venv venv
source venv/bin/activate

# Обновление pip
pip install --upgrade pip
```

### 2.2 Установка основных библиотек

```bash
# Основные ML библиотеки
pip install torch torchvision torchaudio

# MediaPipe для извлечения ключевых точек
pip install mediapipe

# Компьютерное зрение
pip install opencv-python opencv-python-headless

# Работа с данными
pip install numpy pandas

# Визуализация
pip install matplotlib seaborn

# Работа с видео
pip install decord av

# ONNX для инференса готовых моделей
pip install onnx onnxruntime

# Text-to-Speech (русский)
pip install silero-tts torchaudio omegaconf

# Веб-интерфейс для демо
pip install streamlit

# Jupyter для экспериментов
pip install jupyter jupyterlab

# Утилиты
pip install tqdm pyyaml
```

### 2.3 Проверка установки

```bash
python -c "
import torch
import mediapipe as mp
import cv2
import onnxruntime

print(f'PyTorch: {torch.__version__}')
print(f'MediaPipe: {mp.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'ONNX Runtime: {onnxruntime.__version__}')
print(f'MPS (Apple Silicon) available: {torch.backends.mps.is_available()}')
"
```

---

## 3. СТРУКТУРА ПРОЕКТА

```
signbridge/
├── data/
│   ├── slovo/                  # Датасет Slovo
│   │   ├── videos/             # Видео файлы
│   │   └── annotations.csv     # Аннотации
│   ├── bukva/                  # Датасет Bukva
│   │   ├── trimmed/            # Обрезанные видео
│   │   └── annotations.tsv     # Аннотации
│   └── models/                 # Предобученные модели
│       ├── mvit32-2.onnx
│       ├── SignFlow-R.onnx
│       └── MobileNetV2_TSM.onnx
├── src/
│   ├── __init__.py
│   ├── landmark_extractor.py   # Извлечение ключевых точек
│   ├── model.py                # Архитектура нейросети
│   ├── dataset.py              # Загрузчик данных
│   ├── train.py                # Обучение модели
│   ├── inference.py            # Инференс
│   └── tts.py                  # Text-to-Speech
├── app/
│   └── streamlit_app.py        # Веб-интерфейс
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_landmark_extraction.ipynb
│   └── 03_model_training.ipynb
├── requirements.txt
├── config.yaml
└── README.md
```

---

## 4. БИБЛИОТЕКИ И ИХ НАЗНАЧЕНИЕ

### 4.1 MediaPipe — Извлечение ключевых точек

**Что делает:** Извлекает 543 ключевые точки (руки, поза, лицо) из видео в реальном времени

```python
import mediapipe as mp
import cv2

# Инициализация
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Обработка кадра
cap = cv2.VideoCapture(0)  # Веб-камера
ret, frame = cap.read()
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = holistic.process(rgb_frame)

# Получение ключевых точек
if results.left_hand_landmarks:
    for lm in results.left_hand_landmarks.landmark:
        print(f"x={lm.x}, y={lm.y}, z={lm.z}")
```

**Ключевые точки:**
- `pose_landmarks`: 33 точки тела
- `face_landmarks`: 468 точек лица
- `left_hand_landmarks`: 21 точка левой руки
- `right_hand_landmarks`: 21 точка правой руки

---

### 4.2 PyTorch — Обучение моделей

**Что делает:** Обучение нейросетей (LSTM, Transformer) на последовательностях ключевых точек

```python
import torch
import torch.nn as nn

# Проверка Apple Silicon
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Пример модели
class SignLanguageModel(nn.Module):
    def __init__(self, input_size=126, hidden_size=256, num_classes=1000):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, 
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out
```

---

### 4.3 ONNX Runtime — Инференс готовых моделей

**Что делает:** Быстрый запуск предобученных моделей без PyTorch

```python
import onnxruntime as ort
import numpy as np

# Загрузка модели
session = ort.InferenceSession("models/SignFlow-R.onnx")

# Подготовка входных данных
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"Input shape: {input_shape}")

# Инференс
dummy_input = np.random.randn(1, 32, 224, 224, 3).astype(np.float32)
outputs = session.run(None, {input_name: dummy_input})
predictions = outputs[0]
```

---

### 4.4 Silero TTS — Озвучка на русском

**Что делает:** Превращает распознанный текст в речь

```python
import torch

# Загрузка модели
device = torch.device('cpu')
model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='ru',
    speaker='v4_ru'  # Есть несколько голосов
)
model.to(device)

# Генерация речи
text = "Привет, как дела?"
audio = model.apply_tts(text=text, speaker='baya', sample_rate=48000)

# Сохранение в файл
import torchaudio
torchaudio.save('output.wav', audio.unsqueeze(0), 48000)
```

**Доступные голоса (русский):**
- `aidar` — мужской
- `baya` — женский
- `kseniya` — женский
- `xenia` — женский
- `eugene` — мужской
- `random` — случайный

---

### 4.5 OpenCV — Работа с видео и камерой

**Что делает:** Захват видео с камеры, обработка кадров

```python
import cv2

# Захват с веб-камеры
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Обработка кадра
    cv2.imshow('SignBridge', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

### 4.6 Streamlit — Веб-интерфейс

**Что делает:** Быстрое создание веб-приложения для демонстрации

```python
import streamlit as st

st.title("SignBridge Demo")

# Виджет камеры
camera_input = st.camera_input("Покажите жест")

if camera_input:
    # Обработка изображения
    st.image(camera_input)
    st.write("Распознанный жест: Привет")
```

Запуск:
```bash
streamlit run app/streamlit_app.py
```

---

## 5. СЛОВАРЬ ЖЕСТОВ

### 5.1 Slovo — 1000 жестов

Полный список классов находится в `annotations.csv`. Примеры:
```
привет, спасибо, пожалуйста, да, нет, хорошо, плохо, 
я, ты, мы, они, мама, папа, дом, работа, школа,
есть, пить, спать, идти, видеть, слышать, понимать...
```

### 5.2 Bukva — 33 буквы дактиля

```
А, Б, В, Г, Д, Е, Ё, Ж, З, И, Й, К, Л, М, Н, О, П, 
Р, С, Т, У, Ф, Х, Ц, Ч, Ш, Щ, Ъ, Ы, Ь, Э, Ю, Я
```

---

## 6. БЫСТРЫЙ СТАРТ

### Шаг 1: Скачивание данных

```bash
# Создание директорий
mkdir -p data/slovo data/bukva data/models

# Скачивание MediaPipe landmarks (самый быстрый старт!)
curl -o data/slovo/slovo_mediapipe.json \
  https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo_mediapipe.json

# Скачивание предобученной модели для дактиля
curl -o data/models/MobileNetV2_TSM.onnx \
  https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/bukva/models/MobileNetV2_TSM.onnx

# Скачивание модели для жестов (опционально, 140 MB)
curl -o data/models/mvit32-2.onnx \
  https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/mvit/onnx/mvit32-2.onnx
```

### Шаг 2: Тест MediaPipe

```bash
python src/test_mediapipe.py
```

### Шаг 3: Запуск демо

```bash
streamlit run app/streamlit_app.py
```

---

## 7. ПОЛЕЗНЫЕ ССЫЛКИ

### Датасеты
| Название | Ссылка |
|----------|--------|
| Slovo (GitHub) | https://github.com/hukenovs/slovo |
| Slovo (Kaggle) | https://www.kaggle.com/datasets/kapitanov/slovo |
| Bukva (GitHub) | https://github.com/ai-forever/bukva |
| Slovo (arXiv) | https://arxiv.org/abs/2305.14527 |
| Bukva (arXiv) | https://arxiv.org/abs/2410.08675 |

### Документация библиотек
| Библиотека | Документация |
|------------|--------------|
| MediaPipe Hands | https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python |
| MediaPipe Holistic | https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker |
| PyTorch | https://pytorch.org/docs/stable/index.html |
| ONNX Runtime | https://onnxruntime.ai/docs/ |
| Silero TTS | https://github.com/snakers4/silero-models |
| Streamlit | https://docs.streamlit.io/ |

### Похожие проекты для вдохновения
| Проект | Ссылка |
|--------|--------|
| Real-Time Sign Language (LSTM) | https://github.com/uni-creator/real-time-sign-language-recognition |
| Sign Language Detection (DTW) | https://github.com/ritam855/sign-language-detection-using-lstm-and-dtw |
| MediaPipe Sign Language | https://github.com/sumanthpagadala/Sign_Language |

---

## 8. ПЛАН ДЕЙСТВИЙ

### Неделя 1: Базовый прототип
- [ ] Установить окружение
- [ ] Скачать slovo_mediapipe.json
- [ ] Написать скрипт извлечения landmarks с камеры
- [ ] Обучить простую LSTM на дактиле (33 класса)

### Неделя 2: Улучшение модели
- [ ] Скачать полный датасет Bukva
- [ ] Обучить модель на 100+ жестах
- [ ] Добавить TTS озвучку

### Неделя 3: Интерфейс
- [ ] Создать Streamlit приложение
- [ ] Добавить режим реального времени
- [ ] Тестирование и отладка

### Неделя 4: Полировка
- [ ] Оптимизация производительности
- [ ] Документация
- [ ] Подготовка презентации
