# 🤟 SignBridge

**Система распознавания русского жестового языка (РЖЯ) в реальном времени**

Проект для конкурса "Большие вызовы" | Направление: Большие данные, ИИ, автоматизированные системы

---

## 🎯 О проекте

SignBridge — это система для двусторонней коммуникации:
- **Жесты → Текст/Речь** — распознавание РЖЯ через камеру
- **Текст → Жесты** — анимированный аватар (v2.0)

### Возможности
- Распознавание **1000 жестов** из датасета Slovo
- Распознавание **33 букв** дактильной азбуки (Bukva)
- Работа **в реальном времени** через веб-камеру
- **Озвучка** на русском языке (Silero TTS)
- Работа **оффлайн** — всё на устройстве

---

## 🚀 Быстрый старт

### Требования
- macOS 12+ (Apple Silicon или Intel)
- Python 3.10-3.11
- Веб-камера

### Установка

```bash
# 1. Клонирование/переход в папку проекта
cd ~/Documents/big-calls

# 2. Запуск установочного скрипта
chmod +x setup.sh
./setup.sh

# 3. Активация окружения
source venv/bin/activate
```

### Запуск

```bash
# Тест извлечения ключевых точек (веб-камера)
python src/landmark_extractor.py

# Веб-интерфейс
streamlit run app/streamlit_app.py
```

---

## 📁 Структура проекта

```
signbridge/
├── data/
│   ├── slovo/                  # Датасет Slovo (1000 жестов)
│   ├── bukva/                  # Датасет Bukva (33 буквы)
│   └── models/                 # Предобученные модели
├── src/
│   ├── landmark_extractor.py   # Извлечение ключевых точек MediaPipe
│   ├── model.py                # Архитектуры нейросетей
│   ├── dataset.py              # Загрузчики данных
│   ├── train.py                # Обучение
│   ├── inference.py            # Инференс в реальном времени
│   └── tts.py                  # Text-to-Speech
├── app/
│   └── streamlit_app.py        # Веб-приложение
├── requirements.txt
├── setup.sh                    # Скрипт установки
└── SIGNBRIDGE_GUIDE.md         # Подробная документация
```

---

## 📊 Датасеты

| Датасет | Классы | Видео | Размер | Ссылка |
|---------|--------|-------|--------|--------|
| **Slovo** | 1000 жестов | 20,400 | 16 GB | [GitHub](https://github.com/hukenovs/slovo) |
| **Bukva** | 33 буквы | 3,757 | 27 GB | [GitHub](https://github.com/ai-forever/bukva) |

### Скачивание

```bash
# MediaPipe landmarks (быстрый старт, 1.2 GB)
curl -o data/slovo/slovo_mediapipe.json \
  https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo_mediapipe.json

# Полный датасет Slovo (16 GB)
curl -o data/slovo/slovo.zip \
  https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo.zip
```

---

## 🧠 Модели

### Предобученные
| Модель | Задача | Точность | Размер |
|--------|--------|----------|--------|
| MobileNetV2_TSM | Дактиль (33 буквы) | 83.6% | 9 MB |
| MViTv2 32-2 | 1000 жестов | 64.1% | 140 MB |
| SignFlow-R | 267 жестов + GigaChat | — | 37M params |

### Обучение своей модели

```bash
# Дактиль (33 буквы)
python src/train.py \
  --dataset bukva \
  --model lstm \
  --epochs 50 \
  --batch-size 32

# 1000 жестов
python src/train.py \
  --dataset slovo \
  --model cnn_lstm \
  --epochs 100
```

---

## 🛠 Технологии

| Компонент | Технология |
|-----------|-----------|
| Извлечение landmarks | MediaPipe Holistic |
| Обучение моделей | PyTorch |
| Инференс | ONNX Runtime |
| Text-to-Speech | Silero TTS |
| Веб-интерфейс | Streamlit |
| Компьютерное зрение | OpenCV |

---

## 📖 Документация

Подробная документация: [`SIGNBRIDGE_GUIDE.md`](SIGNBRIDGE_GUIDE.md)

- Все ссылки на датасеты и модели
- Инструкции по установке для macOS
- Описание архитектуры
- Примеры кода

---

## 📈 Результаты

| Метрика | Значение |
|---------|----------|
| Точность (дактиль) | 83.6% |
| Точность (1000 жестов) | 64.1% |
| FPS (MacBook M1) | 25-30 |
| Latency | <100ms |

---

## 🗓 Roadmap

- [x] Извлечение landmarks (MediaPipe)
- [x] Модели для классификации (LSTM, Transformer, CNN+LSTM)
- [x] Веб-интерфейс (Streamlit)
- [x] Text-to-Speech (Silero)
- [ ] 3D-аватар для обратного перевода
- [ ] Мобильное приложение (Flutter)
- [ ] Режим диалога

---

## 👥 Авторы

Проект создан для конкурса **"Большие вызовы"**

---

## 📄 Лицензия

MIT License

Датасеты Slovo и Bukva: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

---

## 🙏 Благодарности

- [Slovo Dataset](https://github.com/hukenovs/slovo) — Сбер AI
- [Bukva Dataset](https://github.com/ai-forever/bukva) — Сбер AI
- [MediaPipe](https://mediapipe.dev/) — Google
- [Silero Models](https://github.com/snakers4/silero-models) — Silero Team
