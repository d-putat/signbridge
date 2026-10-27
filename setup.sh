#!/bin/bash

# SignBridge - Скрипт установки для macOS
# ==========================================

echo "🤟 SignBridge Setup Script"
echo "=========================="
echo ""

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Проверка Python
echo "📋 Проверка Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python3 не найден. Установите: brew install python@3.11${NC}"
    exit 1
fi

# Создание виртуального окружения
echo ""
echo "📦 Создание виртуального окружения..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Виртуальное окружение создано${NC}"
else
    echo -e "${YELLOW}! Виртуальное окружение уже существует${NC}"
fi

# Активация
source venv/bin/activate

# Установка зависимостей
echo ""
echo "📥 Установка зависимостей..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Зависимости установлены${NC}"
else
    echo -e "${RED}✗ Ошибка установки зависимостей${NC}"
    exit 1
fi

# Создание директорий
echo ""
echo "📁 Создание директорий..."
mkdir -p data/slovo
mkdir -p data/bukva
mkdir -p data/models
mkdir -p checkpoints
mkdir -p notebooks
echo -e "${GREEN}✓ Директории созданы${NC}"

# Скачивание данных
echo ""
echo "📊 Скачивание данных..."
echo ""

# Функция для скачивания с прогрессом
download_file() {
    local url=$1
    local output=$2
    local name=$3
    
    if [ -f "$output" ]; then
        echo -e "${YELLOW}! $name уже скачан${NC}"
    else
        echo "Скачивание $name..."
        curl -L --progress-bar -o "$output" "$url"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ $name скачан${NC}"
        else
            echo -e "${RED}✗ Ошибка скачивания $name${NC}"
        fi
    fi
}

# MediaPipe landmarks (1.2 GB) - обязательно
echo "1/3: MediaPipe Landmarks (1.2 GB) - для быстрого старта"
download_file \
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo_mediapipe.json" \
    "data/slovo/slovo_mediapipe.json" \
    "Slovo MediaPipe Landmarks"

# Предобученная модель для дактиля (9 MB)
echo ""
echo "2/3: Предобученная модель дактиля (9 MB)"
download_file \
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/bukva/models/MobileNetV2_TSM.onnx" \
    "data/models/MobileNetV2_TSM.onnx" \
    "MobileNetV2 TSM (дактиль)"

# Предобученная модель для жестов (140 MB)
echo ""
echo "3/3: Предобученная модель жестов (140 MB)"
download_file \
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/mvit/onnx/mvit32-2.onnx" \
    "data/models/mvit32-2.onnx" \
    "MViT v2 (1000 жестов)"

# Проверка установки
echo ""
echo "🔍 Проверка установки..."
python3 -c "
import torch
import mediapipe as mp
import cv2
import onnxruntime

print(f'  PyTorch: {torch.__version__}')
print(f'  MediaPipe: {mp.__version__}')
print(f'  OpenCV: {cv2.__version__}')
print(f'  ONNX Runtime: {onnxruntime.__version__}')

# Проверка Apple Silicon
if torch.backends.mps.is_available():
    print('  Apple Silicon (MPS): ✓')
else:
    print('  Apple Silicon (MPS): ✗ (будет использоваться CPU)')
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Все библиотеки работают${NC}"
else
    echo -e "${RED}✗ Ошибка проверки библиотек${NC}"
fi

# Инструкции
echo ""
echo "=========================================="
echo -e "${GREEN}🎉 Установка завершена!${NC}"
echo "=========================================="
echo ""
echo "Следующие шаги:"
echo ""
echo "1. Активируйте окружение:"
echo "   source venv/bin/activate"
echo ""
echo "2. Тест MediaPipe (веб-камера):"
echo "   python src/landmark_extractor.py"
echo ""
echo "3. Запуск веб-интерфейса:"
echo "   streamlit run app/streamlit_app.py"
echo ""
echo "4. Обучение модели (опционально):"
echo "   python src/train.py --dataset bukva --epochs 50"
echo ""
echo "Документация: SIGNBRIDGE_GUIDE.md"
echo ""
