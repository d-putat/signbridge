"""
SignBridge - Streamlit веб-приложение для демонстрации
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Добавляем путь к src
sys.path.append(str(Path(__file__).parent.parent / "src"))

from landmark_extractor import LandmarkExtractor

# Конфигурация страницы
st.set_page_config(
    page_title="SignBridge - Распознавание РЖЯ",
    page_icon="🤟",
    layout="wide"
)

# Стили
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 2rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown('<p class="main-header">🤟 SignBridge</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Система распознавания русского жестового языка</p>', unsafe_allow_html=True)

# Инициализация session state
if 'extractor' not in st.session_state:
    st.session_state.extractor = LandmarkExtractor()

if 'recognized_text' not in st.session_state:
    st.session_state.recognized_text = []

# Боковая панель
with st.sidebar:
    st.header("⚙️ Настройки")
    
    mode = st.radio(
        "Режим работы",
        ["Камера", "Загрузка видео", "Информация"]
    )
    
    st.divider()
    
    st.header("📊 Статистика")
    st.metric("Распознано жестов", len(st.session_state.recognized_text))
    
    if st.button("🗑️ Очистить историю"):
        st.session_state.recognized_text = []
        st.rerun()
    
    st.divider()
    
    st.header("📖 Дактильная азбука")
    dactyl = "А Б В Г Д Е Ё Ж З И Й К Л М Н О П Р С Т У Ф Х Ц Ч Ш Щ Ъ Ы Ь Э Ю Я"
    st.text(dactyl)

# Основной контент
if mode == "Камера":
    st.header("📹 Распознавание с камеры")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Виджет камеры
        camera_input = st.camera_input("Покажите жест")
        
        if camera_input:
            # Обработка изображения
            bytes_data = camera_input.getvalue()
            nparr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Извлечение landmarks
            extractor = st.session_state.extractor
            data = extractor.extract_from_frame(frame)
            
            # Отрисовка
            annotated = extractor.draw_landmarks(frame, data)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            st.image(annotated_rgb, caption="Обработанное изображение", use_container_width=True)
            
            # Информация о landmarks
            hands_detected = (data.left_hand is not None) + (data.right_hand is not None)
            
            st.markdown(f"""
            <div class="info-box">
                <b>Обнаружено:</b><br>
                👋 Руки: {hands_detected}/2<br>
                🧍 Поза: {"✅" if data.pose is not None else "❌"}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📝 Распознанный текст")
        
        # Заглушка для демо
        st.markdown("""
        <div class="result-box">
            Покажите жест для распознавания
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        **Для полноценного распознавания:**
        1. Скачайте датасет Bukva
        2. Обучите модель: `python src/train.py`
        3. Или скачайте готовую модель
        """)

elif mode == "Загрузка видео":
    st.header("📁 Загрузка видео")
    
    uploaded_file = st.file_uploader(
        "Выберите видео с жестами",
        type=['mp4', 'avi', 'mov', 'webm']
    )
    
    if uploaded_file:
        # Сохранение во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        st.video(tmp_path)
        
        if st.button("🔍 Анализировать видео"):
            with st.spinner("Извлечение ключевых точек..."):
                cap = cv2.VideoCapture(tmp_path)
                frames_processed = 0
                
                progress_bar = st.progress(0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Обработка каждого 3-го кадра
                    if frames_processed % 3 == 0:
                        data = st.session_state.extractor.extract_from_frame(frame)
                    
                    frames_processed += 1
                    progress_bar.progress(frames_processed / total_frames)
                
                cap.release()
                
                st.success(f"Обработано {frames_processed} кадров!")
        
        # Очистка
        os.unlink(tmp_path)

elif mode == "Информация":
    st.header("ℹ️ О проекте SignBridge")
    
    st.markdown("""
    ## 🎯 Цель проекта
    
    SignBridge — это система для двусторонней коммуникации между слышащими 
    и людьми с нарушениями слуха, использующая русский жестовый язык (РЖЯ).
    
    ## 🛠️ Возможности
    
    - **Распознавание жестов** → текст/речь
    - **Дактильная азбука** (33 буквы)
    - **1000 жестов** из датасета Slovo
    - **Работа в реальном времени**
    - **Озвучка** с помощью Silero TTS
    
    ## 📊 Датасеты
    
    | Датасет | Классы | Размер |
    |---------|--------|--------|
    | [Slovo](https://github.com/hukenovs/slovo) | 1000 жестов | 16 GB |
    | [Bukva](https://github.com/ai-forever/bukva) | 33 буквы | 27 GB |
    
    ## 🚀 Быстрый старт
    
    ```bash
    # 1. Установка зависимостей
    pip install -r requirements.txt
    
    # 2. Скачивание данных
    curl -o data/slovo/slovo_mediapipe.json \\
      https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo_mediapipe.json
    
    # 3. Тест MediaPipe
    python src/landmark_extractor.py
    
    # 4. Обучение модели
    python src/train.py --dataset bukva --epochs 50
    
    # 5. Запуск демо
    streamlit run app/streamlit_app.py
    ```
    
    ## 👥 Авторы
    
    Проект создан для конкурса "Большие вызовы"
    """)
    
    st.divider()
    
    # Архитектура
    st.subheader("🏗️ Архитектура системы")
    
    st.code("""
    ┌─────────────────────────────────────────────────────┐
    │                    SignBridge                        │
    ├─────────────────────────────────────────────────────┤
    │                                                      │
    │   📹 Камера → MediaPipe → Ключевые точки (543)      │
    │                              │                       │
    │                              ▼                       │
    │   🧠 LSTM/Transformer → Классификация жеста         │
    │                              │                       │
    │                              ▼                       │
    │   🔊 Silero TTS → Озвучка на русском                │
    │                                                      │
    └─────────────────────────────────────────────────────┘
    """, language="text")

# Футер
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9rem;">
    SignBridge © 2026 | Конкурс "Большие вызовы"<br>
    Направление: Большие данные, ИИ, автоматизированные системы
</div>
""", unsafe_allow_html=True)
