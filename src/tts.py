"""
SignBridge - Text-to-Speech модуль для русского языка
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional
import tempfile
import os


class RussianTTS:
    """Text-to-Speech на русском языке с использованием Silero"""
    
    SPEAKERS = {
        'aidar': 'Мужской голос (Aidar)',
        'baya': 'Женский голос (Baya)',
        'kseniya': 'Женский голос (Kseniya)',
        'xenia': 'Женский голос (Xenia)',
        'eugene': 'Мужской голос (Eugene)',
        'random': 'Случайный голос'
    }
    
    def __init__(self, speaker: str = 'baya', sample_rate: int = 48000):
        """
        Args:
            speaker: Имя голоса (aidar, baya, kseniya, xenia, eugene, random)
            sample_rate: Частота дискретизации (8000, 24000, 48000)
        """
        self.speaker = speaker
        self.sample_rate = sample_rate
        self.device = torch.device('cpu')  # Silero работает на CPU
        
        print(f"Loading Silero TTS model...")
        
        # Загрузка модели через torch.hub
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='ru',
            speaker='v4_ru'
        )
        self.model.to(self.device)
        
        print(f"TTS model loaded. Speaker: {speaker}")
    
    def synthesize(self, text: str, speaker: Optional[str] = None) -> np.ndarray:
        """
        Синтез речи из текста
        
        Args:
            text: Текст для озвучки
            speaker: Голос (опционально, иначе используется default)
            
        Returns:
            Numpy array с аудио
        """
        speaker = speaker or self.speaker
        
        # Генерация аудио
        audio = self.model.apply_tts(
            text=text,
            speaker=speaker,
            sample_rate=self.sample_rate
        )
        
        return audio.numpy()
    
    def synthesize_to_file(
        self, 
        text: str, 
        output_path: str,
        speaker: Optional[str] = None
    ) -> str:
        """
        Синтез речи в файл
        
        Args:
            text: Текст для озвучки
            output_path: Путь к выходному файлу
            speaker: Голос
            
        Returns:
            Путь к файлу
        """
        audio = self.synthesize(text, speaker)
        
        # Сохранение
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
        torchaudio.save(output_path, audio_tensor, self.sample_rate)
        
        return output_path
    
    def play(self, text: str, speaker: Optional[str] = None):
        """
        Озвучка текста с воспроизведением
        
        Args:
            text: Текст для озвучки
            speaker: Голос
        """
        # Создаём временный файл
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        try:
            self.synthesize_to_file(text, temp_path, speaker)
            
            # Воспроизведение через afplay (macOS)
            os.system(f'afplay "{temp_path}"')
        finally:
            # Удаление временного файла
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @classmethod
    def list_speakers(cls) -> dict:
        """Список доступных голосов"""
        return cls.SPEAKERS


class SignBridgeTTS:
    """
    TTS для SignBridge с поддержкой очереди и прерывания
    """
    
    def __init__(self, speaker: str = 'baya'):
        self.tts = RussianTTS(speaker=speaker)
        self.queue = []
        self.is_speaking = False
    
    def speak(self, text: str, interrupt: bool = True):
        """
        Озвучка текста
        
        Args:
            text: Текст
            interrupt: Прервать текущее воспроизведение
        """
        if not text.strip():
            return
        
        if interrupt:
            self.queue.clear()
        
        self.queue.append(text)
        
        if not self.is_speaking:
            self._process_queue()
    
    def _process_queue(self):
        """Обработка очереди"""
        self.is_speaking = True
        
        while self.queue:
            text = self.queue.pop(0)
            self.tts.play(text)
        
        self.is_speaking = False
    
    def speak_gesture(self, gesture: str):
        """Озвучка распознанного жеста"""
        self.speak(gesture)
    
    def speak_sentence(self, words: list):
        """Озвучка предложения из слов"""
        sentence = " ".join(words)
        self.speak(sentence)


# === ТЕСТИРОВАНИЕ ===

if __name__ == "__main__":
    print("SignBridge TTS Test")
    print("="*40)
    
    print("\nДоступные голоса:")
    for speaker, desc in RussianTTS.list_speakers().items():
        print(f"  {speaker}: {desc}")
    
    print("\nИнициализация TTS...")
    tts = RussianTTS(speaker='baya')
    
    # Тестовые фразы
    test_phrases = [
        "Привет! Я SignBridge.",
        "Система распознавания русского жестового языка.",
        "Покажите жест, и я его озвучу."
    ]
    
    print("\nТестирование синтеза речи:")
    
    for phrase in test_phrases:
        print(f"\n  Текст: {phrase}")
        
        # Генерация аудио
        audio = tts.synthesize(phrase)
        print(f"  Длительность: {len(audio) / tts.sample_rate:.2f} секунд")
        
        # Сохранение в файл
        output_path = f"test_output_{test_phrases.index(phrase)}.wav"
        tts.synthesize_to_file(phrase, output_path)
        print(f"  Сохранено: {output_path}")
    
    # Воспроизведение
    print("\n\nВоспроизведение (нажмите Enter для каждой фразы):")
    for phrase in test_phrases:
        input(f"  Нажмите Enter для: '{phrase}'")
        tts.play(phrase)
    
    print("\nТест завершён!")
