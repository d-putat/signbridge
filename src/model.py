"""
SignBridge - Архитектуры нейросетей для распознавания жестов
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SignLanguageLSTM(nn.Module):
    """
    Базовая LSTM модель для распознавания жестов
    
    Принимает последовательность векторов ключевых точек и 
    классифицирует жест.
    """
    
    def __init__(
        self,
        input_size: int = 159,      # 126 (руки) + 33 (верхняя часть тела)
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 33,       # Для дактиля
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Входной слой нормализации
        self.input_norm = nn.LayerNorm(input_size)
        
        # LSTM слои
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention механизм
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_size)
            lengths: (batch_size,) - реальные длины последовательностей
            
        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # Нормализация входа
        x = self.input_norm(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Взвешенная сумма
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden*2)
        
        # Классификация
        logits = self.classifier(context)
        
        return logits


class SignLanguageTransformer(nn.Module):
    """
    Transformer модель для распознавания жестов
    
    Лучше работает на длинных последовательностях, 
    но требует больше данных для обучения.
    """
    
    def __init__(
        self,
        input_size: int = 159,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        num_classes: int = 33,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Проекция входа
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # CLS token для классификации
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_size)
            
        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        
        # Проекция
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Добавление CLS токена
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, d_model)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # Берём CLS токен для классификации
        cls_output = x[:, 0, :]  # (batch, d_model)
        
        # Классификация
        logits = self.classifier(cls_output)
        
        return logits


class PositionalEncoding(nn.Module):
    """Синусоидальное позиционное кодирование"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SignLanguageCNNLSTM(nn.Module):
    """
    Гибридная CNN + LSTM модель
    
    CNN извлекает локальные паттерны, LSTM моделирует временные зависимости.
    Оптимальный баланс между точностью и скоростью.
    """
    
    def __init__(
        self,
        input_size: int = 159,
        hidden_size: int = 256,
        num_classes: int = 33,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # 1D CNN для извлечения локальных паттернов
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_size)
            
        Returns:
            logits: (batch_size, num_classes)
        """
        # CNN ожидает (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)  # Обратно (batch, seq_len, channels)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Global Average Pooling
        x = lstm_out.mean(dim=1)
        
        # Классификация
        logits = self.classifier(x)
        
        return logits


def get_model(
    model_type: str = "lstm",
    input_size: int = 159,
    num_classes: int = 33,
    **kwargs
) -> nn.Module:
    """
    Фабрика моделей
    
    Args:
        model_type: "lstm", "transformer", "cnn_lstm"
        input_size: Размер входного вектора
        num_classes: Количество классов
        
    Returns:
        Инициализированная модель
    """
    models = {
        "lstm": SignLanguageLSTM,
        "transformer": SignLanguageTransformer,
        "cnn_lstm": SignLanguageCNNLSTM
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    return models[model_type](input_size=input_size, num_classes=num_classes, **kwargs)


# === ТЕСТИРОВАНИЕ ===

if __name__ == "__main__":
    # Тест моделей
    batch_size = 4
    seq_len = 30
    input_size = 159
    num_classes = 33
    
    # Тестовые данные
    x = torch.randn(batch_size, seq_len, input_size)
    
    print("Testing models...")
    
    for model_type in ["lstm", "transformer", "cnn_lstm"]:
        model = get_model(model_type, input_size, num_classes)
        output = model(x)
        print(f"{model_type}: input {x.shape} -> output {output.shape}")
        
        # Подсчёт параметров
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {params:,}")
    
    print("\nAll models work correctly!")
