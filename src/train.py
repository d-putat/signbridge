"""
SignBridge - Скрипт обучения моделей
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime
from pathlib import Path
import json

from model import get_model
from dataset import create_data_loaders


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """Одна эпоха обучения"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    
    for features, labels in pbar:
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> tuple:
    """Оценка модели"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Eval]")
    
    for features, labels in pbar:
        features = features.to(device)
        labels = labels.to(device)
        
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(loader)
    
    # Top-5 accuracy
    # (упрощённый расчёт, для полного нужно сохранять все logits)
    
    return avg_loss, accuracy, all_preds, all_labels


def train(
    model_type: str = "lstm",
    dataset_type: str = "bukva",
    data_dir: str = "data/bukva",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seq_length: int = 30,
    hidden_size: int = 256,
    save_dir: str = "checkpoints",
    device: str = "auto"
):
    """
    Основная функция обучения
    
    Args:
        model_type: Тип модели (lstm, transformer, cnn_lstm)
        dataset_type: Тип датасета (slovo, bukva)
        data_dir: Путь к данным
        epochs: Количество эпох
        batch_size: Размер батча
        learning_rate: Скорость обучения
        seq_length: Длина последовательности
        hidden_size: Размер скрытого слоя
        save_dir: Папка для сохранения чекпоинтов
        device: Устройство (auto, cpu, mps, cuda)
    """
    # Определение устройства
    if device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")  # Apple Silicon
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Создание DataLoader'ов
    print(f"\nLoading {dataset_type} dataset...")
    train_loader, test_loader, num_classes = create_data_loaders(
        dataset_type=dataset_type,
        data_dir=data_dir,
        batch_size=batch_size,
        seq_length=seq_length,
        num_workers=0,  # На macOS лучше 0 для совместимости
        augment=True
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Number of classes: {num_classes}")
    
    # Определение размера входа
    # 63 (левая рука) + 63 (правая рука) + 33 (верхняя часть тела) = 159
    input_size = 159
    
    # Создание модели
    print(f"\nCreating {model_type} model...")
    model = get_model(
        model_type=model_type,
        input_size=input_size,
        num_classes=num_classes,
        hidden_size=hidden_size
    )
    model = model.to(device)
    
    # Подсчёт параметров
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")
    
    # Loss и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Создание папки для чекпоинтов
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # История обучения
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_acc': [],
        'best_acc': 0.0,
        'config': {
            'model_type': model_type,
            'dataset_type': dataset_type,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'seq_length': seq_length,
            'hidden_size': hidden_size,
            'num_classes': num_classes,
            'input_size': input_size
        }
    }
    
    best_acc = 0.0
    
    # Обучение
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*50}")
        
        # Обучение
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Оценка
        test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device, epoch)
        
        # Обновление scheduler
        scheduler.step()
        
        # Логирование
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Сохранение лучшей модели
        if test_acc > best_acc:
            best_acc = test_acc
            history['best_acc'] = best_acc
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': history['config']
            }
            
            save_path = save_dir / f"best_{model_type}_{dataset_type}.pt"
            torch.save(checkpoint, save_path)
            print(f"  New best model saved: {save_path}")
        
        # Сохранение истории
        history_path = save_dir / f"history_{model_type}_{dataset_type}.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {save_dir}")
    print(f"{'='*50}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train SignBridge model")
    
    parser.add_argument("--model", type=str, default="lstm",
                       choices=["lstm", "transformer", "cnn_lstm"],
                       help="Model type")
    parser.add_argument("--dataset", type=str, default="bukva",
                       choices=["slovo", "bukva"],
                       help="Dataset type")
    parser.add_argument("--data-dir", type=str, default="data/bukva",
                       help="Path to data directory")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--seq-length", type=int, default=30,
                       help="Sequence length")
    parser.add_argument("--hidden-size", type=int, default=256,
                       help="Hidden layer size")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cpu, mps, cuda)")
    
    args = parser.parse_args()
    
    train(
        model_type=args.model,
        dataset_type=args.dataset,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seq_length=args.seq_length,
        hidden_size=args.hidden_size,
        save_dir=args.save_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
