import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Resize, CenterCrop
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    ShortSideScale,
)
from pytorchvideo.data import RandomClipSampler, UniformClipSampler, make_clip_sampler
from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from pathlib import Path
import pandas as pd

import config
from dataset import CustomLabeledVideoDataset
import calculate_time

dataset_path = config.dataset_path
classes_file = config.classes_file

# Загрузка меток классов
class_names = pd.read_csv(classes_file, header=None)
num_classes = len(class_names)

# Создание датасета
labeled_video_paths = LabeledVideoPaths.from_path(dataset_path)
clip_sampler = RandomClipSampler(clip_duration=2.0)

# Преобразования
video_transform = Compose([
    UniformTemporalSubsample(8),
    ShortSideScale(size=256),
    CenterCrop(224),
])


dataset = CustomLabeledVideoDataset(
    labeled_video_paths,
    clip_sampler,
    decode_audio=False,
    transform=ApplyTransformToKey(
        key="video",
        transform=video_transform,
    ),
)

# Разделение датасета на обучающую и валидационную выборки
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader для обучающего и валидационного наборов
train_loader = DataLoader(train_dataset.dataset, batch_size=4, shuffle=False)
val_loader = DataLoader(val_dataset.dataset, batch_size=4, shuffle=False)

model = r3d_18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Определение оптимизатора и функции потерь
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Включение CUDA ядер
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(device.type)

# Обучение и валидация
num_epochs = 500
time_calculator = calculate_time.TimeCalculator()
time_calculator.start()

best_val_loss = float('inf')

for epoch in range(num_epochs):
    # Обучение
    model.train()
    for data in train_loader:
        inputs, labels = data['video'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Валидация
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data['video'].to(device), data['label'].to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()

    # Средняя потеря и точность на валидационном наборе
    avg_val_loss = val_loss / len(val_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item()}, Validation Loss: {avg_val_loss}')

    if avg_val_loss < best_val_loss:
        print(f'Saving new best model at epoch {epoch + 1} with validation loss: {avg_val_loss:.4f}')
        best_val_loss = avg_val_loss
        best_model_path = f'out_models/best_model_epoch_{epoch}.pth'
        torch.save(model.state_dict(), best_model_path)

time_calculator.end()
print(time_calculator.get_passed_time())

torch.save(model.state_dict(), 'model_state_dict.pth')
