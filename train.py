import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from torch.utils.data import DataLoader
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

import calculate_time

dataset_path = Path('videos')
classes_file = Path('classes.csv')

# Загрузка меток классов
class_names = pd.read_csv(classes_file, header=None)
num_classes = len(class_names)

# Создание датасета
labeled_video_paths = LabeledVideoPaths.from_path(dataset_path)
clip_sampler = RandomClipSampler(clip_duration=2.0)

# Преоброзования
video_transform = Compose([
    UniformTemporalSubsample(8),
    ShortSideScale(size=256),
    CenterCrop(224),
])

dataset = LabeledVideoDataset(
    labeled_video_paths,
    clip_sampler,
    decode_audio=False,
    transform=ApplyTransformToKey(
        key="video",
        transform=video_transform,
    ),
)

# DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

model = r3d_18(pretrained=True)  # Загрузка предобученной модели
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Замена последнего слоя

# Определение оптимизатора и функции потерь
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Включение CUDA ядер
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(device.type)

# Обучение
num_epochs = 10
time_calculator = calculate_time.TimeCalculator()
time_calculator.start()

for epoch in range(num_epochs):
    for data in dataloader:
        inputs, labels = data['video'].to(device), data['label'].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Обратный проход и оптимизация
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

time_calculator.end()
print(time_calculator.get_passed_time())

torch.save(model.state_dict(), 'model_state_dict.pth')

