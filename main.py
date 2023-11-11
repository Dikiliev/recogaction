import torch
from torchvision.transforms import Compose, Resize, CenterCrop
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    ShortSideScale,
)
from pytorchvideo.data import UniformClipSampler, LabeledVideoDataset
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from torchvision.models.video import r3d_18
import torch.nn as nn

from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths

import calculate_time

# Путь к тестовым данным
test_dataset_path = Path('test/videos')
test_classes_file = Path('test/classes.csv')

# Загрузка меток классов
class_names = pd.read_csv(test_classes_file, header=None)
num_classes = len(class_names)

# Создание датасета для тестирования
test_labeled_video_paths = LabeledVideoPaths.from_path(test_dataset_path)
clip_sampler = UniformClipSampler(clip_duration=2.0)

# Преобразования
video_transform = Compose([
    UniformTemporalSubsample(8),
    ShortSideScale(size=256),
    CenterCrop(224),
])

test_dataset = LabeledVideoDataset(
    test_labeled_video_paths,
    clip_sampler,
    decode_audio=False,
    transform=ApplyTransformToKey(
        key="video",
        transform=video_transform,
    ),
)

# DataLoader для тестового датасета
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model = r3d_18(pretrained=False)  # Создаем модель с той же архитектурой
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Изменяем последний слой
model.load_state_dict(torch.load('model_state_dict.pth'))  # Загружаем веса
model.eval()  # Переключаем модель в режим оценки


# Включение CUDA ядер
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Тестирование
correct = 0
total = 0

time_calculator = calculate_time.TimeCalculator()
time_calculator.start()

with torch.no_grad():
    for data in test_dataloader:
        inputs, labels = data['video'].to(device), data['label'].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print(f'Accuracy of the network on the test videos: {100 * correct / total}%')

time_calculator.end()
print(time_calculator.get_passed_time())