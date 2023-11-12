from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch.utils.data
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.video import VideoPathHandler

from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset

from torch.utils.data import Dataset, DataLoader



class CustomLabeledVideoDataset(LabeledVideoDataset):
    def __init__(
            self,
            labeled_video_paths: List[Tuple[str, Optional[dict]]],
            clip_sampler: ClipSampler,
            video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
            transform: Optional[Callable[[dict], Any]] = None,
            decode_audio: bool = True,
            decoder: str = "pyav",
    ) -> None:
        super().__init__(labeled_video_paths, clip_sampler, video_sampler, transform, decode_audio, decoder)

        self.paths_count = len(labeled_video_paths)

    def __len__(self):
        return self.paths_count


class CustomDataLoader(DataLoader):
    pass


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        return data_item
