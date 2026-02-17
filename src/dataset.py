import random
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .paths import resolve_image_file


class DrivingDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path,
        transform=None,
        is_train: bool = False,
        hflip_p: float = 0.0,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.is_train = is_train
        self.hflip_p = float(hflip_p)
        self.has_targets = {"angle", "speed"}.issubset(self.df.columns)
        self.image_ids = self.df["image_id"].tolist()
        self.image_paths = [resolve_image_file(image_id, self.image_dir) for image_id in self.image_ids]

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, index: int) -> np.ndarray:
        image_path = self.image_paths[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            size = image_path.stat().st_size if image_path.exists() else -1
            raise ValueError(f"Failed to read image: {image_path} (size_bytes={size})")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image = self._load_image(index)
        targets: Dict[str, torch.Tensor] = {}

        if self.has_targets:
            angle = float(self.df.at[index, "angle"])
            speed = float(self.df.at[index, "speed"])

            if self.is_train and self.hflip_p > 0.0 and random.random() < self.hflip_p:
                image = np.ascontiguousarray(np.fliplr(image))
                angle = 1.0 - angle

            targets["angle"] = torch.tensor(angle, dtype=torch.float32)
            targets["speed"] = torch.tensor(speed, dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0

        if not self.has_targets:
            image_id = self.image_ids[index]
            targets["image_id"] = torch.tensor(int(image_id), dtype=torch.int64)

        return image, targets
