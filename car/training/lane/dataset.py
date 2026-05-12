import random
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import CROP_RATIO

IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")
ANGLE_MIN = 50.0
ANGLE_MAX = 120.0


def resolve_image_file(image_id, image_dir: Path, extra_dirs: tuple = ()) -> Path:
    stem = str(int(image_id))
    for search_dir in (image_dir, *extra_dirs):
        for ext in IMAGE_EXTENSIONS:
            candidate = search_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"Image not found for image_id={image_id} in {image_dir}. "
        f"Tried extensions: {IMAGE_EXTENSIONS}"
    )


def normalise_car_angle(angle: float) -> float:
    """Convert Pi car steering units [50, 120] to model target [0, 1]."""
    return (max(ANGLE_MIN, min(ANGLE_MAX, float(angle))) - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN)


def denormalise_car_angle(angle_norm: float) -> float:
    """Convert model output [0, 1] back to Pi car steering units [50, 120]."""
    return ANGLE_MIN + max(0.0, min(1.0, float(angle_norm))) * (ANGLE_MAX - ANGLE_MIN)


class LaneDataset(Dataset):
    """Dataset that loads images, crops the lower portion for lane visibility,
    and provides the steering angle as the regression target."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path,
        transform=None,
        is_train: bool = False,
        hflip_p: float = 0.0,
        crop_ratio: float = CROP_RATIO,
        extra_image_dirs: tuple = (),
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.extra_image_dirs = tuple(Path(d) for d in extra_image_dirs)
        self.transform = transform
        self.is_train = is_train
        self.hflip_p = float(hflip_p)
        self.crop_ratio = crop_ratio
        if "image_path" in self.df.columns:
            self.image_paths = [Path(p) for p in self.df["image_path"].tolist()]
            self.image_ids = [p.stem for p in self.image_paths]
        else:
            self.image_ids = self.df["image_id"].tolist()
            self.image_paths = [
                resolve_image_file(iid, self.image_dir, self.extra_image_dirs)
                for iid in self.image_ids
            ]

    def __len__(self) -> int:
        return len(self.df)

    def _load_and_crop(self, index: int) -> np.ndarray:
        path = self.image_paths[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h = image.shape[0]
        crop_start = int(h * self.crop_ratio)
        image = image[crop_start:, :, :]
        return image

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image = self._load_and_crop(index)
        angle = float(self.df.at[index, "angle"])

        if self.is_train and self.hflip_p > 0.0 and random.random() < self.hflip_p:
            image = np.ascontiguousarray(np.fliplr(image))
            angle = 1.0 - angle

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0

        targets = {"angle": torch.tensor(angle, dtype=torch.float32)}
        return image, targets
