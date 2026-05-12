import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def discover_classes(root: Path) -> List[str]:
    """Return sorted list of class folder names under root."""
    classes = sorted(
        d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not classes:
        raise ValueError(f"No class subdirectories found in {root}")
    return classes


def collect_samples(root: Path, classes: List[str]) -> Tuple[List[Path], List[int]]:
    """Walk class folders and return (image_paths, labels)."""
    paths: List[Path] = []
    labels: List[int] = []
    for idx, cls_name in enumerate(classes):
        cls_dir = root / cls_name
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file():
                paths.append(f)
                labels.append(idx)
    return paths, labels


def build_event_train_transforms(size: int = 96, hflip: bool = False):
    hflip_p = 0.5 if hflip else 0.0
    return A.Compose(
        [
            A.Resize(height=size, width=size),
            A.HorizontalFlip(p=hflip_p),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussNoise(p=0.15),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_event_valid_transforms(size: int = 96):
    return A.Compose(
        [
            A.Resize(height=size, width=size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class EventDataset(Dataset):
    """Image-folder classification dataset for arrow events."""

    def __init__(
        self,
        root: Path,
        classes: List[str],
        transform=None,
    ) -> None:
        self.root = Path(root)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.paths, self.labels = collect_samples(self.root, classes)
        self.transform = transform

        if len(self.paths) == 0:
            raise ValueError(f"No images found in {root}. Add images to class subfolders.")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path = self.paths[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0

        return image, self.labels[index]
