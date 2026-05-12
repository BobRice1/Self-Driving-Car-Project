"""
Offline augmentation for arrow class folders.

This is intended for balancing left/right arrow classes before training.
It never uses horizontal flips, because flipping changes arrow direction.

Usage:
    python -m car.training.events.augment_arrows --root car/data/events/arrows --class_name left --target_count 487
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate augmented arrow images for one class.")
    parser.add_argument("--root", type=Path, default=Path("car/data/events/arrows"))
    parser.add_argument("--class_name", type=str, required=True, choices=["left", "right"])
    parser.add_argument("--target_count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=95)
    return parser.parse_args()


def build_transform() -> A.Compose:
    return A.Compose(
        [
            A.Affine(
                scale=(0.9, 1.12),
                translate_percent=(-0.08, 0.08),
                rotate=(-12, 12),
                shear=(-4, 4),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.9,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.28, contrast_limit=0.28, p=0.75),
            A.RandomGamma(gamma_limit=(75, 130), p=0.45),
            A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=20, val_shift_limit=18, p=0.35),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                ],
                p=0.2,
            ),
            A.GaussNoise(p=0.18),
            A.ImageCompression(quality_range=(70, 95), p=0.25),
        ]
    )


def list_original_images(class_dir: Path) -> list[Path]:
    return [
        p for p in sorted(class_dir.iterdir())
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not p.stem.startswith("aug_")
    ]


def list_all_images(class_dir: Path) -> list[Path]:
    return [
        p for p in sorted(class_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def next_aug_path(class_dir: Path, class_name: str, index: int) -> Path:
    while True:
        candidate = class_dir / f"aug_{class_name}_{index:05d}.png"
        if not candidate.exists():
            return candidate
        index += 1


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    class_dir = args.root / args.class_name
    if not class_dir.exists():
        raise FileNotFoundError(f"Class folder not found: {class_dir}")

    originals = list_original_images(class_dir)
    if not originals:
        raise ValueError(f"No original images found in {class_dir}")

    current_count = len(list_all_images(class_dir))
    needed = max(0, args.target_count - current_count)
    if needed == 0:
        print(f"{args.class_name}: already has {current_count} images, target={args.target_count}")
        return

    transform = build_transform()
    generated = 0
    next_index = 0

    while generated < needed:
        source_path = random.choice(originals)
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable image: {source_path}")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = transform(image=rgb)["image"]
        augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        out_path = next_aug_path(class_dir, args.class_name, next_index)
        next_index += 1
        cv2.imwrite(str(out_path), augmented_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        generated += 1

    final_count = len(list_all_images(class_dir))
    print(f"{args.class_name}: generated {generated}; final count={final_count}")


if __name__ == "__main__":
    main()
