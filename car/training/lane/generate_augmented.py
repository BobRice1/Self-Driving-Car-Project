"""
Offline synthetic data generator for lane-following training.

Pre-generates augmented copies of training images so the model sees a
larger effective dataset each epoch (on top of the regular online
augmentations applied during training).

Usage:
    python -m car.training.lane.generate_augmented \
        --data_dir data \
        --num_synthetic 10000 \
        --seed 42

Produces:
    <data_dir>/train_augmented.csv   – original rows + synthetic rows
    <data_dir>/train_images_aug/     – augmented image files
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")


def resolve_image_file(image_id, image_dir: Path) -> Path:
    stem = str(int(image_id))
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found for image_id={image_id} in {image_dir}")


def build_offline_augment():
    """Heavier augmentations than online training transforms — these are
    baked into the saved images, and online transforms are applied on top."""
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.2),
        A.GaussNoise(p=0.15),
        A.Affine(
            scale=(0.93, 1.07),
            translate_percent=(-0.05, 0.05),
            rotate=(-5, 5),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.4,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3,
        ),
        A.ImageCompression(quality_range=(70, 95), p=0.15),
    ])


def parse_args():
    p = argparse.ArgumentParser(description="Generate offline augmented images.")
    p.add_argument("--data_dir", type=Path, default=Path("data"))
    p.add_argument("--csv", type=Path, default=None,
                   help="Path to train.csv (default: <data_dir>/train.csv)")
    p.add_argument("--num_synthetic", type=int, default=10000,
                   help="Number of augmented images to generate")
    p.add_argument("--hflip_p", type=float, default=0.3,
                   help="Probability of horizontal flip (mirrors angle)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quality", type=int, default=95,
                   help="JPEG save quality for augmented images")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    csv_path = args.csv or args.data_dir / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    image_dir = None
    for candidate in [args.data_dir / "train_images", args.data_dir / "training_data"]:
        if candidate.exists():
            image_dir = candidate
            break
    if image_dir is None:
        raise FileNotFoundError(f"No image directory found in {args.data_dir}")

    df = pd.read_csv(csv_path)
    print(f"Original dataset: {len(df)} rows")
    print(f"Generating {args.num_synthetic} augmented images...")

    aug_dir = args.data_dir / "train_images_aug"
    aug_dir.mkdir(parents=True, exist_ok=True)

    transform = build_offline_augment()

    source_indices = [random.randint(0, len(df) - 1) for _ in range(args.num_synthetic)]

    aug_rows = []
    max_id = int(df["image_id"].max())
    next_id = max_id + 1

    for i, src_idx in enumerate(tqdm(source_indices, desc="Augmenting")):
        row = df.iloc[src_idx]
        image_id = row["image_id"]
        angle = float(row["angle"])

        try:
            img_path = resolve_image_file(image_id, image_dir)
        except FileNotFoundError:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = transform(image=img_rgb)["image"]

        if random.random() < args.hflip_p:
            augmented = np.ascontiguousarray(np.fliplr(augmented))
            angle = 1.0 - angle

        augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        aug_id = next_id
        next_id += 1
        out_path = aug_dir / f"{aug_id}.jpg"
        cv2.imwrite(str(out_path), augmented_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, args.quality])

        new_row = {"image_id": aug_id, "angle": angle}
        if "speed" in df.columns:
            new_row["speed"] = row["speed"]
        aug_rows.append(new_row)

    aug_df = pd.DataFrame(aug_rows)
    combined = pd.concat([df, aug_df], ignore_index=True)

    out_csv = args.data_dir / "train_augmented.csv"
    combined.to_csv(out_csv, index=False)

    print(f"\nDone! Generated {len(aug_rows)} augmented images in {aug_dir}")
    print(f"Combined CSV: {out_csv} ({len(combined)} total rows)")
    print(f"  Original: {len(df)} | Synthetic: {len(aug_rows)}")


if __name__ == "__main__":
    main()
