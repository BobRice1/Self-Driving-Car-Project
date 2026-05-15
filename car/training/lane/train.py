"""
Lane-following model training script.

Usage:
    python -m car.training.lane.train --config car/configs/lane.yaml --data_dir data
"""
from __future__ import annotations

import argparse
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import pandas as pd
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import IMAGE_EXTENSIONS, LaneDataset, normalise_car_angle, resolve_image_file
from .model import LaneFollower
from .transforms import build_train_transforms, build_valid_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lane-following model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--out_dir", type=Path, default=Path("car/models/lane/checkpoints"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--filename_labels", action="store_true",
                        help="Read labels from filenames like timestamp_angle_speed.png")
    parser.add_argument("--extra_csv_data_dir", type=Path, default=None,
                        help="Optional CSV-backed data dir to merge with filename-labelled data")
    parser.add_argument("--no_augmented", action="store_true",
                        help="Ignore train_augmented.csv even if it exists")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(mode: str) -> torch.device:
    if mode == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(mode)


def find_csv(data_dir: Path, prefer_augmented: bool = True) -> Path:
    if prefer_augmented:
        aug = data_dir / "train_augmented.csv"
        if aug.exists():
            print(f"Using augmented CSV: {aug}")
            return aug
    for candidate in [data_dir / "train.csv", Path("train.csv"), Path("data/train.csv")]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Cannot find train.csv")


def find_image_dir(data_dir: Path) -> Path:
    if data_dir.exists() and any(
        p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in data_dir.iterdir()
    ):
        return data_dir
    for candidate in [
        data_dir / "train_images",
        data_dir / "training_data",
        data_dir / "data" / "train_images",
        data_dir / "data" / "training_data",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Cannot find training image directory")


def parse_filename_label(path: Path) -> dict | None:
    match = re.match(r"^(.+)_([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)$", path.stem)
    if not match:
        return None
    angle_car = float(match.group(2))
    speed = float(match.group(3))
    return {
        "image_id": match.group(1),
        "image_path": str(path),
        "angle": normalise_car_angle(angle_car),
        "angle_car": angle_car,
        "speed": speed,
    }


def load_filename_labelled_data(image_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        row = parse_filename_label(path)
        if row is not None:
            rows.append(row)
    if not rows:
        raise ValueError(
            f"No filename-labelled images found in {image_dir}. "
            "Expected names like 1772205714119_90_0.png"
        )
    return pd.DataFrame(rows)


def load_csv_labelled_data(data_dir: Path, prefer_augmented: bool = True) -> tuple[pd.DataFrame, Path, Path]:
    csv_path = find_csv(data_dir, prefer_augmented=prefer_augmented)
    image_dir = find_image_dir(data_dir)
    df = pd.read_csv(csv_path).sort_values("image_id").reset_index(drop=True)
    rows = []
    for _, row in df.iterrows():
        try:
            image_path = resolve_image_file(row["image_id"], image_dir)
        except FileNotFoundError:
            continue
        rows.append({
            "image_id": row["image_id"],
            "image_path": str(image_path),
            "angle": float(row["angle"]),
            "source": "csv",
        })
    return pd.DataFrame(rows), csv_path, image_dir


def filter_invalid_rows(df: pd.DataFrame, image_dir: Path, extra_dirs: tuple = ()) -> pd.DataFrame:
    valid = []
    for _, row in df.iterrows():
        try:
            if "image_path" in df.columns:
                fp = Path(row["image_path"])
            else:
                fp = resolve_image_file(row["image_id"], image_dir, extra_dirs)
            if fp.stat().st_size > 0 and cv2.imread(str(fp), cv2.IMREAD_COLOR) is not None:
                valid.append(True)
            else:
                valid.append(False)
        except FileNotFoundError:
            valid.append(False)
    dropped = sum(1 for v in valid if not v)
    if dropped:
        print(f"Dropped {dropped} invalid images.")
    return df.loc[valid].reset_index(drop=True)


def blocked_split(df: pd.DataFrame, val_ratio: float = 0.15):
    """Blocked temporal split: last val_ratio% of sorted data is validation."""
    sort_col = "image_path" if "image_path" in df.columns else "image_id"
    df = df.sort_values(sort_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - val_ratio))
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True)


def create_warmup_cosine_scheduler(optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, use_amp):
    model.train()
    running_loss = 0.0
    seen = 0

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        angle_true = targets["angle"].to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            angle_pred = model(images)
            loss = torch.nn.functional.mse_loss(angle_pred, angle_true)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.detach().item() * batch_size
        seen += batch_size

    return running_loss / max(seen, 1)


@torch.no_grad()
def validate(model, loader, device, use_amp):
    model.eval()
    running_loss = 0.0
    seen = 0

    for images, targets in tqdm(loader, desc="valid", leave=False):
        images = images.to(device, non_blocking=True)
        angle_true = targets["angle"].to(device, non_blocking=True)
        batch_size = images.size(0)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            angle_pred = model(images)
            loss = torch.nn.functional.mse_loss(angle_pred, angle_true)

        running_loss += loss.detach().item() * batch_size
        seen += batch_size

    return running_loss / max(seen, 1)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(args.seed)

    device = resolve_device(args.device)
    data_cfg = config["data"]
    train_cfg = config["training"]

    image_dir = find_image_dir(args.data_dir)

    extra_image_dirs = ()
    aug_dir = args.data_dir / "train_images_aug"
    if aug_dir.exists():
        extra_image_dirs = (aug_dir,)
        print(f"Found augmented image dir: {aug_dir}")

    if args.filename_labels:
        csv_path = None
        df = load_filename_labelled_data(image_dir)
        df["source"] = "filename"
        print(f"Filename labels: {image_dir}")
        print(
            f"Angle car units: min={df['angle_car'].min():.1f}, "
            f"max={df['angle_car'].max():.1f}; speeds={sorted(df['speed'].unique().tolist())}"
        )
        if args.extra_csv_data_dir is not None:
            extra_df, extra_csv_path, extra_image_dir = load_csv_labelled_data(
                args.extra_csv_data_dir,
                prefer_augmented=not args.no_augmented,
            )
            print(f"Extra CSV labels: {extra_csv_path}, Images: {extra_image_dir}, Rows: {len(extra_df)}")
            df = pd.concat([df, extra_df], ignore_index=True)
    else:
        csv_path = find_csv(args.data_dir, prefer_augmented=not args.no_augmented)
        print(f"CSV: {csv_path}, Images: {image_dir}")
        df = pd.read_csv(csv_path).sort_values("image_id").reset_index(drop=True)
    df = filter_invalid_rows(df, image_dir, extra_dirs=extra_image_dirs)
    train_df, valid_df = blocked_split(df, val_ratio=0.15)
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}")

    height = int(data_cfg["height"])
    width = int(data_cfg["width"])
    crop_ratio = float(data_cfg.get("crop_ratio", 0.4))
    num_workers = int(args.num_workers if args.num_workers is not None else data_cfg.get("num_workers", 4))

    train_ds = LaneDataset(
        df=train_df, image_dir=image_dir,
        transform=build_train_transforms(height, width),
        is_train=True, hflip_p=float(data_cfg.get("hflip_p", 0.3)),
        crop_ratio=crop_ratio,
        extra_image_dirs=extra_image_dirs,
    )
    valid_ds = LaneDataset(
        df=valid_df, image_dir=image_dir,
        transform=build_valid_transforms(height, width),
        is_train=False, crop_ratio=crop_ratio,
        extra_image_dirs=extra_image_dirs,
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=int(data_cfg.get("batch_size", 64)),
        shuffle=True, num_workers=num_workers, pin_memory=pin, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=int(data_cfg.get("val_batch_size", 128)),
        shuffle=False, num_workers=num_workers, pin_memory=pin,
    )

    model = LaneFollower(
        pretrained=bool(config["model"].get("pretrained", True)),
        dropout=float(train_cfg.get("dropout", 0.2)),
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(train_cfg.get("epochs", 25))
    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(train_cfg.get("warmup_epochs", 2)) * steps_per_epoch
    scheduler = create_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps)

    use_amp = bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda" and not args.disable_amp
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    run_name = config.get("experiment", {}).get("run_name", "lane")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    patience = int(train_cfg.get("early_stopping_patience", 6))
    bad_epochs = 0
    ckpt_path = out_dir / "best.pt"

    print(f"Device: {device}, AMP: {use_amp}")
    print(f"Output: {out_dir}")

    logs: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, use_amp)
        val_loss = validate(model, valid_loader, device, use_amp)

        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch={epoch:03d} train_mse={train_loss:.6f} val_mse={val_loss:.6f} lr={lr:.2e}")
        logs.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss, "lr": lr})

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_mse": val_loss, "config": config},
                ckpt_path,
            )
            print(f"  -> Saved best (val_mse={val_loss:.6f}): {ckpt_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (patience {patience}).")
                break

    pd.DataFrame(logs).to_csv(out_dir / "train_log.csv", index=False)
    print(f"Training complete. Best val_mse={best_val:.6f}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
