from __future__ import annotations

import argparse
import math
import random
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from train_lane_model import build_model, IMAGENET_MEAN, IMAGENET_STD


ANGLE_MIN = 50.0
ANGLE_MAX = 135.0
ANGLE_STRAIGHT = 94.0
FIG8_NAME_RE = re.compile(r"^(\d+)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)\.(?:png|jpg|jpeg)$", re.IGNORECASE)


class FineTuneLaneDataset(Dataset):
    def __init__(self, csv_path: Path, height: int, width: int, crop_top_ratio: float, train: bool) -> None:
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.height = int(height)
        self.width = int(width)
        self.crop_top_ratio = float(crop_top_ratio)
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {row['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        steering = float(row["steering"])

        h = image.shape[0]
        image = image[int(h * self.crop_top_ratio):, :, :]
        if self.train:
            image, steering = self._augment(image, steering)

        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image), torch.tensor(np.float32(steering)), str(row.get("source", ""))

    def _augment(self, image: np.ndarray, steering: float) -> tuple[np.ndarray, float]:
        if random.random() < 0.45:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-12.0, 12.0)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        if random.random() < 0.25:
            shift = random.randint(-8, 8)
            matrix = np.float32([[1, 0, shift], [0, 1, 0]])
            image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
            steering = steering + shift * 0.18
        return image, float(np.clip(steering, ANGLE_MIN, ANGLE_MAX))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune lane model checkpoints with figure-8 data.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fig8-dir", type=Path, default=Path("NewModel/ensemble_model/8"))
    parser.add_argument("--base-splits", type=Path, default=Path("NewModel/splits"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--fig8-weight", type=float, default=12.0)
    parser.add_argument("--old-row-limit", type=int, default=4500)
    parser.add_argument("--val-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(mode: str) -> torch.device:
    if mode == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(mode)


def collect_fig8_rows(fig8_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(fig8_dir.rglob("*")):
        if not path.is_file():
            continue
        match = FIG8_NAME_RE.match(path.name)
        if match is None:
            continue
        timestamp, steering, speed = match.groups()
        rows.append({
            "image_path": str(path),
            "steering": float(np.clip(float(steering), ANGLE_MIN, ANGLE_MAX)),
            "speed": float(speed),
            "source": "figure8",
            "sequence_key": f"figure8_{timestamp}",
            "chunk_id": path.parent.name,
        })
    if not rows:
        raise SystemExit(f"No figure-8 images found under {fig8_dir}")
    return pd.DataFrame(rows)


def make_splits(args: argparse.Namespace, ckpt_meta: dict) -> tuple[Path, Path, Path]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = args.out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    old_train = pd.read_csv(args.base_splits / "train.csv")
    old_val = pd.read_csv(args.base_splits / "val.csv")
    if args.old_row_limit > 0 and len(old_train) > args.old_row_limit:
        old_train = old_train.sample(n=args.old_row_limit, random_state=args.seed).reset_index(drop=True)

    fig8 = collect_fig8_rows(args.fig8_dir)
    order = rng.permutation(len(fig8))
    val_count = max(1, int(round(len(fig8) * args.val_frac)))
    fig8_val = fig8.iloc[order[:val_count]].reset_index(drop=True)
    fig8_train = fig8.iloc[order[val_count:]].reset_index(drop=True)

    train = pd.concat([old_train, fig8_train], ignore_index=True)
    train["steering"] = train["steering"].astype(float).clip(ANGLE_MIN, ANGLE_MAX)
    old_val = old_val.copy()
    old_val["steering"] = old_val["steering"].astype(float).clip(ANGLE_MIN, ANGLE_MAX)

    train_csv = split_dir / "train.csv"
    old_val_csv = split_dir / "val_old.csv"
    fig8_val_csv = split_dir / "val_figure8.csv"
    train.to_csv(train_csv, index=False)
    old_val.to_csv(old_val_csv, index=False)
    fig8_val.to_csv(fig8_val_csv, index=False)

    summary = pd.DataFrame([
        {"split": "train_old", "rows": len(old_train)},
        {"split": "train_figure8", "rows": len(fig8_train)},
        {"split": "val_old", "rows": len(old_val)},
        {"split": "val_figure8", "rows": len(fig8_val)},
        {"split": "checkpoint_arch", "rows": ckpt_meta.get("arch", "nvidia")},
    ])
    summary.to_csv(split_dir / "summary.csv", index=False)
    return train_csv, old_val_csv, fig8_val_csv


def make_sampler(ds: FineTuneLaneDataset, fig8_weight: float) -> WeightedRandomSampler:
    angles = ds.df["steering"].astype(float).to_numpy()
    sources = ds.df["source"].astype(str).to_numpy() if "source" in ds.df.columns else np.array([""] * len(ds))
    weights = 1.0 + 2.0 * np.abs(angles - ANGLE_STRAIGHT) / 45.0
    weights *= np.where(sources == "figure8", float(fig8_weight), 1.0)
    return WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)


def load_checkpoint(checkpoint: Path, device: torch.device):
    ckpt = torch.load(str(checkpoint), map_location=device)
    arch = str(ckpt.get("arch", "nvidia"))
    height = int(ckpt.get("height", 80))
    width = int(ckpt.get("width", 160))
    crop_top_ratio = float(ckpt.get("crop_top_ratio", 0.35))
    model = build_model(arch, height=height, width=width, pretrained=False).to(device)
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    return model, {"arch": arch, "height": height, "width": width, "crop_top_ratio": crop_top_ratio}


def weighted_huber(pred: torch.Tensor, target: torch.Tensor, source: list[str], fig8_weight: float) -> torch.Tensor:
    loss = F.huber_loss(pred, target, delta=5.0, reduction="none")
    weights = torch.ones_like(loss)
    for idx, src in enumerate(source):
        if src == "figure8":
            weights[idx] = fig8_weight
    return (loss * weights).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> dict:
    model.eval()
    errors = []
    targets = []
    for images, target, _source in tqdm(loader, desc="val", leave=False):
        images = images.to(device)
        target = target.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred = model(images).clamp(ANGLE_MIN, ANGLE_MAX)
        errors.extend((pred - target).abs().detach().cpu().numpy().tolist())
        targets.extend(target.detach().cpu().numpy().tolist())
    if not errors:
        return {"mae": math.inf, "mae_left": math.nan, "mae_straight": math.nan, "mae_right": math.nan}
    err = np.asarray(errors, dtype=np.float32)
    tgt = np.asarray(targets, dtype=np.float32)
    return {
        "mae": float(err.mean()),
        "mae_left": float(err[tgt < 85].mean()) if np.any(tgt < 85) else math.nan,
        "mae_straight": float(err[(tgt >= 85) & (tgt <= 100)].mean()) if np.any((tgt >= 85) & (tgt <= 100)) else math.nan,
        "mae_right": float(err[tgt > 100].mean()) if np.any(tgt > 100) else math.nan,
    }


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp, fig8_weight: float) -> float:
    model.train()
    total = 0.0
    seen = 0
    for images, target, source in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred = model(images)
            loss = weighted_huber(pred, target, list(source), fig8_weight)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu()) * images.size(0)
        seen += images.size(0)
    return total / max(seen, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    model, meta = load_checkpoint(args.checkpoint, device)
    train_csv, old_val_csv, fig8_val_csv = make_splits(args, meta)

    train_ds = FineTuneLaneDataset(train_csv, meta["height"], meta["width"], meta["crop_top_ratio"], train=True)
    old_val_ds = FineTuneLaneDataset(old_val_csv, meta["height"], meta["width"], meta["crop_top_ratio"], train=False)
    fig8_val_ds = FineTuneLaneDataset(fig8_val_csv, meta["height"], meta["width"], meta["crop_top_ratio"], train=False)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=make_sampler(train_ds, args.fig8_weight),
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    old_val_loader = DataLoader(old_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    fig8_val_loader = DataLoader(fig8_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_score = math.inf
    logs = []
    best_path = args.out_dir / "best.pt"
    print(f"Fine-tuning {meta['arch']} from {args.checkpoint}")
    print(f"Train rows: {len(train_ds)}; old val: {len(old_val_ds)}; figure8 val: {len(fig8_val_ds)}")
    print(f"Device: {device}; AMP: {use_amp}; output: {args.out_dir}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp, args.fig8_weight)
        old_metrics = evaluate(model, old_val_loader, device, use_amp)
        fig8_metrics = evaluate(model, fig8_val_loader, device, use_amp)
        scheduler.step()
        score = 0.35 * old_metrics["mae"] + 0.65 * fig8_metrics["mae"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "old_mae": old_metrics["mae"],
            "fig8_mae": fig8_metrics["mae"],
            "score": score,
            "lr": optimizer.param_groups[0]["lr"],
        }
        logs.append(row)
        print(
            f"epoch {epoch:02d} train={train_loss:.3f} "
            f"old_mae={old_metrics['mae']:.2f} fig8_mae={fig8_metrics['mae']:.2f} score={score:.2f}"
        )
        if score < best_score:
            best_score = score
            torch.save({
                "model_state_dict": model.state_dict(),
                "arch": meta["arch"],
                "height": meta["height"],
                "width": meta["width"],
                "crop_top_ratio": meta["crop_top_ratio"],
                "epoch": epoch,
                "fine_tuned_from": str(args.checkpoint),
                "old_mae": old_metrics["mae"],
                "fig8_mae": fig8_metrics["mae"],
                "selection_score": score,
                "angle_min": ANGLE_MIN,
                "angle_max": ANGLE_MAX,
                "angle_straight": ANGLE_STRAIGHT,
            }, best_path)
            print(f"  saved {best_path}")

    pd.DataFrame(logs).to_csv(args.out_dir / "fine_tune_log.csv", index=False)


if __name__ == "__main__":
    main()
