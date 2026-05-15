from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights, mobilenet_v3_large, mobilenet_v3_small
from tqdm import tqdm


ANGLE_MIN = 50.0
ANGLE_MAX = 120.0
ANGLE_STRAIGHT = 90.0
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact lane keeping CNN.")
    parser.add_argument("--splits-dir", type=Path, default=Path("car/data/lane_keeping/splits"))
    parser.add_argument("--out-dir", type=Path, default=Path("car/models/lane_keeping/runs/lane_nvidia"))
    parser.add_argument("--height", type=int, default=80)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--crop-top-ratio", type=float, default=0.35)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--arch", choices=["nvidia", "mobilenet_v3_small", "mobilenet_v3_large"], default="nvidia")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--right-turn-weight", type=float, default=1.0)
    parser.add_argument("--drive-frame-weight", type=float, default=1.0)
    parser.add_argument(
        "--selection-metric",
        choices=["val_mae", "bend_mae", "bend_right_mae", "combined"],
        default="bend_mae",
        help="Metric used to save best.pt. combined = 0.4*val_mae + 0.6*bend_mae.",
    )
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


class LaneDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        height: int,
        width: int,
        crop_top_ratio: float,
        train: bool,
        right_turn_weight: float = 1.0,
        drive_frame_weight: float = 1.0,
    ) -> None:
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.height = int(height)
        self.width = int(width)
        self.crop_top_ratio = float(crop_top_ratio)
        self.train = train
        self.right_turn_weight = float(right_turn_weight) if train else 1.0
        self.drive_frame_weight = float(drive_frame_weight) if train else 1.0

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
        target = np.float32(steering)
        weight = np.float32(1.0 + 2.0 * abs(steering - ANGLE_STRAIGHT) / 40.0)
        if steering > 95.0:
            weight *= np.float32(self.right_turn_weight)
        if str(row.get("source", "")) == "drive_frames":
            weight *= np.float32(self.drive_frame_weight)
        return torch.from_numpy(image), torch.tensor(target), torch.tensor(weight)

    def _augment(self, image: np.ndarray, steering: float) -> tuple[np.ndarray, float]:
        if random.random() < 0.6:
            alpha = random.uniform(0.80, 1.20)
            beta = random.uniform(-18.0, 18.0)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if random.random() < 0.25:
            image = cv2.GaussianBlur(image, (3, 3), 0)
        if random.random() < 0.45:
            shift = random.randint(-18, 18)
            matrix = np.float32([[1, 0, shift], [0, 1, random.randint(-4, 4)]])
            image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
            steering = steering + shift * 0.22
        if random.random() < 0.25:
            angle = random.uniform(-3.0, 3.0)
            centre = (image.shape[1] / 2.0, image.shape[0] / 2.0)
            matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
            steering = steering - angle * 0.35
        return image, float(np.clip(steering, ANGLE_MIN, ANGLE_MAX))


class NvidiaLaneNet(nn.Module):
    def __init__(self, height: int = 80, width: int = 160) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, int(height), int(width))
            flat = self.features(dummy).numel()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1)


class MobileNetLaneNet(nn.Module):
    def __init__(self, variant: str = "mobilenet_v3_large", pretrained: bool = False, dropout: float = 0.15) -> None:
        super().__init__()
        if variant == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = mobilenet_v3_small(weights=weights)
        elif variant == "mobilenet_v3_large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            backbone = mobilenet_v3_large(weights=weights)
        else:
            raise ValueError(f"Unsupported MobileNet variant: {variant}")

        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 64),
            nn.Hardswish(inplace=True),
            nn.Linear(64, 1),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)


def build_model(arch: str, height: int, width: int, pretrained: bool = False) -> nn.Module:
    if arch == "nvidia":
        return NvidiaLaneNet(height=height, width=width)
    return MobileNetLaneNet(variant=arch, pretrained=pretrained)


def make_sampler(ds: LaneDataset, right_turn_weight: float, drive_frame_weight: float) -> WeightedRandomSampler:
    angles = ds.df["steering"].astype(float).to_numpy()
    weights = 1.0 + 3.0 * np.abs(angles - ANGLE_STRAIGHT) / 40.0
    weights = weights * np.where(angles > 95.0, float(right_turn_weight), 1.0)
    if "source" in ds.df.columns:
        sources = ds.df["source"].astype(str).to_numpy()
        weights = weights * np.where(sources == "drive_frames", float(drive_frame_weight), 1.0)
    return WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)


def weighted_huber(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = F.huber_loss(pred, target, delta=5.0, reduction="none")
    return (loss * weight).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> dict:
    model.eval()
    rows = []
    for images, target, _ in tqdm(loader, desc="val", leave=False):
        images = images.to(device)
        target = target.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred = model(images).clamp(ANGLE_MIN, ANGLE_MAX)
        error = (pred - target).abs()
        for p, t, e in zip(pred.cpu().numpy(), target.cpu().numpy(), error.cpu().numpy()):
            rows.append((float(p), float(t), float(e)))
    if not rows:
        return {"mae": math.inf}
    arr = np.array(rows, dtype=np.float32)
    target = arr[:, 1]
    error = arr[:, 2]
    return {
        "mae": float(error.mean()),
        "mae_left": float(error[target < 85].mean()) if np.any(target < 85) else float("nan"),
        "mae_straight": float(error[(target >= 85) & (target <= 95)].mean()) if np.any((target >= 85) & (target <= 95)) else float("nan"),
        "mae_right": float(error[target > 95].mean()) if np.any(target > 95) else float("nan"),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    total = 0.0
    seen = 0
    for images, target, weight in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        weight = weight.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred = model(images)
            loss = weighted_huber(pred, target, weight)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu()) * images.size(0)
        seen += images.size(0)
    return total / max(seen, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = LaneDataset(
        args.splits_dir / "train.csv",
        args.height,
        args.width,
        args.crop_top_ratio,
        train=True,
        right_turn_weight=args.right_turn_weight,
        drive_frame_weight=args.drive_frame_weight,
    )
    val_ds = LaneDataset(args.splits_dir / "val.csv", args.height, args.width, args.crop_top_ratio, train=False)
    bend_csv = args.splits_dir / "val_bend.csv"
    bend_ds = LaneDataset(bend_csv, args.height, args.width, args.crop_top_ratio, train=False) if bend_csv.exists() else None
    sampler = None if args.no_weighted_sampler else make_sampler(train_ds, args.right_turn_weight, args.drive_frame_weight)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    bend_loader = (
        DataLoader(
            bend_ds,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        if bend_ds is not None
        else None
    )

    model = build_model(args.arch, args.height, args.width, pretrained=args.pretrained).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    use_amp = device.type == "cuda" and not args.disable_amp
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"Device: {device} ({device_name}), AMP: {use_amp}")

    best_score = float("inf")
    logs = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        metrics = evaluate(model, val_loader, device, use_amp)
        bend_metrics = evaluate(model, bend_loader, device, use_amp) if bend_loader is not None else {}
        scheduler.step()
        val_mae = metrics["mae"]
        bend_mae = bend_metrics.get("mae", float("nan"))
        bend_right_mae = bend_metrics.get("mae_right", float("nan"))
        if args.selection_metric == "val_mae" or not np.isfinite(bend_mae):
            score = val_mae
        elif args.selection_metric == "bend_right_mae" and np.isfinite(bend_right_mae):
            score = bend_right_mae
        elif args.selection_metric == "combined":
            score = 0.4 * val_mae + 0.6 * bend_mae
        else:
            score = bend_mae

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "selection_score": score,
            **{f"val_{k}": v for k, v in metrics.items()},
            **{f"bend_{k}": v for k, v in bend_metrics.items()},
        }
        logs.append(row)
        print(
            f"epoch={epoch:03d} train={train_loss:.4f} val_mae={val_mae:.2f} "
            f"bend_mae={bend_mae:.2f} score={score:.2f} "
            f"bend_left={bend_metrics.get('mae_left', float('nan')):.2f} "
            f"bend_straight={bend_metrics.get('mae_straight', float('nan')):.2f} "
            f"bend_right={bend_metrics.get('mae_right', float('nan')):.2f}"
        )
        if score < best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "height": args.height,
                    "width": args.width,
                    "crop_top_ratio": args.crop_top_ratio,
                    "arch": args.arch,
                    "pretrained": args.pretrained,
                    "output_mode": "angle",
                    "epoch": epoch,
                    "selection_metric": args.selection_metric,
                    "selection_score": best_score,
                    "val_metrics": metrics,
                    "bend_metrics": bend_metrics,
                },
                args.out_dir / "best.pt",
            )
            print(f"  saved {args.out_dir / 'best.pt'}")
    pd.DataFrame(logs).to_csv(args.out_dir / "train_log.csv", index=False)


if __name__ == "__main__":
    main()
