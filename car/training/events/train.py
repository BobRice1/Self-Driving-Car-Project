"""
Event classifier training for arrow/obstacle events.

Usage:
    python -m car.training.events.train --config car/configs/events.yaml --task arrow
    python -m car.training.events.train --config car/configs/obstacle_classifier.yaml --task obstacle
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .dataset import (
    EventDataset,
    build_event_train_transforms,
    build_event_valid_transforms,
    discover_classes,
)
from .model import EventClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train event classifier.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", type=str, default="arrow")
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(mode: str) -> torch.device:
    if mode == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(mode)


def create_warmup_cosine_scheduler(optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, use_amp, class_weights):
    model.train()
    running_loss = 0.0
    correct = 0
    seen = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.detach().item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += batch_size

    return running_loss / max(seen, 1), correct / max(seen, 1)


@torch.no_grad()
def validate(model, loader, device, use_amp):
    model.eval()
    running_loss = 0.0
    correct = 0
    seen = 0

    for images, labels in tqdm(loader, desc="valid", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)

        running_loss += loss.detach().item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += batch_size

    return running_loss / max(seen, 1), correct / max(seen, 1)


def main() -> None:
    args = parse_args()
    full_config = load_yaml(args.config)
    cfg = full_config[args.task]
    set_seed(args.seed)

    device = resolve_device(args.device)
    data_dir = Path(cfg["data_dir"])
    classes = cfg["classes"]
    input_size = int(cfg.get("input_size", 96))

    full_ds = EventDataset(
        root=data_dir,
        classes=classes,
        transform=None,
    )

    val_ratio = float(cfg.get("val_ratio", 0.2))
    indices = list(range(len(full_ds)))
    labels_for_split = [full_ds.labels[i] for i in indices]
    train_idx, val_idx = train_test_split(
        indices, test_size=val_ratio, stratify=labels_for_split, random_state=args.seed,
    )

    train_ds = EventDataset(
        root=data_dir,
        classes=classes,
        transform=build_event_train_transforms(input_size, hflip=bool(cfg.get("hflip", False))),
    )
    valid_ds = EventDataset(root=data_dir, classes=classes, transform=build_event_valid_transforms(input_size))
    train_subset = Subset(train_ds, train_idx)
    valid_subset = Subset(valid_ds, val_idx)
    train_label_counts = np.bincount([full_ds.labels[i] for i in train_idx], minlength=len(classes))
    class_weights_np = len(train_idx) / np.maximum(train_label_counts, 1)
    class_weights_np = class_weights_np / class_weights_np.mean()

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_subset, batch_size=int(cfg.get("batch_size", 32)),
        shuffle=True, num_workers=args.num_workers, pin_memory=pin, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_subset, batch_size=int(cfg.get("batch_size", 32)),
        shuffle=False, num_workers=args.num_workers, pin_memory=pin,
    )

    num_classes = len(classes)
    model = EventClassifier(
        num_classes=num_classes,
        pretrained=bool(cfg.get("pretrained", True)),
        dropout=float(cfg.get("dropout", 0.2)),
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(cfg.get("epochs", 30))
    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(cfg.get("warmup_epochs", 3)) * steps_per_epoch
    scheduler = create_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps)

    use_amp = not args.disable_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_dir or Path(f"car/models/{args.task}/checkpoints")
    out_dir = out_root / f"{args.task}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience = int(cfg.get("early_stopping_patience", 8))
    bad_epochs = 0
    ckpt_path = out_dir / "best.pt"

    print(f"Task: {args.task}, Classes: {classes}, Samples: {len(full_ds)}")
    print(f"Train: {len(train_subset)}, Valid: {len(valid_subset)}")
    print(f"Train class counts: {dict(zip(classes, train_label_counts.tolist()))}")
    print(f"Device: {device}, AMP: {use_amp}")
    print(f"Output: {out_dir}")

    logs: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, use_amp, class_weights,
        )
        val_loss, val_acc = validate(model, valid_loader, device, use_amp)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} lr={lr:.2e}"
        )
        logs.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": lr,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "classes": classes,
                "task": args.task,
            }, ckpt_path)
            print(f"  -> Saved best (val_loss={val_loss:.4f}, val_acc={val_acc:.3f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    pd.DataFrame(logs).to_csv(out_dir / "train_log.csv", index=False)
    print(f"Training complete. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
