from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fine_tune_lane_model import FineTuneLaneDataset, evaluate
from train_lane_model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a NewModel lane checkpoint on a CSV split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(str(args.checkpoint), map_location="cpu")
    arch = str(ckpt.get("arch", "nvidia"))
    height = int(ckpt.get("height", 80))
    width = int(ckpt.get("width", 160))
    crop_top_ratio = float(ckpt.get("crop_top_ratio", 0.35))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = device.type == "cuda"

    model = build_model(arch, height=height, width=width, pretrained=False)
    model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()})
    model.to(device)

    dataset = FineTuneLaneDataset(args.csv, height, width, crop_top_ratio, train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate(model, loader, device, use_amp)

    print(f"checkpoint={args.checkpoint}")
    print(f"arch={arch} rows={len(dataset)} device={device}")
    for key in ("mae", "mae_left", "mae_straight", "mae_right"):
        print(f"{key}={metrics[key]:.4f}")


if __name__ == "__main__":
    main()
