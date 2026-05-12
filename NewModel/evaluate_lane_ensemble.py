from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fine_tune_lane_model import FineTuneLaneDataset
from train_lane_model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate averaged lane checkpoints on a CSV split.")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def load_model(path: Path, device: torch.device):
    ckpt = torch.load(str(path), map_location="cpu")
    arch = str(ckpt.get("arch", "nvidia"))
    height = int(ckpt.get("height", 80))
    width = int(ckpt.get("width", 160))
    crop_top_ratio = float(ckpt.get("crop_top_ratio", 0.35))
    model = build_model(arch, height=height, width=width, pretrained=False)
    model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()})
    model.to(device)
    model.eval()
    return model, {"arch": arch, "height": height, "width": width, "crop_top_ratio": crop_top_ratio}


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = device.type == "cuda"

    loaded = [load_model(path, device) for path in args.checkpoint]
    models = [item[0] for item in loaded]
    meta = loaded[0][1]
    for _, other in loaded[1:]:
        if (
            other["height"] != meta["height"]
            or other["width"] != meta["width"]
            or not math.isclose(other["crop_top_ratio"], meta["crop_top_ratio"])
        ):
            raise SystemExit("All checkpoints must use the same input shape and crop_top_ratio.")

    dataset = FineTuneLaneDataset(args.csv, meta["height"], meta["width"], meta["crop_top_ratio"], train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    preds = []
    targets = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            images, angles = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                batch_preds = torch.stack([model(images) for model in models], dim=0).mean(dim=0)
            preds.append(batch_preds.detach().cpu().numpy())
            targets.append(angles.numpy())

    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    error = np.abs(pred - target)

    print("checkpoints=" + ",".join(str(path) for path in args.checkpoint))
    print(f"rows={len(dataset)} device={device}")
    print(f"mae={float(error.mean()):.4f}")
    print(f"mae_left={float(error[target < 85].mean()) if np.any(target < 85) else float('nan'):.4f}")
    print(
        f"mae_straight={float(error[(target >= 85) & (target <= 100)].mean()) if np.any((target >= 85) & (target <= 100)) else float('nan'):.4f}"
    )
    print(f"mae_right={float(error[target > 100].mean()) if np.any(target > 100) else float('nan'):.4f}")


if __name__ == "__main__":
    main()
