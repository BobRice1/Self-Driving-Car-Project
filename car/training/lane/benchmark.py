"""
Benchmark lane model inference speed.

Usage:
    python -m car.training.lane.benchmark --checkpoint car/models/lane/checkpoints/lane_mobilenetv3s_*/best.pt
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .model import LaneFollower
from .transforms import IMAGENET_MEAN, IMAGENET_STD, LANE_HEIGHT, LANE_WIDTH


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark lane model inference.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to .pt checkpoint (optional).")
    parser.add_argument("--num_iterations", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--torchscript", action="store_true", help="Benchmark TorchScript version.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    model = LaneFollower(pretrained=False, dropout=0.0).to(device)
    if args.checkpoint and args.checkpoint.exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    model.eval()

    if args.torchscript:
        dummy = torch.randn(1, 3, LANE_HEIGHT, LANE_WIDTH, device=device)
        model = torch.jit.trace(model, dummy)
        print("Using TorchScript model.")

    dummy_input = torch.randn(1, 3, LANE_HEIGHT, LANE_WIDTH, device=device)

    for _ in range(10):
        with torch.no_grad():
            model(dummy_input)

    times = []
    for _ in range(args.num_iterations):
        start = time.perf_counter()
        with torch.no_grad():
            output = model(dummy_input)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    times = np.array(times)
    print(f"\nResults over {args.num_iterations} iterations on {device}:")
    print(f"  Mean:   {times.mean():.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Std:    {times.std():.2f} ms")
    print(f"  Min:    {times.min():.2f} ms")
    print(f"  Max:    {times.max():.2f} ms")
    print(f"  Output: angle_pred = {output.item():.4f}")

    param_count = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else "N/A"
    print(f"  Params: {param_count}")


if __name__ == "__main__":
    main()
