"""
Export trained PyTorch models to TorchScript for faster inference on the Pi.

Usage:
    python -m car.training.export_torchscript --type lane --checkpoint car/checkpoints/.../best.pt --output car/checkpoints/lane_best_ts.pt
    python -m car.training.export_torchscript --type event --checkpoint car/checkpoints/.../best.pt --output car/checkpoints/arrow_best_ts.pt --num_classes 3 --input_size 96
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Export model to TorchScript.")
    parser.add_argument("--type", type=str, required=True, choices=["lane", "event"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_classes", type=int, default=3, help="For event models only.")
    parser.add_argument("--input_size", type=int, default=96, help="For event models only.")
    parser.add_argument("--benchmark", action="store_true", help="Run speed benchmark after export.")
    return parser.parse_args()


def export_lane(checkpoint_path: Path, output_path: Path):
    from car.training.lane.model import LaneFollower

    model = LaneFollower(pretrained=False, dropout=0.0)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.randn(1, 3, 120, 160)
    traced = torch.jit.trace(model, dummy)
    traced.save(str(output_path))
    return traced, dummy


def export_event(checkpoint_path: Path, output_path: Path, num_classes: int, input_size: int):
    from car.training.events.model import EventClassifier

    model = EventClassifier(num_classes=num_classes, pretrained=False, dropout=0.0)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.randn(1, 3, input_size, input_size)
    traced = torch.jit.trace(model, dummy)
    traced.save(str(output_path))
    return traced, dummy


def benchmark(model, dummy_input, iterations=200):
    for _ in range(20):
        with torch.no_grad():
            model(dummy_input)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        with torch.no_grad():
            model(dummy_input)
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    print(f"  Mean:   {times.mean():.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Min:    {times.min():.2f} ms")
    print(f"  Max:    {times.max():.2f} ms")


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.type == "lane":
        traced, dummy = export_lane(args.checkpoint, args.output)
    else:
        traced, dummy = export_event(args.checkpoint, args.output, args.num_classes, args.input_size)

    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"Exported TorchScript model: {args.output} ({size_mb:.1f} MB)")

    if args.benchmark:
        print("\nBenchmark (CPU):")
        benchmark(traced, dummy)


if __name__ == "__main__":
    main()
