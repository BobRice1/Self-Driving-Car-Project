"""
Export trained PyTorch checkpoints to ONNX as an intermediate deployment format.

The ONNX files can then be converted to TensorFlow Lite on Linux/Colab with
car.training.convert_onnx_to_tflite. Keeping this as a separate step avoids
Windows-specific LiteRT converter issues.

Usage:
    python -m car.training.export_onnx --all
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PyTorch checkpoints to ONNX.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--type", choices=["lane", "event"], default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--input_size", type=int, default=96)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def strip_state_dict(state_dict: dict) -> dict:
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def load_lane(checkpoint_path: Path):
    from car.training.lane.model import LaneFollower

    model = LaneFollower(pretrained=False, dropout=0.0)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(strip_state_dict(ckpt["model_state_dict"]))
    model.eval()
    return model, torch.randn(1, 3, 120, 160), ["lane_angle_norm"]


def load_event(checkpoint_path: Path, num_classes: int, input_size: int):
    from car.training.events.model import EventClassifier

    model = EventClassifier(num_classes=num_classes, pretrained=False, dropout=0.0)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(strip_state_dict(ckpt["model_state_dict"]))
    model.eval()
    return model, torch.randn(1, 3, input_size, input_size), ["logits"]


def export_one(model_type: str, checkpoint_path: Path, output_path: Path, num_classes: int, input_size: int, opset: int):
    if model_type == "lane":
        model, dummy, output_names = load_lane(checkpoint_path)
    else:
        model, dummy, output_names = load_event(checkpoint_path, num_classes, input_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {model_type}: {checkpoint_path} -> {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            export_params=True,
            opset_version=opset,
            external_data=False,
            do_constant_folding=True,
            input_names=["input"],
            output_names=output_names,
        )
    print(f"Saved {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


def main() -> None:
    args = parse_args()
    if args.all:
        export_one(
            "lane",
            Path("car/checkpoints/lane_best.pt"),
            Path("car/checkpoints/lane_best.onnx"),
            args.num_classes,
            args.input_size,
            args.opset,
        )
        export_one(
            "event",
            Path("car/checkpoints/arrow_best.pt"),
            Path("car/checkpoints/arrow_best.onnx"),
            args.num_classes,
            args.input_size,
            args.opset,
        )
        return

    if args.type is None or args.checkpoint is None or args.output is None:
        raise SystemExit("Use --all, or provide --type, --checkpoint, and --output.")
    export_one(args.type, args.checkpoint, args.output, args.num_classes, args.input_size, args.opset)


if __name__ == "__main__":
    main()
