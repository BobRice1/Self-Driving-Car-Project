"""
Export trained PyTorch checkpoints to TensorFlow Lite / LiteRT flatbuffers.

The project car guidance says the Raspberry Pi has TensorFlow 2.15 and the
Coral TPU uses TensorFlow Lite. This exporter converts the already-trained
PyTorch lane and arrow checkpoints into .tflite files for deployment.

Requires one of:
    pip install ai-edge-torch
    pip install litert-torch

Usage:
    python -m car.training.export_litert --all
    python -m car.training.export_litert --type lane --checkpoint car/checkpoints/lane_best.pt --output car/checkpoints/lane_best.tflite
    python -m car.training.export_litert --type event --checkpoint car/checkpoints/arrow_best.pt --output car/checkpoints/arrow_best.tflite --num_classes 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PyTorch checkpoints to TFLite/LiteRT.")
    parser.add_argument("--all", action="store_true", help="Export lane and arrow default checkpoints.")
    parser.add_argument("--type", type=str, choices=["lane", "event"], default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num_classes", type=int, default=2, help="For event models only.")
    parser.add_argument("--input_size", type=int, default=96, help="For event models only.")
    return parser.parse_args()


def load_converter_module():
    try:
        import litert_torch

        return litert_torch
    except ImportError:
        try:
            import ai_edge_torch

            return ai_edge_torch
        except ImportError as exc:
            raise SystemExit(
                "Missing PyTorch-to-TFLite converter. Install one of:\n"
                "  pip install litert-torch\n"
                "  pip install ai-edge-torch"
            ) from exc


def strip_state_dict(state_dict: dict) -> dict:
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def load_lane_model(checkpoint_path: Path):
    from car.training.lane.model import LaneFollower

    model = LaneFollower(pretrained=False, dropout=0.0)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(strip_state_dict(ckpt["model_state_dict"]))
    model.eval()
    return model, (torch.randn(1, 3, 120, 160),)


def load_event_model(checkpoint_path: Path, num_classes: int, input_size: int):
    from car.training.events.model import EventClassifier

    model = EventClassifier(num_classes=num_classes, pretrained=False, dropout=0.0)
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(strip_state_dict(ckpt["model_state_dict"]))
    model.eval()
    return model, (torch.randn(1, 3, input_size, input_size),)


def export_one(model_type: str, checkpoint_path: Path, output_path: Path, num_classes: int, input_size: int) -> None:
    converter = load_converter_module()
    if model_type == "lane":
        model, sample_inputs = load_lane_model(checkpoint_path)
    else:
        model, sample_inputs = load_event_model(checkpoint_path, num_classes, input_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Converting {model_type}: {checkpoint_path} -> {output_path}")
    with torch.no_grad():
        edge_model = converter.convert(model, sample_inputs)
    edge_model.export(str(output_path))
    print(f"Saved {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


def main() -> None:
    args = parse_args()
    if args.all:
        export_one(
            "lane",
            Path("car/checkpoints/lane_best.pt"),
            Path("car/checkpoints/lane_best.tflite"),
            args.num_classes,
            args.input_size,
        )
        export_one(
            "event",
            Path("car/checkpoints/arrow_best.pt"),
            Path("car/checkpoints/arrow_best.tflite"),
            args.num_classes,
            args.input_size,
        )
        return

    if args.type is None or args.checkpoint is None or args.output is None:
        raise SystemExit("Use --all, or provide --type, --checkpoint, and --output.")

    export_one(args.type, args.checkpoint, args.output, args.num_classes, args.input_size)


if __name__ == "__main__":
    main()
