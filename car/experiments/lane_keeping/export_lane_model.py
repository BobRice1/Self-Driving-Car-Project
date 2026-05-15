from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train_lane_model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export lane keeping checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--tflite", type=Path, default=None)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def load_model(checkpoint: Path):
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    height = int(ckpt.get("height", 80))
    width = int(ckpt.get("width", 160))
    arch = str(ckpt.get("arch", "nvidia"))
    model = build_model(arch, height=height, width=width, pretrained=False)
    model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()})
    model.eval()
    return model, (height, width)


def export_onnx(model, size: tuple[int, int], output: Path, opset: int) -> None:
    height, width = size
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, height, width)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["steering_angle"],
            dynamo=False,
        )
    print(f"Saved ONNX: {output}")


def export_tflite(model, size: tuple[int, int], output: Path) -> None:
    import_errors = []
    try:
        import litert_torch as converter
        if hasattr(converter, "convert"):
            pass
        else:
            raise ImportError("litert_torch has no convert() function")
    except Exception as exc:
        import_errors.append(f"litert_torch: {exc}")
        try:
            import ai_edge_torch as converter
            if not hasattr(converter, "convert"):
                raise ImportError("ai_edge_torch has no convert() function")
        except Exception as fallback_exc:
            import_errors.append(f"ai_edge_torch: {fallback_exc}")
            raise SystemExit(
                "Direct TFLite export is unavailable in this Python environment.\n"
                "Use --onnx and convert the ONNX model with the project ONNX-to-TFLite workflow, "
                "or install a working litert-torch stack.\n"
                + "\n".join(import_errors)
            ) from fallback_exc

    height, width = size
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_inputs = (torch.randn(1, 3, height, width),)
    with torch.no_grad():
        edge_model = converter.convert(model, sample_inputs)
    edge_model.export(str(output))
    print(f"Saved TFLite: {output}")


def main() -> None:
    args = parse_args()
    if args.onnx is None and args.tflite is None:
        raise SystemExit("Provide --onnx, --tflite, or both.")
    model, size = load_model(args.checkpoint)
    if args.onnx is not None:
        export_onnx(model, size, args.onnx, args.opset)
    if args.tflite is not None:
        export_tflite(model, size, args.tflite)


if __name__ == "__main__":
    main()
