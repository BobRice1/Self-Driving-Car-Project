"""
Convert exported ONNX models to TensorFlow Lite.

Run this on Linux/Colab/MLIS if the LiteRT converter is unavailable on Windows.

Setup example:
    pip install onnx onnx2tf tensorflow==2.15.*

Usage:
    python -m car.training.convert_onnx_to_tflite --onnx car/checkpoints/lane_best.onnx --output car/checkpoints/lane_best.tflite
    python -m car.training.convert_onnx_to_tflite --onnx car/checkpoints/arrow_best.onnx --output car/checkpoints/arrow_best.tflite
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorFlow Lite.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--saved_model_dir", type=Path, default=None)
    return parser.parse_args()


def ensure_onnx2tf_sample_data() -> None:
    """Avoid onnx2tf downloading its sample calibration array at conversion time."""
    import numpy as np

    path = Path.cwd() / "calibration_image_sample_data_20x128x128x3_float32.npy"
    if not path.exists():
        np.save(path, np.random.rand(20, 128, 128, 3).astype(np.float32))


def convert_onnx_to_saved_model(onnx_path: Path, saved_model_dir: Path) -> None:
    ensure_onnx2tf_sample_data()
    try:
        from onnx2tf import convert

        convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(saved_model_dir),
            non_verbose=True,
        )
        return
    except Exception as exc:
        print(f"onnx2tf Python API failed, trying CLI: {exc}")

    exe = shutil.which("onnx2tf")
    if exe is None:
        raise RuntimeError("onnx2tf is not installed or is not on PATH.")

    subprocess.run(
        [exe, "-i", str(onnx_path), "-o", str(saved_model_dir), "-osd"],
        check=True,
    )


def convert_saved_model_to_tflite(saved_model_dir: Path, output_path: Path) -> None:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    print(f"Saved {output_path} ({len(tflite_model) / 1024 / 1024:.1f} MB)")


def main() -> None:
    args = parse_args()
    if args.saved_model_dir is None:
        with tempfile.TemporaryDirectory() as tmp:
            saved_model_dir = Path(tmp) / args.onnx.stem
            convert_onnx_to_saved_model(args.onnx, saved_model_dir)
            convert_saved_model_to_tflite(saved_model_dir, args.output)
    else:
        convert_onnx_to_saved_model(args.onnx, args.saved_model_dir)
        convert_saved_model_to_tflite(args.saved_model_dir, args.output)


if __name__ == "__main__":
    main()
