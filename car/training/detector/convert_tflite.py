"""
Download a pre-trained SSD MobileNet V2 from TF Hub and export to TFLite (INT8).

The Edge TPU compiler can then be run on the output:
    edgetpu_compiler car/models/obstacle/checkpoints/obstacle_detector.tflite

Usage:
    python -m car.training.detector.convert_tflite --output car/models/obstacle/checkpoints/obstacle_detector.tflite
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Export SSD MobileNet V2 to INT8 TFLite.")
    parser.add_argument(
        "--model_url",
        type=str,
        default="https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2",
        help="TF Hub model URL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("car/models/obstacle/checkpoints/obstacle_detector.tflite"),
    )
    parser.add_argument("--input_size", type=int, default=320, help="Model input size (square).")
    parser.add_argument("--num_calibration", type=int, default=100, help="Calibration samples for INT8.")
    return parser.parse_args()


def representative_dataset_gen(input_size: int, num_samples: int):
    """Generate random calibration data. Replace with real images for better accuracy."""
    for _ in range(num_samples):
        data = np.random.randint(0, 256, (1, input_size, input_size, 3)).astype(np.uint8)
        yield [data]


def main():
    args = parse_args()

    try:
        import tensorflow as tf
    except ImportError:
        print(
            "TensorFlow is required for TFLite conversion.\n"
            "Install with: pip install tensorflow\n"
            "This is NOT needed on the Pi at runtime -- only for the conversion step."
        )
        return

    print(f"Loading model from {args.model_url} ...")
    import tensorflow_hub as hub

    detector = hub.load(args.model_url)

    input_size = args.input_size

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, input_size, input_size, 3], dtype=tf.uint8)])
    def detect(image):
        return detector(image)

    concrete_func = detect.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(input_size, args.num_calibration)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32

    print("Converting to INT8 TFLite ...")
    tflite_model = converter.convert()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(tflite_model)
    print(f"Saved TFLite model: {args.output} ({len(tflite_model) / 1024 / 1024:.1f} MB)")
    print(
        "\nNext step: compile for Edge TPU:\n"
        f"  edgetpu_compiler {args.output}"
    )


if __name__ == "__main__":
    main()
