"""Small TensorFlow Lite helpers used by the Pi-car inference wrappers."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def make_interpreter(model_path: str | Path, use_edgetpu: bool = False):
    if use_edgetpu:
        from pycoral.utils.edgetpu import make_interpreter as make_edgetpu_interpreter

        return make_edgetpu_interpreter(str(model_path))

    try:
        import tflite_runtime.interpreter as tflite

        return tflite.Interpreter(model_path=str(model_path))
    except ImportError:
        import tensorflow as tf

        return tf.lite.Interpreter(model_path=str(model_path))


def quantize_input(array: np.ndarray, input_detail: dict) -> np.ndarray:
    dtype = input_detail["dtype"]
    if dtype == np.float32:
        return array.astype(np.float32)

    scale, zero_point = input_detail.get("quantization", (0.0, 0))
    if scale == 0:
        return array.astype(dtype)
    return np.round(array / scale + zero_point).astype(dtype)


def dequantize_output(array: np.ndarray, output_detail: dict) -> np.ndarray:
    if array.dtype == np.float32:
        return array

    scale, zero_point = output_detail.get("quantization", (0.0, 0))
    if scale == 0:
        return array.astype(np.float32)
    return (array.astype(np.float32) - zero_point) * scale
