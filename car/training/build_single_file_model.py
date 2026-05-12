"""
Build a self-contained autopilot model.py with TFLite models embedded.

The generated file is intended for the MLiS autopilot skeleton where the model
submission is expected to be a single Python file defining a Model class.

Usage:
    python -m car.training.build_single_file_model --output deploy/picar_single_file/model.py
"""
from __future__ import annotations

import argparse
import base64
from pathlib import Path
from string import Template
import textwrap
import zlib


TEMPLATE = Template(
    r'''"""
Self-contained MLiS PiCar autopilot model.

This file embeds the trained lane, arrow, and obstacle TFLite models so it can be
submitted/copied as a single model.py. It expects OpenCV, NumPy, TensorFlow Lite
via TensorFlow or tflite_runtime, and PyCoral for Edge TPU acceleration.
"""
from __future__ import annotations

import base64
import os
import tempfile
import zlib

import cv2
import numpy as np


LANE_MODEL_B64 = """
$lane_model_b64
"""

ARROW_MODEL_B64 = """
$arrow_model_b64
"""

OBSTACLE_CPU_MODEL_B64 = """
$obstacle_cpu_model_b64
"""

OBSTACLE_EDGETPU_MODEL_B64 = """
$obstacle_edgetpu_model_b64
"""


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CROP_RATIO = 0.4
ANGLE_MIN = 50
ANGLE_MAX = 120
CRUISE_SPEED = 35

ARROW_CLASSES = ["left", "right"]
ARROW_CONFIDENCE_THRESHOLD = 0.85
ARROW_ROI = (0, 0, 320, 120)
EVENT_INTERVAL = 5
OBSTACLE_INTERVAL = 5

COCO_PERSON_CLASS = 0
LANE_REGION_X_MIN = 0.25
LANE_REGION_X_MAX = 0.75
LANE_REGION_Y_MIN = 0.4


def _decode_model(data: str) -> bytes:
    return zlib.decompress(base64.b64decode(data))


def _make_interpreter(model_bytes: bytes):
    try:
        import tflite_runtime.interpreter as tflite

        return tflite.Interpreter(model_content=model_bytes)
    except ImportError:
        import tensorflow as tf

        return tf.lite.Interpreter(model_content=model_bytes)


def _quantize_input(array: np.ndarray, input_detail: dict) -> np.ndarray:
    dtype = input_detail["dtype"]
    if dtype == np.float32:
        return array.astype(np.float32)
    scale, zero_point = input_detail.get("quantization", (0.0, 0))
    if scale == 0:
        return array.astype(dtype)
    return np.round(array / scale + zero_point).astype(dtype)


def _dequantize_output(array: np.ndarray, output_detail: dict) -> np.ndarray:
    if array.dtype == np.float32:
        return array
    scale, zero_point = output_detail.get("quantization", (0.0, 0))
    if scale == 0:
        return array.astype(np.float32)
    return (array.astype(np.float32) - zero_point) * scale


def _normalised_to_car_angle(angle_norm: float) -> int:
    clipped = max(0.0, min(1.0, float(angle_norm)))
    return int(round(ANGLE_MIN + clipped * (ANGLE_MAX - ANGLE_MIN)))


def _decide(lane_angle_norm: float, arrow: str, obstacle_in_lane: bool) -> tuple[int, int]:
    angle = _normalised_to_car_angle(lane_angle_norm)
    if obstacle_in_lane:
        return angle, 0
    if arrow == "left":
        angle = max(ANGLE_MIN, angle - 15)
    elif arrow == "right":
        angle = min(ANGLE_MAX, angle + 15)
    return angle, CRUISE_SPEED


class _TFLiteLanePredictor:
    def __init__(self, model_bytes: bytes) -> None:
        self.interpreter = _make_interpreter(model_bytes)
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        shape = [int(v) for v in self.input_detail["shape"]]
        self.channels_first = shape[1] == 3
        if self.channels_first:
            self.height = shape[2]
            self.width = shape[3]
        else:
            self.height = shape[1]
            self.width = shape[2]

    def predict(self, bgr_frame: np.ndarray) -> float:
        h = bgr_frame.shape[0]
        crop_start = int(h * CROP_RATIO)
        cropped = bgr_frame[crop_start:, :, :]
        resized = cv2.resize(cropped, (self.width, self.height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normed = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        if self.channels_first:
            normed = normed.transpose(2, 0, 1)
        batched = np.expand_dims(normed, axis=0)
        tensor = _quantize_input(batched, self.input_detail)
        self.interpreter.set_tensor(self.input_detail["index"], tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_detail["index"])
        output = _dequantize_output(output, self.output_detail)
        return float(np.clip(output.reshape(-1)[0], 0.0, 1.0))


class _TFLiteArrowPredictor:
    def __init__(self, model_bytes: bytes) -> None:
        self.interpreter = _make_interpreter(model_bytes)
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        shape = [int(v) for v in self.input_detail["shape"]]
        self.channels_first = shape[1] == 3
        if self.channels_first:
            self.height = shape[2]
            self.width = shape[3]
        else:
            self.height = shape[1]
            self.width = shape[2]

    def predict_with_confidence(self, bgr_roi: np.ndarray) -> tuple[str, float]:
        resized = cv2.resize(bgr_roi, (self.width, self.height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normed = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        if self.channels_first:
            normed = normed.transpose(2, 0, 1)
        batched = np.expand_dims(normed, axis=0)
        tensor = _quantize_input(batched, self.input_detail)
        self.interpreter.set_tensor(self.input_detail["index"], tensor)
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(self.output_detail["index"])
        logits = _dequantize_output(logits, self.output_detail).reshape(-1)
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        probs = exp / np.maximum(exp.sum(), 1e-12)
        idx = int(np.argmax(probs))
        return ARROW_CLASSES[idx], float(probs[idx])


class _ObstacleDetector:
    def __init__(self, cpu_model_bytes: bytes, edgetpu_model_bytes: bytes, score_threshold: float = 0.4) -> None:
        self.score_threshold = score_threshold
        self.use_edgetpu = False
        self.interpreter = None
        try:
            if os.path.exists("/dev/bus/usb"):
                from pycoral.utils.edgetpu import make_interpreter

                model_path = self._write_temp_model(edgetpu_model_bytes, "mlis_obstacle_detector_edgetpu.tflite")
                self.interpreter = make_interpreter(model_path)
                self.use_edgetpu = True
        except Exception as exc:
            print(f"[model] Edge TPU unavailable, using CPU obstacle model: {exc}")

        if self.interpreter is None:
            self.interpreter = _make_interpreter(cpu_model_bytes)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        input_shape = self.input_details[0]["shape"]
        self.input_height = int(input_shape[1])
        self.input_width = int(input_shape[2])

    @staticmethod
    def _write_temp_model(model_bytes: bytes, filename: str) -> str:
        path = os.path.join(tempfile.gettempdir(), filename)
        if not os.path.exists(path) or os.path.getsize(path) != len(model_bytes):
            with open(path, "wb") as f:
                f.write(model_bytes)
        return path

    def detect(self, bgr_frame: np.ndarray) -> list[dict]:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_width, self.input_height))
        input_data = np.expand_dims(resized, axis=0).astype(self.input_details[0]["dtype"])
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]

        results = []
        for i in range(len(scores)):
            if scores[i] < self.score_threshold:
                continue
            class_id = int(classes[i])
            if class_id != COCO_PERSON_CLASS:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            results.append({
                "box": (float(xmin), float(ymin), float(xmax), float(ymax)),
                "class_id": class_id,
                "score": float(scores[i]),
            })
        return results

    @staticmethod
    def is_in_lane(box: tuple[float, float, float, float]) -> bool:
        xmin, ymin, xmax, ymax = box
        x_overlap = xmin < LANE_REGION_X_MAX and xmax > LANE_REGION_X_MIN
        y_overlap = ymax > LANE_REGION_Y_MIN
        return x_overlap and y_overlap

    def detect_obstacle_in_lane(self, bgr_frame: np.ndarray) -> bool:
        return any(self.is_in_lane(det["box"]) for det in self.detect(bgr_frame))


class Model:
    def __init__(self):
        self.lane = _TFLiteLanePredictor(_decode_model(LANE_MODEL_B64))
        print("[model] Lane TFLite model loaded.")
        self.arrow = _TFLiteArrowPredictor(_decode_model(ARROW_MODEL_B64))
        print("[model] Arrow TFLite classifier loaded.")
        self.obstacle = _ObstacleDetector(
            cpu_model_bytes=_decode_model(OBSTACLE_CPU_MODEL_B64),
            edgetpu_model_bytes=_decode_model(OBSTACLE_EDGETPU_MODEL_B64),
        )
        print(f"[model] Obstacle detector loaded (edgetpu={self.obstacle.use_edgetpu}).")
        self.frame_count = 0
        self.last_arrow = "none"
        self.last_obstacle_in_lane = False

    @staticmethod
    def _extract_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = roi
        return image[y1:y2, x1:x2].copy()

    def predict(self, image: np.ndarray) -> tuple[int, int]:
        return self.predict_debug(image)["decision"]

    def predict_debug(self, image: np.ndarray) -> dict:
        lane_angle_norm = self.lane.predict(image)

        if self.frame_count % EVENT_INTERVAL == 0:
            arrow_roi = self._extract_roi(image, ARROW_ROI)
            arrow, confidence = self.arrow.predict_with_confidence(arrow_roi)
            self.last_arrow = arrow if confidence >= ARROW_CONFIDENCE_THRESHOLD else "none"

        if self.frame_count % OBSTACLE_INTERVAL == 0:
            self.last_obstacle_in_lane = self.obstacle.detect_obstacle_in_lane(image)

        self.frame_count += 1
        angle, speed = _decide(
            lane_angle_norm=lane_angle_norm,
            arrow=self.last_arrow,
            obstacle_in_lane=self.last_obstacle_in_lane,
        )
        return {
            "decision": (angle, speed),
            "lane_angle_norm": lane_angle_norm,
            "arrow": self.last_arrow,
            "obstacle_in_lane": self.last_obstacle_in_lane,
            "frame_count": self.frame_count,
        }
'''
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build single-file autopilot model.py.")
    parser.add_argument("--lane", type=Path, default=Path("car/checkpoints/lane_best.tflite"))
    parser.add_argument("--arrow", type=Path, default=Path("car/checkpoints/arrow_best.tflite"))
    parser.add_argument("--obstacle_cpu", type=Path, default=Path("car/checkpoints/obstacle_detector.tflite"))
    parser.add_argument("--obstacle_edgetpu", type=Path, default=Path("car/checkpoints/obstacle_detector_edgetpu.tflite"))
    parser.add_argument("--output", type=Path, default=Path("deploy/picar_single_file/model.py"))
    return parser.parse_args()


def encode_model(path: Path) -> str:
    compressed = zlib.compress(path.read_bytes(), level=1)
    encoded = base64.b64encode(compressed).decode("ascii")
    return "\n".join(textwrap.wrap(encoded, width=88))


def main() -> None:
    args = parse_args()
    content = TEMPLATE.substitute(
        lane_model_b64=encode_model(args.lane),
        arrow_model_b64=encode_model(args.arrow),
        obstacle_cpu_model_b64=encode_model(args.obstacle_cpu),
        obstacle_edgetpu_model_b64=encode_model(args.obstacle_edgetpu),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
