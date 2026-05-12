"""
PiCar lane-following + arrow-turning model.

This is designed for the adammoss/autopilot skeleton. Put this whole folder under:
    /home/pi/autopilot/autopilot/models/<your_model_name>/

Required files in the same folder:
    model.py
    lane_model.tflite
    arrow_model.tflite

No obstacle detector and no Coral TPU code is used here.
"""
from __future__ import annotations

import os

import cv2
import numpy as np


MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
LANE_MODEL_PATH = os.path.join(MODEL_DIR, "lane_model.tflite")
ARROW_MODEL_PATH = os.path.join(MODEL_DIR, "arrow_model.tflite")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# The lane model was trained with targets normalised from car units [50, 120].
ANGLE_MIN = 50
ANGLE_MAX = 120
ANGLE_STRAIGHT = 90
CRUISE_SPEED = 35
INVERT_LANE_STEERING = True
STEERING_GAIN = 2.0
ANGLE_RUNTIME_MIN = 70
ANGLE_RUNTIME_MAX = 110

CROP_RATIO = 0.25
ARROW_CLASSES = ["left", "right"]
ARROW_CONFIDENCE_THRESHOLD = 0.98
ARROW_ROI = (0, 0, 320, 120)
ARROW_INTERVAL = 5
ARROW_STEERING_BIAS = 8
USE_ARROW_TURNS = False
DEBUG_EVERY = 10


def _make_interpreter(model_path: str):
    try:
        import tensorflow as tf

        return tf.lite.Interpreter(model_path=model_path)
    except ImportError:
        import tflite_runtime.interpreter as tflite

        return tflite.Interpreter(model_path=model_path)


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
    if not np.isfinite(angle_norm) or angle_norm < -0.25 or angle_norm > 1.25:
        return ANGLE_STRAIGHT
    if INVERT_LANE_STEERING:
        angle_norm = 1.0 - angle_norm
    clipped = max(0.0, min(1.0, float(angle_norm)))
    angle = ANGLE_MIN + clipped * (ANGLE_MAX - ANGLE_MIN)
    angle = ANGLE_STRAIGHT + (angle - ANGLE_STRAIGHT) * STEERING_GAIN
    angle = max(ANGLE_RUNTIME_MIN, min(ANGLE_RUNTIME_MAX, angle))
    return int(round(angle))


def _decide(lane_angle_norm: float, arrow: str) -> tuple[int, int]:
    angle = _normalised_to_car_angle(lane_angle_norm)
    if arrow == "left":
        angle = max(ANGLE_MIN, angle - ARROW_STEERING_BIAS)
    elif arrow == "right":
        angle = min(ANGLE_MAX, angle + ARROW_STEERING_BIAS)
    return angle, CRUISE_SPEED


class TFLiteLanePredictor:
    def __init__(self, model_path: str) -> None:
        self.interpreter = _make_interpreter(model_path)
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
        crop_start = int(bgr_frame.shape[0] * CROP_RATIO)
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
        return float(output.reshape(-1)[0])


class TFLiteArrowPredictor:
    def __init__(self, model_path: str) -> None:
        self.interpreter = _make_interpreter(model_path)
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


class Model:
    """Autopilot-compatible model with predict(image) -> (angle, speed)."""

    def __init__(self):
        self.lane = TFLiteLanePredictor(LANE_MODEL_PATH)
        self.arrow = TFLiteArrowPredictor(ARROW_MODEL_PATH)
        self.frame_count = 0
        self.last_arrow = "none"
        self.last_arrow_confidence = 0.0
        print("[model] Lane + arrow model loaded; obstacle detection disabled.")

    @staticmethod
    def _extract_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = roi
        return image[y1:y2, x1:x2].copy()

    def predict(self, image: np.ndarray) -> tuple[int, int]:
        return self.predict_debug(image)["decision"]

    def predict_debug(self, image: np.ndarray) -> dict:
        lane_angle_norm = self.lane.predict(image)

        if USE_ARROW_TURNS and self.frame_count % ARROW_INTERVAL == 0:
            arrow_roi = self._extract_roi(image, ARROW_ROI)
            arrow, confidence = self.arrow.predict_with_confidence(arrow_roi)
            self.last_arrow_confidence = confidence
            self.last_arrow = arrow if confidence >= ARROW_CONFIDENCE_THRESHOLD else "none"
        elif not USE_ARROW_TURNS:
            self.last_arrow = "none"
            self.last_arrow_confidence = 0.0

        self.frame_count += 1
        angle, speed = _decide(lane_angle_norm, self.last_arrow)
        if self.frame_count % DEBUG_EVERY == 0:
            print(
                f"[model] lane_raw={lane_angle_norm:.3f} "
                f"angle={angle} arrow={self.last_arrow} "
                f"arrow_conf={self.last_arrow_confidence:.3f} speed={speed}"
            )
        return {
            "decision": (angle, speed),
            "lane_angle_norm": lane_angle_norm,
            "arrow": self.last_arrow,
            "arrow_confidence": self.last_arrow_confidence,
            "frame_count": self.frame_count,
        }
