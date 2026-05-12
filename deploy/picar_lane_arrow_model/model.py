"""
PiCar hybrid autonomy model for the adammoss/autopilot skeleton.

Folder layout expected on the Pi:
    model.py
    lane_model.tflite
    arrow_model.tflite
    car/checkpoints/obstacle_detector.tflite       optional

The primary lane keeper uses OpenCV line geometry. The trained lane model is kept
as a fallback when geometry is uncertain. Arrow and obstacle models run as slower
event detectors and feed a small stateful controller.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np


MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
LANE_MODEL_PATH = os.path.join(MODEL_DIR, "lane_model.tflite")
ARROW_MODEL_PATH = os.path.join(MODEL_DIR, "arrow_model.tflite")
OBSTACLE_MODEL_PATH = os.path.join(MODEL_DIR, "car", "checkpoints", "obstacle_detector.tflite")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ANGLE_MIN = 50
ANGLE_MAX = 120
ANGLE_STRAIGHT = 90
CRUISE_SPEED = 30
SLOW_SPEED = 22

GEOMETRY_STEERING_SIGN = 1.0
GEOMETRY_STEERING_GAIN = 150.0
FALLBACK_STEERING_GAIN = 120.0
FALLBACK_CENTER_RAW = 0.42
INVERT_FALLBACK_STEERING = True
ANGLE_RUNTIME_MIN = 65
ANGLE_RUNTIME_MAX = 115
ANGLE_SMOOTHING = 0.80

LANE_CROP_TOP_RATIO = 0.40
LANE_MIN_PIXELS = 80
LANE_DEFAULT_WIDTH_FRAC = 0.34
LANE_LOOKAHEAD_WEIGHT = 0.30
LANE_TARGET_POSITION = 0.62
LANE_TARGET_SWING_GAIN = 1.60
LANE_TARGET_SWING_LIMIT = 0.24
LANE_RETURN_MEMORY = 0.70
LANE_WIDTH_UPDATE_RATE = 0.05
LANE_TARGET_MAX_STEP = 0.30
LANE_REJECT_BORDER_MARGIN = 3
LANE_REJECT_MAX_COMPONENT_FRAC = 0.08

ARROW_CLASSES = ["left", "right"]
ARROW_ROI = (0, 0, 320, 120)
ARROW_INTERVAL = 5
ARROW_CONFIDENCE_THRESHOLD = 0.98
ARROW_CONFIRM_FRAMES = 2
TURN_FRAMES = 22
TURN_ANGLE_LEFT = 65
TURN_ANGLE_RIGHT = 115
USE_ARROW_TURNS = False

OBSTACLE_INTERVAL = 5
OBSTACLE_SCORE_THRESHOLD = 0.4
OBSTACLE_LANE_Y_MIN = 0.38
USE_OBSTACLE_DETECTION = False

DEBUG_EVERY = 10

ENABLE_DEBUG_STREAM = os.environ.get("PICAR_DEBUG_STREAM", "1") != "0"
DEBUG_STREAM_HOST = "0.0.0.0"
DEBUG_STREAM_PORT = int(os.environ.get("PICAR_DEBUG_PORT", "8080"))
DEBUG_STREAM_FPS = 5
DEBUG_STREAM_JPEG_QUALITY = 70


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


def _clip_angle(angle: float) -> int:
    return int(round(max(ANGLE_RUNTIME_MIN, min(ANGLE_RUNTIME_MAX, angle))))


@dataclass
class LaneGeometry:
    ok: bool
    angle: int
    confidence: float
    lane_x_min: float
    lane_x_max: float
    target_x: float
    left_x: Optional[float]
    centre_x: Optional[float]
    reason: str


class LaneGeometryDetector:
    """Detect the left-lane corridor from black track markings."""

    def __init__(self) -> None:
        self.last_width_frac = LANE_DEFAULT_WIDTH_FRAC
        self.last_target_x = 0.5

    def estimate(self, bgr_frame: np.ndarray) -> LaneGeometry:
        height, width = bgr_frame.shape[:2]
        crop_y = int(height * LANE_CROP_TOP_RATIO)
        crop = bgr_frame[crop_y:, :]
        mask = self._line_mask(crop)
        rows = mask.shape[0]
        near = self._estimate_band(mask, width, int(rows * 0.45), int(rows * 0.90))
        far = self._estimate_band(mask, width, int(rows * 0.05), int(rows * 0.55))
        if near is None and far is None:
            return self._fallback_geometry("few_pixels")

        if near is None:
            target_x, lane_width, left_x, centre_x, pixel_count = far
            confidence = 0.55
            reason = "far_only"
        elif far is None:
            target_x, lane_width, left_x, centre_x, pixel_count = near
            confidence = 0.60
            reason = "near_only"
        else:
            near_target, near_width, near_left, near_centre, near_pixels = near
            far_target, far_width, far_left, far_centre, far_pixels = far
            base_target = float((1.0 - LANE_LOOKAHEAD_WEIGHT) * near_target + LANE_LOOKAHEAD_WEIGHT * far_target)
            swing = float(max(-LANE_TARGET_SWING_LIMIT, min(LANE_TARGET_SWING_LIMIT, (base_target - 0.5) * LANE_TARGET_SWING_GAIN)))
            target_x = 0.5 + swing
            lane_width = float((near_width + far_width) / 2.0)
            left_x = near_left if near_left is not None else far_left
            centre_x = near_centre if near_centre is not None else far_centre
            pixel_count = near_pixels + far_pixels
            confidence = min(1.0, pixel_count / 2400.0)
            reason = "lookahead"

        if left_x is None and centre_x is None:
            return self._fallback_geometry("no_sides")

        measured_width = float(max(0.22, min(0.55, lane_width)))
        if left_x is not None and centre_x is not None and 0.24 <= (centre_x - left_x) <= 0.52:
            lane_width = float(
                self.last_width_frac * (1.0 - LANE_WIDTH_UPDATE_RATE)
                + measured_width * LANE_WIDTH_UPDATE_RATE
            )
        else:
            lane_width = self.last_width_frac
        target_x = float(max(0.15, min(0.85, target_x)))
        if reason in ("near_only", "far_only"):
            target_x = self._apply_target_swing(target_x)
        target_x = self._apply_target_memory(target_x)
        target_x = self._limit_target_step(target_x)
        lane_x_min = float(max(0.0, target_x - lane_width / 2.0))
        lane_x_max = float(min(1.0, target_x + lane_width / 2.0))
        self.last_width_frac = lane_width
        self.last_target_x = target_x

        error = target_x - 0.5
        angle = _clip_angle(ANGLE_STRAIGHT + GEOMETRY_STEERING_SIGN * error * GEOMETRY_STEERING_GAIN)
        return LaneGeometry(True, angle, confidence, lane_x_min, lane_x_max, target_x, left_x, centre_x, reason)

    def _estimate_band(
        self,
        mask: np.ndarray,
        frame_width: int,
        y_start: int,
        y_end: int,
    ) -> Optional[tuple[float, float, Optional[float], Optional[float], int]]:
        band = mask[max(0, y_start):max(y_start + 1, y_end), :]
        _, xs = np.nonzero(band)
        if len(xs) < LANE_MIN_PIXELS:
            return None
        x_norm = xs.astype(np.float32) / max(frame_width - 1, 1)
        left_x = self._median_x(x_norm[x_norm < 0.44])
        centre_candidates = x_norm[(x_norm > 0.32) & (x_norm < 0.82)]
        centre_x = self._median_x(centre_candidates)
        stable_width = self.last_width_frac
        if left_x is not None and centre_x is not None and centre_x > left_x + 0.12:
            lane_width = float(centre_x - left_x)
            left_target = float(left_x + stable_width * LANE_TARGET_POSITION)
            centre_target = float(centre_x - stable_width * (1.0 - LANE_TARGET_POSITION))
            if abs(left_target - centre_target) < 0.16:
                target_x = float((left_target + centre_target) / 2.0)
            else:
                target_x = left_target if abs(left_target - self.last_target_x) < abs(centre_target - self.last_target_x) else centre_target
        elif left_x is not None:
            lane_width = stable_width
            target_x = float(left_x + stable_width * LANE_TARGET_POSITION)
        elif centre_x is not None:
            lane_width = stable_width
            target_x = float(centre_x - stable_width * (1.0 - LANE_TARGET_POSITION))
        else:
            return None
        return target_x, lane_width, left_x, centre_x, len(xs)

    def _apply_target_memory(self, target_x: float) -> float:
        previous_error = self.last_target_x - 0.5
        new_error = target_x - 0.5
        same_turn = previous_error * new_error > 0
        returning_to_straight = abs(new_error) < abs(previous_error)
        if same_turn and returning_to_straight:
            return float(self.last_target_x * LANE_RETURN_MEMORY + target_x * (1.0 - LANE_RETURN_MEMORY))
        return target_x

    @staticmethod
    def _apply_target_swing(target_x: float) -> float:
        swing = max(
            -LANE_TARGET_SWING_LIMIT,
            min(LANE_TARGET_SWING_LIMIT, (target_x - 0.5) * LANE_TARGET_SWING_GAIN),
        )
        return float(0.5 + swing)

    def _limit_target_step(self, target_x: float) -> float:
        delta = target_x - self.last_target_x
        if abs(delta) <= LANE_TARGET_MAX_STEP:
            return target_x
        return float(self.last_target_x + np.sign(delta) * LANE_TARGET_MAX_STEP)

    @staticmethod
    def _line_mask(bgr_crop: np.ndarray) -> np.ndarray:
        """Segment black solid/dashed lane markings on the white paper track."""
        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)

        # Dark carpet/background can be the same colour as the track lines.
        # Only keep dark pixels that have nearby white paper support.
        white_paper = cv2.inRange(hsv, np.array([0, 0, 125]), np.array([180, 95, 255]))
        white_gray = cv2.inRange(gray, 135, 255)
        white_support = cv2.bitwise_and(white_paper, white_gray)
        support_kernel = np.ones((11, 11), np.uint8)
        white_support = cv2.dilate(white_support, support_kernel, iterations=1)

        hsv_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 130, 115]))
        gray_mask = cv2.inRange(gray, 0, 105)
        mask = cv2.bitwise_or(hsv_mask, gray_mask)
        mask = cv2.bitwise_and(mask, white_support)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return LaneGeometryDetector._reject_fabric_edges(mask)

    @staticmethod
    def _reject_fabric_edges(mask: np.ndarray) -> np.ndarray:
        height, width = mask.shape[:2]
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered = np.zeros_like(mask)
        max_area = int(width * height * LANE_REJECT_MAX_COMPONENT_FRAC)
        margin = LANE_REJECT_BORDER_MARGIN

        for label in range(1, labels_count):
            x, y, w, h, area = stats[label]
            touches_side_or_top = x <= margin or y <= margin or (x + w) >= width - margin
            too_large = area > max_area
            large_border_edge = touches_side_or_top and area > max(350, max_area // 4)
            wide_sheet_edge = w > width * 0.80 and h < height * 0.16
            if large_border_edge or too_large or wide_sheet_edge:
                continue
            filtered[labels == label] = 255

        return filtered

    @staticmethod
    def _median_x(values: np.ndarray) -> Optional[float]:
        if values.size < LANE_MIN_PIXELS // 2:
            return None
        return float(np.median(values))

    def _fallback_geometry(self, reason: str) -> LaneGeometry:
        lane_width = self.last_width_frac
        target_x = self.last_target_x
        return LaneGeometry(
            False,
            ANGLE_STRAIGHT,
            0.0,
            max(0.0, target_x - lane_width / 2.0),
            min(1.0, target_x + lane_width / 2.0),
            target_x,
            None,
            None,
            reason,
        )


class TFLiteLanePredictor:
    def __init__(self, model_path: str) -> None:
        self.available = os.path.isfile(model_path)
        if not self.available:
            self.interpreter = None
            return
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
        if not self.available or self.interpreter is None:
            return np.nan
        crop_start = int(bgr_frame.shape[0] * 0.25)
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
        self.available = os.path.isfile(model_path)
        if not self.available:
            self.interpreter = None
            return
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
        if not self.available or self.interpreter is None:
            return "none", 0.0
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


class ObstacleDetector:
    def __init__(self, model_path: str, score_threshold: float = OBSTACLE_SCORE_THRESHOLD) -> None:
        self.available = os.path.isfile(model_path)
        self.score_threshold = score_threshold
        if not self.available:
            self.interpreter = None
            return
        try:
            self.interpreter = _make_interpreter(model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            shape = self.input_details[0]["shape"]
            self.input_height = int(shape[1])
            self.input_width = int(shape[2])
        except Exception as exc:
            print(f"[model] Obstacle detector unavailable: {exc}")
            self.available = False
            self.interpreter = None

    def detect(self, bgr_frame: np.ndarray) -> list[dict]:
        if not self.available or self.interpreter is None:
            return []
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_width, self.input_height))
        input_data = np.expand_dims(resized, axis=0).astype(self.input_details[0]["dtype"])
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]
        detections = []
        for i, score in enumerate(scores):
            if float(score) < self.score_threshold or int(classes[i]) != 0:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            detections.append({"box": (float(xmin), float(ymin), float(xmax), float(ymax)), "score": float(score)})
        return detections


def _fallback_lane_angle(lane_raw: float) -> int:
    if not np.isfinite(lane_raw):
        return ANGLE_STRAIGHT
    angle = ANGLE_STRAIGHT + (float(lane_raw) - FALLBACK_CENTER_RAW) * FALLBACK_STEERING_GAIN
    if INVERT_FALLBACK_STEERING:
        angle = ANGLE_STRAIGHT - (float(lane_raw) - FALLBACK_CENTER_RAW) * FALLBACK_STEERING_GAIN
    return _clip_angle(angle)


def _box_overlaps_lane(box: tuple[float, float, float, float], lane: LaneGeometry) -> bool:
    xmin, ymin, xmax, ymax = box
    if ymax < OBSTACLE_LANE_Y_MIN:
        return False
    return bool(xmin < lane.lane_x_max and xmax > lane.lane_x_min)


class LiveDebugStream:
    """MJPEG web stream fed by the exact frames received by predict()."""

    _started_ports: set[int] = set()

    def __init__(self, detector: LaneGeometryDetector) -> None:
        self.detector = detector
        self.latest_jpeg: Optional[bytes] = None
        self.lock = threading.Lock()
        self.last_encode_time = 0.0
        self.min_encode_interval = 1.0 / max(DEBUG_STREAM_FPS, 1)
        self.enabled = ENABLE_DEBUG_STREAM
        if self.enabled:
            self._start_server()

    def update(
        self,
        frame: np.ndarray,
        lane: LaneGeometry,
        angle: int,
        speed: int,
        lane_source: str,
        controller_state: str,
    ) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self.last_encode_time < self.min_encode_interval:
            return
        self.last_encode_time = now

        debug = self._draw_debug(frame, lane, angle, speed, lane_source, controller_state)
        ok, encoded = cv2.imencode(".jpg", debug, [int(cv2.IMWRITE_JPEG_QUALITY), DEBUG_STREAM_JPEG_QUALITY])
        if ok:
            with self.lock:
                self.latest_jpeg = encoded.tobytes()

    def _draw_debug(
        self,
        frame: np.ndarray,
        lane: LaneGeometry,
        angle: int,
        speed: int,
        lane_source: str,
        controller_state: str,
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        crop_y = int(height * LANE_CROP_TOP_RATIO)
        overlay = frame.copy()

        cv2.line(overlay, (0, crop_y), (width - 1, crop_y), (255, 0, 255), 1)
        x_min = int(lane.lane_x_min * width)
        x_max = int(lane.lane_x_max * width)
        target_x = int(lane.target_x * width)
        cv2.line(overlay, (x_min, crop_y), (x_min, height - 1), (0, 255, 255), 2)
        cv2.line(overlay, (x_max, crop_y), (x_max, height - 1), (0, 255, 255), 2)
        cv2.line(overlay, (target_x, crop_y), (target_x, height - 1), (0, 255, 0), 2)
        cv2.line(overlay, (width // 2, crop_y), (width // 2, height - 1), (255, 0, 0), 1)

        if lane.left_x is not None:
            left_x = int(lane.left_x * width)
            cv2.line(overlay, (left_x, crop_y), (left_x, height - 1), (0, 0, 255), 1)
        if lane.centre_x is not None:
            centre_x = int(lane.centre_x * width)
            cv2.line(overlay, (centre_x, crop_y), (centre_x, height - 1), (0, 165, 255), 1)

        cv2.rectangle(overlay, (0, 0), (width - 1, 58), (0, 0, 0), -1)
        cv2.putText(
            overlay,
            f"state={controller_state} angle={angle} speed={speed} target={lane.target_x:.2f}",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            overlay,
            f"source={lane_source} ok={lane.ok} reason={lane.reason}",
            (8, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            overlay,
            "yellow=corridor green=target blue=centre red=left orange=dashed",
            (8, 53),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (255, 255, 255),
            1,
        )

        crop = frame[crop_y:, :]
        mask = self.detector._line_mask(crop)
        mask_full = np.zeros_like(frame)
        mask_full[crop_y:, :] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(mask_full, "OpenCV black-line mask", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        return np.hstack([overlay, mask_full])

    def _start_server(self) -> None:
        if DEBUG_STREAM_PORT in LiveDebugStream._started_ports:
            self.enabled = False
            return

        stream = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(
                        b"""<!doctype html>
<html>
<head><title>Pi Car Live Debug</title></head>
<body style="background:#111;color:#eee;font-family:Arial,sans-serif">
<h2>Pi Car Live Debug</h2>
<p>Left: live driving frame with overlay. Right: OpenCV black-line mask.</p>
<img src="/stream" style="max-width:100%;height:auto;border:1px solid #555">
</body>
</html>"""
                    )
                    return
                if self.path != "/stream":
                    self.send_error(404)
                    return

                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()

                while True:
                    with stream.lock:
                        jpeg = stream.latest_jpeg
                    if jpeg is None:
                        time.sleep(0.05)
                        continue
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        time.sleep(0.05)
                    except (BrokenPipeError, ConnectionResetError):
                        break

            def log_message(self, format: str, *args: object) -> None:
                return

        try:
            server = ThreadingHTTPServer((DEBUG_STREAM_HOST, DEBUG_STREAM_PORT), Handler)
        except OSError as exc:
            print(f"[model] Debug stream disabled: {exc}")
            self.enabled = False
            return

        LiveDebugStream._started_ports.add(DEBUG_STREAM_PORT)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"[model] Live debug stream: http://192.168.50.1:{DEBUG_STREAM_PORT}")


class Model:
    """Autopilot-compatible model with predict(image) -> (angle, speed)."""

    def __init__(self):
        self.geometry = LaneGeometryDetector()
        self.lane_fallback = TFLiteLanePredictor(LANE_MODEL_PATH)
        self.arrow = TFLiteArrowPredictor(ARROW_MODEL_PATH)
        self.obstacle = ObstacleDetector(OBSTACLE_MODEL_PATH)
        self.frame_count = 0
        self.last_angle = ANGLE_STRAIGHT
        self.last_arrow = "none"
        self.last_arrow_confidence = 0.0
        self.arrow_streak = 0
        self.pending_turn = "none"
        self.turn_frames_left = 0
        self.last_obstacle_in_lane = False
        self.debug_stream = LiveDebugStream(self.geometry)
        print(
            "[model] Hybrid lane geometry loaded "
            f"(arrow={self.arrow.available}, obstacle={self.obstacle.available}, fallback={self.lane_fallback.available})."
        )

    @staticmethod
    def _extract_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = roi
        return image[y1:y2, x1:x2].copy()

    def predict(self, image: np.ndarray) -> tuple[int, int]:
        return self.predict_debug(image)["decision"]

    def predict_debug(self, image: np.ndarray) -> dict:
        lane = self.geometry.estimate(image)
        lane_raw = np.nan
        if lane.ok:
            lane_angle = lane.angle
            lane_source = f"geometry:{lane.reason}"
            speed = CRUISE_SPEED
        else:
            lane_raw = self.lane_fallback.predict(image)
            lane_angle = _fallback_lane_angle(lane_raw)
            lane_source = f"fallback:{lane.reason}"
            speed = SLOW_SPEED

        if USE_ARROW_TURNS and self.frame_count % ARROW_INTERVAL == 0:
            arrow_roi = self._extract_roi(image, ARROW_ROI)
            arrow, confidence = self.arrow.predict_with_confidence(arrow_roi)
            self.last_arrow_confidence = confidence
            if confidence >= ARROW_CONFIDENCE_THRESHOLD:
                self.arrow_streak = self.arrow_streak + 1 if arrow == self.last_arrow else 1
                self.last_arrow = arrow
            else:
                self.arrow_streak = 0
                self.last_arrow = "none"
            if self.arrow_streak >= ARROW_CONFIRM_FRAMES:
                self.pending_turn = self.last_arrow

        if USE_OBSTACLE_DETECTION and self.frame_count % OBSTACLE_INTERVAL == 0:
            detections = self.obstacle.detect(image)
            self.last_obstacle_in_lane = any(_box_overlaps_lane(det["box"], lane) for det in detections)

        if self.last_obstacle_in_lane:
            angle, speed = self.last_angle, 0
            controller_state = "obstacle_stop"
        elif self.turn_frames_left > 0:
            angle = TURN_ANGLE_LEFT if self.pending_turn == "left" else TURN_ANGLE_RIGHT
            self.turn_frames_left -= 1
            if self.turn_frames_left == 0:
                self.pending_turn = "none"
            controller_state = "turning"
        elif self.pending_turn in ("left", "right") and not lane.ok:
            self.turn_frames_left = TURN_FRAMES
            angle = TURN_ANGLE_LEFT if self.pending_turn == "left" else TURN_ANGLE_RIGHT
            controller_state = "start_turn"
        else:
            angle = lane_angle
            controller_state = "lane_keep"

        angle = int(round(self.last_angle * (1.0 - ANGLE_SMOOTHING) + angle * ANGLE_SMOOTHING))
        angle = _clip_angle(angle)
        self.last_angle = angle
        self.frame_count += 1
        self.debug_stream.update(image, lane, angle, speed, lane_source, controller_state)

        if self.frame_count % DEBUG_EVERY == 0:
            print(
                f"[model] state={controller_state} source={lane_source} angle={angle} speed={speed} "
                f"lane_target={lane.target_x:.2f} corridor=({lane.lane_x_min:.2f},{lane.lane_x_max:.2f}) "
                f"raw={lane_raw:.3f} arrow={self.last_arrow} arrow_conf={self.last_arrow_confidence:.3f} "
                f"obstacle={self.last_obstacle_in_lane}"
            )

        return {
            "decision": (angle, speed),
            "lane_source": lane_source,
            "lane_raw": lane_raw,
            "lane_target": lane.target_x,
            "lane_corridor": (lane.lane_x_min, lane.lane_x_max),
            "arrow": self.last_arrow,
            "arrow_confidence": self.last_arrow_confidence,
            "pending_turn": self.pending_turn,
            "obstacle_in_lane": self.last_obstacle_in_lane,
            "frame_count": self.frame_count,
        }
