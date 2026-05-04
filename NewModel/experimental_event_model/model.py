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
ARROW_MODEL_PATH = os.environ.get("PICAR_ARROW_MODEL", os.path.join(MODEL_DIR, "arrow_model.tflite"))
OBSTACLE_MODEL_PATH = os.environ.get("PICAR_OBSTACLE_MODEL", os.path.join(MODEL_DIR, "obstacle_detector.tflite"))

ANGLE_MIN = 50
ANGLE_MAX = 120
ANGLE_STRAIGHT = 90
ANGLE_RUNTIME_MIN = int(os.environ.get("PICAR_ANGLE_RUNTIME_MIN", "70"))
ANGLE_RUNTIME_MAX = int(os.environ.get("PICAR_ANGLE_RUNTIME_MAX", "110"))

BASE_SPEED = int(os.environ.get("PICAR_BASE_SPEED", "12"))
SLOW_SPEED = int(os.environ.get("PICAR_SLOW_SPEED", "10"))
VERY_SLOW_SPEED = int(os.environ.get("PICAR_VERY_SLOW_SPEED", "8"))

CROP_TOP_RATIO = float(os.environ.get("PICAR_CROP_TOP_RATIO", "0.35"))
MODEL_OUTPUT_MODE = os.environ.get("PICAR_MODEL_OUTPUT_MODE", "angle")
STEERING_EMA_ALPHA = float(os.environ.get("PICAR_STEERING_EMA_ALPHA", "0.35"))
MAX_STEERING_DELTA = float(os.environ.get("PICAR_MAX_STEERING_DELTA", "7"))

USE_OPENCV_SAFETY = os.environ.get("PICAR_USE_OPENCV_SAFETY", "0") == "1"
SAFETY_CORRECTION = float(os.environ.get("PICAR_SAFETY_CORRECTION", "4"))
DEBUG_EVERY = int(os.environ.get("PICAR_DEBUG_EVERY", "10"))

ENABLE_DEBUG_STREAM = os.environ.get("PICAR_DEBUG_STREAM", "0") == "1"
DEBUG_STREAM_HOST = os.environ.get("PICAR_DEBUG_HOST", "0.0.0.0")
DEBUG_STREAM_PORT = int(os.environ.get("PICAR_DEBUG_PORT", "8080"))
DEBUG_STREAM_FPS = float(os.environ.get("PICAR_DEBUG_FPS", "5"))
DEBUG_STREAM_JPEG_QUALITY = int(os.environ.get("PICAR_DEBUG_JPEG_QUALITY", "70"))

ENABLE_ARROW_CONTROL = os.environ.get("PICAR_ENABLE_ARROW_CONTROL", "0") == "1"
ENABLE_OBSTACLE_STOP = os.environ.get("PICAR_ENABLE_OBSTACLE_STOP", "0") == "1"
EVENT_INTERVAL = int(os.environ.get("PICAR_EVENT_INTERVAL", "5"))

ARROW_CLASSES = ["left", "right"]
ARROW_ROI = tuple(int(v) for v in os.environ.get("PICAR_ARROW_ROI", "0,0,320,130").split(","))
ARROW_CONFIDENCE_THRESHOLD = float(os.environ.get("PICAR_ARROW_CONFIDENCE", "0.92"))
ARROW_CONFIRM_FRAMES = int(os.environ.get("PICAR_ARROW_CONFIRM_FRAMES", "2"))
ARROW_TURN_FRAMES = int(os.environ.get("PICAR_ARROW_TURN_FRAMES", "20"))
ARROW_LEFT_ANGLE = int(os.environ.get("PICAR_ARROW_LEFT_ANGLE", "70"))
ARROW_RIGHT_ANGLE = int(os.environ.get("PICAR_ARROW_RIGHT_ANGLE", "110"))

OBSTACLE_SCORE_THRESHOLD = float(os.environ.get("PICAR_OBSTACLE_CONFIDENCE", "0.4"))
OBSTACLE_LANE_X_MIN = float(os.environ.get("PICAR_OBSTACLE_LANE_X_MIN", "0.22"))
OBSTACLE_LANE_X_MAX = float(os.environ.get("PICAR_OBSTACLE_LANE_X_MAX", "0.78"))
OBSTACLE_LANE_Y_MIN = float(os.environ.get("PICAR_OBSTACLE_LANE_Y_MIN", "0.38"))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
        return array.astype(np.float32)
    scale, zero_point = output_detail.get("quantization", (0.0, 0))
    if scale == 0:
        return array.astype(np.float32)
    return (array.astype(np.float32) - zero_point) * scale


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _to_angle(raw: float) -> float:
    if not np.isfinite(raw):
        return ANGLE_STRAIGHT
    if MODEL_OUTPUT_MODE == "normalized":
        return ANGLE_MIN + _clip(raw, 0.0, 1.0) * (ANGLE_MAX - ANGLE_MIN)
    return _clip(raw, ANGLE_MIN, ANGLE_MAX)


@dataclass
class SafetyStatus:
    active: bool
    correction: float
    reason: str
    outer_x: Optional[float]
    dashed_x: Optional[float]
    confidence: float


@dataclass
class EventStatus:
    arrow: str
    arrow_confidence: float
    arrow_control_active: bool
    pending_turn: str
    turn_frames_left: int
    obstacle_seen: bool
    obstacle_score: float
    obstacle_box: Optional[tuple[float, float, float, float]]
    obstacle_stop_active: bool


class TFLiteLanePredictor:
    def __init__(self, model_path: str) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Missing lane model: {model_path}")
        self.interpreter = _make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        shape = [int(v) for v in self.input_detail["shape"]]
        self.channels_first = len(shape) == 4 and shape[1] == 3
        if self.channels_first:
            self.height = shape[2]
            self.width = shape[3]
        else:
            self.height = shape[1]
            self.width = shape[2]

    def predict_raw(self, bgr_frame: np.ndarray) -> float:
        h = bgr_frame.shape[0]
        cropped = bgr_frame[int(h * CROP_TOP_RATIO):, :, :]
        resized = cv2.resize(cropped, (self.width, self.height), interpolation=cv2.INTER_AREA)
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


class OptionalTFLiteClassifier:
    def __init__(self, model_path: str, classes: list[str]) -> None:
        self.available = os.path.isfile(model_path)
        self.classes = classes
        if not self.available:
            self.interpreter = None
            self.input_detail = {}
            self.output_detail = {}
            self.height = 0
            self.width = 0
            self.channels_first = False
            return
        self.interpreter = _make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        shape = [int(v) for v in self.input_detail["shape"]]
        self.channels_first = len(shape) == 4 and shape[1] == 3
        if self.channels_first:
            self.height = shape[2]
            self.width = shape[3]
        else:
            self.height = shape[1]
            self.width = shape[2]

    def predict_with_confidence(self, bgr_roi: np.ndarray) -> tuple[str, float]:
        if not self.available or self.interpreter is None or bgr_roi.size == 0:
            return "none", 0.0
        resized = cv2.resize(bgr_roi, (self.width, self.height), interpolation=cv2.INTER_AREA)
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
        if idx >= len(self.classes):
            return "unknown", float(probs[idx])
        return self.classes[idx], float(probs[idx])


class OptionalObstacleDetector:
    def __init__(self, model_path: str) -> None:
        self.available = os.path.isfile(model_path)
        if not self.available:
            self.interpreter = None
            self.input_details = []
            self.output_details = []
            self.input_height = 0
            self.input_width = 0
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
            self.input_details = []
            self.output_details = []
            self.input_height = 0
            self.input_width = 0

    def detect(self, bgr_frame: np.ndarray) -> list[dict]:
        if not self.available or self.interpreter is None:
            return []
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        input_data = np.expand_dims(resized, axis=0).astype(self.input_details[0]["dtype"])
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        boxes = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]["index"])[0]
        detections = []
        for i, score in enumerate(scores):
            if float(score) < OBSTACLE_SCORE_THRESHOLD or int(classes[i]) != 0:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            detections.append({
                "box": (float(xmin), float(ymin), float(xmax), float(ymax)),
                "score": float(score),
            })
        return detections


class OpenCVSafetyMonitor:
    def check(self, bgr_frame: np.ndarray) -> SafetyStatus:
        mask = self.mask(bgr_frame)
        h, w = bgr_frame.shape[:2]
        crop_y = int(h * 0.45)
        rows = mask.shape[0]
        band = mask[int(rows * 0.45):int(rows * 0.95), :]
        _, xs = np.nonzero(band)
        if len(xs) < 80:
            return SafetyStatus(False, 0.0, "few_pixels", None, None, 0.0)

        x_norm = xs.astype(np.float32) / max(w - 1, 1)
        outer = self._median(x_norm[x_norm < 0.40])
        dashed = self._median(x_norm[(x_norm > 0.42) & (x_norm < 0.86)])
        correction = 0.0
        reasons = []
        if dashed is not None and dashed < 0.55:
            correction -= SAFETY_CORRECTION
            reasons.append("near_dashed")
        if outer is not None and outer > 0.28:
            correction += SAFETY_CORRECTION
            reasons.append("near_outer")
        active = bool(reasons)
        confidence = min(1.0, len(xs) / 1600.0)
        return SafetyStatus(active, correction, "+".join(reasons) if reasons else "ok", outer, dashed, confidence)

    @staticmethod
    def mask(bgr_frame: np.ndarray) -> np.ndarray:
        h, w = bgr_frame.shape[:2]
        crop = bgr_frame[int(h * 0.45):, :, :]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        white = cv2.inRange(hsv, np.array([0, 0, 125]), np.array([180, 95, 255]))
        dark = cv2.inRange(gray, 0, 105)
        support = cv2.dilate(white, np.ones((13, 13), np.uint8), iterations=1)
        mask = cv2.bitwise_and(dark, support)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return mask

    @staticmethod
    def _median(values: np.ndarray) -> Optional[float]:
        if values.size < 30:
            return None
        return float(np.median(values))


class DebugStream:
    _started_ports: set[int] = set()

    def __init__(self, safety_monitor: OpenCVSafetyMonitor) -> None:
        self.safety_monitor = safety_monitor
        self.enabled = ENABLE_DEBUG_STREAM
        self.latest_jpeg: Optional[bytes] = None
        self.lock = threading.Lock()
        self.last_encode_time = 0.0
        self.min_encode_interval = 1.0 / max(DEBUG_STREAM_FPS, 0.1)
        if self.enabled:
            self._start_server()

    def update(self, frame: np.ndarray, debug: dict) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self.last_encode_time < self.min_encode_interval:
            return
        self.last_encode_time = now

        image = self._draw(frame, debug)
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), DEBUG_STREAM_JPEG_QUALITY])
        if ok:
            with self.lock:
                self.latest_jpeg = encoded.tobytes()

    def _draw(self, frame: np.ndarray, debug: dict) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        safety: SafetyStatus = debug["safety"]
        event: EventStatus = debug.get(
            "event",
            EventStatus("none", 0.0, False, "none", 0, False, 0.0, None, False),
        )
        final_angle, speed = debug["decision"]
        raw = float(debug["raw_angle"])
        model_angle = float(debug["model_angle"])
        crop_y = int(h * CROP_TOP_RATIO)
        safety_y = int(h * 0.45)

        cv2.line(overlay, (0, crop_y), (w - 1, crop_y), (255, 0, 255), 1)
        cv2.line(overlay, (0, safety_y), (w - 1, safety_y), (0, 165, 255), 1)
        cv2.line(overlay, (w // 2, safety_y), (w // 2, h - 1), (255, 0, 0), 1)
        if safety.outer_x is not None:
            x = int(safety.outer_x * w)
            cv2.line(overlay, (x, safety_y), (x, h - 1), (0, 0, 255), 2)
        if safety.dashed_x is not None:
            x = int(safety.dashed_x * w)
            cv2.line(overlay, (x, safety_y), (x, h - 1), (0, 255, 255), 2)
        if event.obstacle_box is not None:
            xmin, ymin, xmax, ymax = event.obstacle_box
            p1 = (int(xmin * w), int(ymin * h))
            p2 = (int(xmax * w), int(ymax * h))
            cv2.rectangle(overlay, p1, p2, (0, 0, 255), 2)
            cv2.putText(overlay, f"obstacle {event.obstacle_score:.2f}", (p1[0], max(16, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)

        x1, y1, x2, y2 = ARROW_ROI
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 1)

        panel_h = 96
        cv2.rectangle(overlay, (0, 0), (w - 1, panel_h), (0, 0, 0), -1)
        lines = [
            f"frame={debug['frame_count']} raw={raw:.1f} model={model_angle:.1f} final={final_angle} speed={speed}",
            f"safety={safety.reason} active={safety.active} corr={safety.correction:.1f} conf={safety.confidence:.2f}",
            f"arrow={event.arrow}:{event.arrow_confidence:.2f} pending={event.pending_turn} turn={event.turn_frames_left} control={event.arrow_control_active}",
            f"obstacle={event.obstacle_seen}:{event.obstacle_score:.2f} stop={event.obstacle_stop_active}",
        ]
        for i, text in enumerate(lines):
            cv2.putText(overlay, text, (8, 19 + i * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)

        mask = self.safety_monitor.mask(frame)
        mask_full = np.zeros_like(frame)
        mask_full[safety_y:, :] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(mask_full, "OpenCV safety mask", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        return np.hstack([overlay, mask_full])

    def _start_server(self) -> None:
        if DEBUG_STREAM_PORT in DebugStream._started_ports:
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
<html><head><title>PiCar Lane Debug</title></head>
<body style="background:#111;color:#eee;font-family:Arial,sans-serif">
<h2>PiCar Lane Debug</h2>
<img src="/stream" style="max-width:100%;height:auto;border:1px solid #555">
</body></html>"""
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

        DebugStream._started_ports.add(DEBUG_STREAM_PORT)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"[model] Debug stream: http://<pi-ip>:{DEBUG_STREAM_PORT}")


class Model:
    def __init__(self) -> None:
        self.lane = TFLiteLanePredictor(LANE_MODEL_PATH)
        self.arrow = OptionalTFLiteClassifier(ARROW_MODEL_PATH, ARROW_CLASSES)
        self.obstacle = OptionalObstacleDetector(OBSTACLE_MODEL_PATH)
        self.safety_monitor = OpenCVSafetyMonitor()
        self.frame_count = 0
        self.last_angle = float(ANGLE_STRAIGHT)
        self.last_log_time = time.monotonic()
        self.last_arrow = "none"
        self.last_arrow_confidence = 0.0
        self.arrow_streak = 0
        self.pending_turn = "none"
        self.turn_frames_left = 0
        self.last_obstacle_seen = False
        self.last_obstacle_score = 0.0
        self.last_obstacle_box: Optional[tuple[float, float, float, float]] = None
        self.debug_stream = DebugStream(self.safety_monitor)
        print(
            "[model] Experimental lane+events model loaded "
            f"(lane={self.lane.width}x{self.lane.height}, arrow={self.arrow.available}, "
            f"obstacle={self.obstacle.available}, arrow_control={ENABLE_ARROW_CONTROL}, "
            f"obstacle_stop={ENABLE_OBSTACLE_STOP}, stream={self.debug_stream.enabled})."
        )

    def predict(self, image: np.ndarray) -> tuple[int, int]:
        return self.predict_debug(image)["decision"]

    def predict_debug(self, image: np.ndarray) -> dict:
        raw = self.lane.predict_raw(image)
        model_angle = _to_angle(raw)
        safe_model_angle = _clip(model_angle, ANGLE_MIN, ANGLE_MAX)

        safety = self.safety_monitor.check(image) if USE_OPENCV_SAFETY else SafetyStatus(False, 0.0, "disabled", None, None, 0.0)
        event = self._update_events(image)
        requested_angle = safe_model_angle + safety.correction

        if ENABLE_OBSTACLE_STOP and event.obstacle_stop_active:
            requested_angle = self.last_angle
        elif ENABLE_ARROW_CONTROL:
            if self.turn_frames_left > 0:
                requested_angle = ARROW_LEFT_ANGLE if self.pending_turn == "left" else ARROW_RIGHT_ANGLE
                self.turn_frames_left -= 1
                if self.turn_frames_left == 0:
                    self.pending_turn = "none"
            elif self.pending_turn in ("left", "right"):
                requested_angle = ARROW_LEFT_ANGLE if self.pending_turn == "left" else ARROW_RIGHT_ANGLE
                self.turn_frames_left = max(0, ARROW_TURN_FRAMES - 1)

        delta = _clip(requested_angle - self.last_angle, -MAX_STEERING_DELTA, MAX_STEERING_DELTA)
        limited_angle = self.last_angle + delta
        smoothed_angle = self.last_angle * (1.0 - STEERING_EMA_ALPHA) + limited_angle * STEERING_EMA_ALPHA
        final_angle = int(round(_clip(smoothed_angle, ANGLE_RUNTIME_MIN, ANGLE_RUNTIME_MAX)))
        speed = 0 if ENABLE_OBSTACLE_STOP and event.obstacle_stop_active else self._choose_speed(final_angle, safety)

        self.last_angle = float(final_angle)
        self.frame_count += 1
        debug = {
            "decision": (final_angle, speed),
            "raw_angle": raw,
            "model_angle": model_angle,
            "safety": safety,
            "event": event,
            "frame_count": self.frame_count,
        }
        self.debug_stream.update(image, debug)

        if DEBUG_EVERY > 0 and self.frame_count % DEBUG_EVERY == 0:
            print(
                f"[model] raw={raw:.2f} converted={model_angle:.1f} final={final_angle} speed={speed} "
                f"safety={safety.reason} corr={safety.correction:.1f} outer={safety.outer_x} dashed={safety.dashed_x} "
                f"arrow={event.arrow}:{event.arrow_confidence:.2f} pending={event.pending_turn} "
                f"obstacle={event.obstacle_seen}:{event.obstacle_score:.2f}"
            )

        return debug

    def _update_events(self, image: np.ndarray) -> EventStatus:
        if EVENT_INTERVAL > 0 and self.frame_count % EVENT_INTERVAL == 0:
            self._update_arrow(image)
            self._update_obstacle(image)
        return EventStatus(
            arrow=self.last_arrow,
            arrow_confidence=self.last_arrow_confidence,
            arrow_control_active=ENABLE_ARROW_CONTROL and (self.pending_turn in ("left", "right") or self.turn_frames_left > 0),
            pending_turn=self.pending_turn,
            turn_frames_left=self.turn_frames_left,
            obstacle_seen=self.last_obstacle_seen,
            obstacle_score=self.last_obstacle_score,
            obstacle_box=self.last_obstacle_box,
            obstacle_stop_active=ENABLE_OBSTACLE_STOP and self.last_obstacle_seen,
        )

    def _update_arrow(self, image: np.ndarray) -> None:
        x1, y1, x2, y2 = ARROW_ROI
        h, w = image.shape[:2]
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        arrow, confidence = self.arrow.predict_with_confidence(image[y1:y2, x1:x2])
        self.last_arrow_confidence = confidence
        if confidence >= ARROW_CONFIDENCE_THRESHOLD and arrow in ("left", "right"):
            self.arrow_streak = self.arrow_streak + 1 if arrow == self.last_arrow else 1
            self.last_arrow = arrow
        else:
            self.arrow_streak = 0
            self.last_arrow = "none"
        if self.arrow_streak >= ARROW_CONFIRM_FRAMES:
            self.pending_turn = self.last_arrow

    def _update_obstacle(self, image: np.ndarray) -> None:
        detections = self.obstacle.detect(image)
        best_score = 0.0
        best_box = None
        seen = False
        for det in detections:
            box = det["box"]
            score = float(det["score"])
            xmin, ymin, xmax, ymax = box
            overlaps_lane = xmin < OBSTACLE_LANE_X_MAX and xmax > OBSTACLE_LANE_X_MIN and ymax > OBSTACLE_LANE_Y_MIN
            if score > best_score:
                best_score = score
                best_box = box
            if overlaps_lane:
                seen = True
        self.last_obstacle_seen = seen
        self.last_obstacle_score = best_score
        self.last_obstacle_box = best_box

    @staticmethod
    def _choose_speed(angle: int, safety: SafetyStatus) -> int:
        demand = abs(float(angle) - ANGLE_STRAIGHT)
        if demand > 22:
            speed = VERY_SLOW_SPEED
        elif demand > 14:
            speed = SLOW_SPEED
        else:
            speed = BASE_SPEED
        if safety.active:
            speed = min(speed, SLOW_SPEED)
        return int(speed)
