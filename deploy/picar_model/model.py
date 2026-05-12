"""
Autopilot integration model.

Copy or symlink this file into autopilot/models/<your_group>/model.py.
Adjust the PATHS at the top to point to your trained checkpoints.

The autopilot framework expects a Model class with a predict(image) method
that receives a BGR 320x240 image and returns (angle, speed) in car units.
"""
from __future__ import annotations

import os
import sys

import numpy as np

def _find_project_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        here,
        os.path.abspath(os.path.join(here, "..")),
        os.path.abspath(os.path.join(here, "..", "..")),
        os.getcwd(),
    ]
    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "car", "checkpoints")):
            return candidate
    return os.path.abspath(os.path.join(here, "..", ".."))


PROJECT_ROOT = _find_project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from car.inference.controller import decide
from car.inference.lane_model import LanePredictor

LANE_CHECKPOINT = os.path.join(PROJECT_ROOT, "car", "checkpoints", "lane_best.pt")
LANE_TFLITE = os.path.join(PROJECT_ROOT, "car", "checkpoints", "lane_best.tflite")
ARROW_CHECKPOINT = os.path.join(PROJECT_ROOT, "car", "checkpoints", "arrow_best.pt")
ARROW_TFLITE = os.path.join(PROJECT_ROOT, "car", "checkpoints", "arrow_best.tflite")
OBSTACLE_MODEL = os.path.join(PROJECT_ROOT, "car", "checkpoints", "obstacle_detector.tflite")
OBSTACLE_EDGETPU_MODEL = os.path.join(PROJECT_ROOT, "car", "checkpoints", "obstacle_detector_edgetpu.tflite")

ARROW_CLASSES = ["left", "right"]
ARROW_CONFIDENCE_THRESHOLD = 0.85

EVENT_INTERVAL = 5
OBSTACLE_INTERVAL = 5

CRUISE_SPEED = 35

ARROW_ROI = (0, 0, 320, 120)


class Model:
    """Autopilot-compatible model combining lane following, event detection,
    and obstacle detection with a priority-based controller."""

    def __init__(self):
        self.lane = self._load_lane_model()

        self.arrow = None
        self.arrow = self._load_arrow_model()

        self.obstacle = None
        use_edgetpu = os.path.exists("/dev/bus/usb") and os.path.isfile(OBSTACLE_EDGETPU_MODEL)
        obstacle_model = OBSTACLE_EDGETPU_MODEL if use_edgetpu else OBSTACLE_MODEL
        if os.path.isfile(obstacle_model):
            try:
                from car.inference.obstacle_detector import ObstacleDetector

                self.obstacle = ObstacleDetector(
                    model_path=obstacle_model,
                    score_threshold=0.4,
                    use_edgetpu=use_edgetpu,
                )
                print(f"[autopilot] Obstacle detector loaded (edgetpu={use_edgetpu}).")
            except Exception as e:
                print(f"[autopilot] Obstacle detector not available: {e}")

        self.frame_count = 0
        self.last_arrow = "none"
        self.last_obstacle_in_lane = False

    def _load_lane_model(self):
        if os.path.isfile(LANE_TFLITE):
            try:
                from car.inference.lane_model import TFLiteLanePredictor

                model = TFLiteLanePredictor(LANE_TFLITE)
                print("[autopilot] Lane TFLite model loaded.")
                return model
            except Exception as e:
                print(f"[autopilot] Lane TFLite unavailable, falling back to PyTorch: {e}")

        model = LanePredictor(checkpoint_path=LANE_CHECKPOINT, device="cpu")
        print("[autopilot] Lane PyTorch model loaded.")
        return model

    def _load_arrow_model(self):
        if os.path.isfile(ARROW_TFLITE):
            try:
                from car.inference.event_models import TFLiteEventPredictor

                model = TFLiteEventPredictor(ARROW_TFLITE, classes=ARROW_CLASSES)
                print("[autopilot] Arrow TFLite classifier loaded.")
                return model
            except Exception as e:
                print(f"[autopilot] Arrow TFLite unavailable, falling back to PyTorch: {e}")

        if os.path.isfile(ARROW_CHECKPOINT):
            from car.inference.event_models import EventPredictor

            model = EventPredictor(
                checkpoint_path=ARROW_CHECKPOINT,
                classes=ARROW_CLASSES,
                input_size=96,
                device="cpu",
            )
            print("[autopilot] Arrow PyTorch classifier loaded.")
            return model
        return None

    def _extract_roi(self, image: np.ndarray, roi: tuple) -> np.ndarray:
        x1, y1, x2, y2 = roi
        return image[y1:y2, x1:x2].copy()

    def predict(self, image: np.ndarray) -> tuple:
        """Takes a BGR 320x240 frame, returns (angle, speed) in car units."""
        return self.predict_debug(image)["decision"]

    def predict_debug(self, image: np.ndarray) -> dict:
        """Return decision plus intermediate model state for debugging UIs."""
        lane_angle_norm = self.lane.predict(image)

        if self.frame_count % EVENT_INTERVAL == 0:
            if self.arrow is not None:
                arrow_roi = self._extract_roi(image, ARROW_ROI)
                arrow, confidence = self.arrow.predict_with_confidence(arrow_roi)
                self.last_arrow = arrow if confidence >= ARROW_CONFIDENCE_THRESHOLD else "none"

        if self.frame_count % OBSTACLE_INTERVAL == 0:
            if self.obstacle is not None:
                self.last_obstacle_in_lane = self.obstacle.detect_obstacle_in_lane(image)

        self.frame_count += 1

        angle, speed = decide(
            lane_angle_norm=lane_angle_norm,
            arrow=self.last_arrow,
            obstacle_in_lane=self.last_obstacle_in_lane,
            cruise_speed=CRUISE_SPEED,
        )

        return {
            "decision": (angle, speed),
            "lane_angle_norm": lane_angle_norm,
            "arrow": self.last_arrow,
            "obstacle_in_lane": self.last_obstacle_in_lane,
            "frame_count": self.frame_count,
        }
