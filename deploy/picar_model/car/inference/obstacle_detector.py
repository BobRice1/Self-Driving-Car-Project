"""Obstacle/pedestrian detector using TFLite (optionally on Edge TPU)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

COCO_PERSON_CLASS = 0
LANE_REGION_X_MIN = 0.25
LANE_REGION_X_MAX = 0.75
LANE_REGION_Y_MIN = 0.4


class ObstacleDetector:
    """TFLite-based object detector with lane overlap post-processing."""

    def __init__(
        self,
        model_path: str | Path,
        score_threshold: float = 0.4,
        target_classes: Tuple[int, ...] = (COCO_PERSON_CLASS,),
        use_edgetpu: bool = False,
    ) -> None:
        self.score_threshold = score_threshold
        self.target_classes = set(target_classes)

        try:
            if use_edgetpu:
                from pycoral.utils.edgetpu import make_interpreter
                self.interpreter = make_interpreter(str(model_path))
            else:
                import tflite_runtime.interpreter as tflite
                self.interpreter = tflite.Interpreter(model_path=str(model_path))
        except ImportError:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=str(model_path))

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        input_shape = self.input_details[0]["shape"]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

    def preprocess(self, bgr_frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_width, self.input_height))
        return np.expand_dims(resized, axis=0).astype(
            self.input_details[0]["dtype"]
        )

    def detect(self, bgr_frame: np.ndarray) -> List[dict]:
        """Run detection and return list of {box, class_id, score} dicts."""
        input_data = self.preprocess(bgr_frame)
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
            if class_id not in self.target_classes:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            results.append({
                "box": (float(xmin), float(ymin), float(xmax), float(ymax)),
                "class_id": class_id,
                "score": float(scores[i]),
            })
        return results

    @staticmethod
    def is_in_lane(
        box: Tuple[float, float, float, float],
        lane_x_min: float = LANE_REGION_X_MIN,
        lane_x_max: float = LANE_REGION_X_MAX,
        lane_y_min: float = LANE_REGION_Y_MIN,
    ) -> bool:
        """Check if a normalised bounding box overlaps the lane region."""
        xmin, ymin, xmax, ymax = box
        x_overlap = xmin < lane_x_max and xmax > lane_x_min
        y_overlap = ymax > lane_y_min
        return x_overlap and y_overlap

    def detect_obstacle_in_lane(self, bgr_frame: np.ndarray) -> bool:
        """Return True if any target-class object is detected in the lane region."""
        detections = self.detect(bgr_frame)
        for det in detections:
            if self.is_in_lane(det["box"]):
                return True
        return False
