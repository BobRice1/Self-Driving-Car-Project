"""Lane-following model inference wrapper."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from car.inference.tflite_utils import dequantize_output, make_interpreter, quantize_input

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

LANE_HEIGHT = 120
LANE_WIDTH = 160
CROP_RATIO = 0.4


class LanePredictor:
    """Loads a trained LaneFollower checkpoint and predicts steering angle from a BGR frame."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        use_torchscript: bool = False,
    ) -> None:
        import torch

        self.device = torch.device(device)
        self.torch = torch

        if use_torchscript:
            self.model = torch.jit.load(str(checkpoint_path), map_location=self.device)
        else:
            from car.training.lane.model import LaneFollower

            self.model = LaneFollower(pretrained=False, dropout=0.0)
            ckpt = torch.load(str(checkpoint_path), map_location=self.device)
            state_dict = ckpt["model_state_dict"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)

        self.model.eval()

    def preprocess(self, bgr_frame: np.ndarray):
        """Crop lower portion, resize, normalise, return (1, 3, H, W) tensor."""
        torch = self.torch
        h = bgr_frame.shape[0]
        crop_start = int(h * CROP_RATIO)
        cropped = bgr_frame[crop_start:, :, :]

        resized = cv2.resize(cropped, (LANE_WIDTH, LANE_HEIGHT))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normed = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(normed.transpose(2, 0, 1)).unsqueeze(0)
        return tensor

    def predict(self, bgr_frame: np.ndarray) -> float:
        """Return steering angle in [0, 1]."""
        torch = self.torch
        tensor = self.preprocess(bgr_frame).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
        return float(torch.clamp(output, 0.0, 1.0).item())


class TFLiteLanePredictor:
    """TensorFlow Lite lane predictor for the Raspberry Pi deployment environment."""

    def __init__(self, model_path: str | Path) -> None:
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]

        shape = [int(v) for v in self.input_detail["shape"]]
        if len(shape) != 4:
            raise ValueError(f"Expected 4D TFLite input shape, got {shape}")

        self.channels_first = shape[1] == 3
        if self.channels_first:
            self.height = shape[2]
            self.width = shape[3]
        else:
            self.height = shape[1]
            self.width = shape[2]

    def preprocess(self, bgr_frame: np.ndarray) -> np.ndarray:
        h = bgr_frame.shape[0]
        crop_start = int(h * CROP_RATIO)
        cropped = bgr_frame[crop_start:, :, :]

        resized = cv2.resize(cropped, (self.width, self.height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normed = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        if self.channels_first:
            normed = normed.transpose(2, 0, 1)
        batched = np.expand_dims(normed, axis=0)
        return quantize_input(batched, self.input_detail)

    def predict(self, bgr_frame: np.ndarray) -> float:
        tensor = self.preprocess(bgr_frame)
        self.interpreter.set_tensor(self.input_detail["index"], tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_detail["index"])
        output = dequantize_output(output, self.output_detail)
        return float(np.clip(output.reshape(-1)[0], 0.0, 1.0))
