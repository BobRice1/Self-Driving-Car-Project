"""Event classifier inference wrappers."""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from car.inference.tflite_utils import dequantize_output, make_interpreter, quantize_input

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class EventPredictor:
    """Loads a trained EventClassifier and predicts class from a BGR ROI crop."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        classes: List[str],
        input_size: int = 96,
        device: str = "cpu",
        use_torchscript: bool = False,
    ) -> None:
        import torch

        self.device = torch.device(device)
        self.torch = torch
        self.classes = classes
        self.input_size = input_size

        if use_torchscript:
            self.model = torch.jit.load(str(checkpoint_path), map_location=self.device)
        else:
            from car.training.events.model import EventClassifier

            self.model = EventClassifier(
                num_classes=len(classes), pretrained=False, dropout=0.0,
            )
            ckpt = torch.load(str(checkpoint_path), map_location=self.device)
            state_dict = ckpt["model_state_dict"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)

        self.model.eval()

    def preprocess(self, bgr_roi: np.ndarray):
        torch = self.torch
        resized = cv2.resize(bgr_roi, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normed = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(normed.transpose(2, 0, 1)).unsqueeze(0)
        return tensor

    def predict(self, bgr_roi: np.ndarray) -> str:
        """Return predicted class name."""
        torch = self.torch
        tensor = self.preprocess(bgr_roi).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        idx = int(logits.argmax(dim=1).item())
        return self.classes[idx]

    def predict_with_confidence(self, bgr_roi: np.ndarray) -> tuple[str, float]:
        """Return (class_name, confidence)."""
        torch = self.torch
        tensor = self.preprocess(bgr_roi).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)
        return self.classes[int(idx.item())], float(conf.item())


class TFLiteEventPredictor:
    """TensorFlow Lite classifier for arrow/event predictions."""

    def __init__(self, model_path: str | Path, classes: List[str]) -> None:
        self.classes = classes
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

    def preprocess(self, bgr_roi: np.ndarray) -> np.ndarray:
        resized = cv2.resize(bgr_roi, (self.width, self.height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normed = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        if self.channels_first:
            normed = normed.transpose(2, 0, 1)
        batched = np.expand_dims(normed, axis=0)
        return quantize_input(batched, self.input_detail)

    def logits(self, bgr_roi: np.ndarray) -> np.ndarray:
        tensor = self.preprocess(bgr_roi)
        self.interpreter.set_tensor(self.input_detail["index"], tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_detail["index"])
        return dequantize_output(output, self.output_detail).reshape(-1)

    def predict(self, bgr_roi: np.ndarray) -> str:
        logits = self.logits(bgr_roi)
        return self.classes[int(np.argmax(logits))]

    def predict_with_confidence(self, bgr_roi: np.ndarray) -> tuple[str, float]:
        logits = self.logits(bgr_roi)
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        probs = exp / np.maximum(exp.sum(), 1e-12)
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])
