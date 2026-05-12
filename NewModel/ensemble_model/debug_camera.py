"""
Standalone browser debug stream for arrow/obstacle/lane model outputs.

Run this on the Pi from the same folder as model.py:
    python3 debug_camera.py --camera 0 --port 8081

Then open:
    http://<pi-ip-address>:8081

By default this does not drive the car. With --enable-control it also exposes
manual steering/speed buttons in the browser.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np

import model as car_model


latest_jpeg: Optional[bytes] = None
latest_capture: dict[str, np.ndarray] = {}
latest_lock = threading.Lock()
capture_enabled = False
capture_lock = threading.Lock()
stop_event = threading.Event()
manual_control = None
lane_activation_debugger = None
CAPTURE_DIR = Path(
    os.environ.get("PICAR_CAPTURE_DIR", Path(__file__).resolve().parent / "captures")
).expanduser()


def _resolve_control_config(config_file: Optional[str]) -> Optional[str]:
    if config_file:
        return str(Path(config_file).expanduser())

    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path("/home/pi/SunFounder_PiCar-V/remote_control/remote_control/driver/config"),
        script_dir / "remote_control" / "remote_control" / "driver" / "config",
        script_dir / "picarfolders" / "remote_control" / "remote_control" / "driver" / "config",
        script_dir.parent / "remote_control" / "remote_control" / "driver" / "config",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


class ManualCarControl:
    def __init__(self, enabled: bool, config_file: Optional[str], control_path: Optional[str]) -> None:
        self.enabled = False
        self.status = "disabled"
        self.config_file = _resolve_control_config(config_file)
        self.lock = threading.Lock()
        self.fw = None
        self.bw = None
        self.angle = 90
        self.speed = 0
        self.min_angle = 45
        self.max_angle = 135
        self.straight_angle = 90

        if not enabled:
            return

        try:
            if control_path:
                sys.path.insert(0, control_path)
            import picar
            from picar import back_wheels, front_wheels

            picar.setup()
            kwargs = {"debug": False}
            if self.config_file:
                kwargs["db"] = self.config_file
            self.fw = front_wheels.Front_Wheels(**kwargs)
            self.bw = back_wheels.Back_Wheels(**kwargs)
            self.bw.ready()
            self.fw.ready()
            self.straight_angle = int(getattr(self.fw, "_straight_angle", 90))
            self.min_angle = int(getattr(self.fw, "_min_angle", 45))
            self.max_angle = int(getattr(self.fw, "_max_angle", 135))
            self.angle = self.straight_angle
            self.enabled = True
            config_label = self.config_file if self.config_file else "picar default"
            self.status = (
                f"ready straight={self.straight_angle} min={self.min_angle} "
                f"max={self.max_angle} config={config_label}"
            )
        except Exception as exc:
            self.status = f"unavailable: {exc}"
            self.enabled = False

    def command(self, angle: Optional[int] = None, speed: Optional[int] = None) -> str:
        if not self.enabled or self.fw is None or self.bw is None:
            return f"control {self.status}"
        with self.lock:
            if angle is not None:
                self.angle = max(self.min_angle, min(self.max_angle, int(angle)))
                self.fw.turn(self.angle)
            if speed is not None:
                self.speed = max(-100, min(100, int(speed)))
                if self.speed < 0:
                    self.bw.backward()
                    self.bw.speed = abs(self.speed)
                elif self.speed == 0:
                    self.bw.stop()
                else:
                    self.bw.forward()
                    self.bw.speed = self.speed
            return f"angle={self.angle} speed={self.speed}"

    def stop(self) -> None:
        if not self.enabled or self.bw is None:
            return
        with self.lock:
            self.speed = 0
            self.bw.stop()


class LaneActivationDebugger:
    def __init__(
        self,
        enabled: bool,
        mobilenet_checkpoint: Path,
        nvidia_checkpoint: Path,
        device: str,
    ) -> None:
        self.enabled = False
        self.status = "disabled"
        self.models = {}
        if not enabled:
            return

        try:
            import torch
            import torch.nn as nn

            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
            self.torch = torch
            self.nn = nn

            loaded = []
            for name, checkpoint in (
                ("mobilenet", mobilenet_checkpoint),
                ("nvidia", nvidia_checkpoint),
            ):
                try:
                    state = self._load_model(name, Path(checkpoint))
                except Exception as exc:
                    print(f"[debug_camera] Lane activations unavailable for {name}: {exc}")
                    state = None
                if state is not None:
                    self.models[name] = state
                    loaded.append(name)
            if loaded:
                self.enabled = True
                self.status = "ready: " + ", ".join(loaded)
            else:
                self.status = "unavailable: no lane checkpoints loaded"
        except Exception as exc:
            self.status = f"unavailable: {exc}"

    def _load_model(self, name: str, checkpoint: Path):
        if not checkpoint.is_file():
            print(f"[debug_camera] Lane activation checkpoint missing for {name}: {checkpoint}")
            return None
        ckpt = self.torch.load(str(checkpoint), map_location=self.device)
        height = int(ckpt.get("height", 80))
        width = int(ckpt.get("width", 160))
        arch = str(ckpt.get("arch", "nvidia"))
        crop_top_ratio = float(ckpt.get("crop_top_ratio", car_model.CROP_TOP_RATIO))
        model = self._build_model(arch, height, width)
        state_dict = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        target_layer = self._find_last_conv(model)
        data = {
            "name": name,
            "model": model,
            "height": height,
            "width": width,
            "crop_top_ratio": crop_top_ratio,
            "activation": None,
        }

        def forward_hook(_module, _inputs, output):
            data["activation"] = output
            output.retain_grad()

        target_layer.register_forward_hook(forward_hook)
        return data

    def _build_model(self, arch: str, height: int, width: int):
        nn = self.nn
        if arch == "nvidia":
            class NvidiaLaneNet(nn.Module):
                def __init__(self, model_height: int, model_width: int) -> None:
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 24, kernel_size=5, stride=2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(24, 36, kernel_size=5, stride=2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(36, 48, kernel_size=5, stride=2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(48, 64, kernel_size=3),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 64, kernel_size=3),
                        nn.ReLU(inplace=True),
                    )
                    with self_torch_no_grad():
                        dummy = self_torch_zeros(1, 3, int(model_height), int(model_width))
                        flat = self.features(dummy).numel()
                    self.head = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(flat, 100),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.15),
                        nn.Linear(100, 50),
                        nn.ReLU(inplace=True),
                        nn.Linear(50, 10),
                        nn.ReLU(inplace=True),
                        nn.Linear(10, 1),
                    )

                def forward(self, x):
                    return self.head(self.features(x)).squeeze(-1)

            self_torch_no_grad = self.torch.no_grad
            self_torch_zeros = self.torch.zeros
            return NvidiaLaneNet(height, width)

        newmodel_dir = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(newmodel_dir))
        from train_lane_model import build_model
        return build_model(arch, height=height, width=width, pretrained=False)

    def _find_last_conv(self, model):
        last_conv = None
        for module in model.modules():
            if isinstance(module, self.nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d layer found for Grad-CAM")
        return last_conv

    def render_panels(self, frame: np.ndarray, selected: str, width: int, height: int) -> list[np.ndarray]:
        if not self.enabled:
            return [_preview_panel(np.zeros((1, 1, 3), dtype=np.uint8), f"Lane activations: {self.status}", width, height)]
        names = ["mobilenet", "nvidia"] if selected == "both" else [selected]
        panels = []
        for name in names:
            state = self.models.get(name)
            if state is None:
                panels.append(_preview_panel(np.zeros((1, 1, 3), dtype=np.uint8), f"{name} Grad-CAM: unavailable", width, height))
                continue
            panels.append(self._render_one(frame, state, width, height))
        return panels

    def _render_one(self, frame: np.ndarray, state: dict, width: int, height: int) -> np.ndarray:
        torch = self.torch
        model = state["model"]
        if car_model.FLIP_INPUT:
            frame = cv2.flip(frame, 1)
        crop_y = int(frame.shape[0] * state["crop_top_ratio"])
        crop = frame[crop_y:, :, :]
        resized = cv2.resize(crop, (state["width"], state["height"]), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normed = (rgb.astype(np.float32) / 255.0 - car_model.IMAGENET_MEAN) / car_model.IMAGENET_STD
        tensor = torch.from_numpy(normed.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        model.zero_grad(set_to_none=True)
        output = model(tensor).reshape(-1)[0]
        output.backward()

        activation = state["activation"]
        gradient = activation.grad if activation is not None else None
        if activation is None or gradient is None:
            return _preview_panel(resized, f"{state['name']} Grad-CAM: no gradients", width, height)

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((activation * weights).sum(dim=1))[0]
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(_normalize_map(cam), (state["width"], state["height"]), interpolation=cv2.INTER_LINEAR)
        overlay = _heatmap_overlay(resized, cam)
        return _preview_panel(overlay, f"{state['name']} Grad-CAM {float(output.detach().cpu()):.1f}", width, height)


def _put_text(image, text: str, xy: tuple[int, int], scale: float = 0.5, color=(235, 235, 235), thickness: int = 1) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_section(panel, title: str, x: int, y: int, width: int, height: int) -> None:
    cv2.rectangle(panel, (x, y), (x + width, y + height), (31, 34, 39), -1)
    cv2.rectangle(panel, (x, y), (x + width, y + height), (72, 78, 88), 1)
    _put_text(panel, title, (x + 10, y + 22), 0.55, (255, 255, 255), 1)


def _draw_metric(panel, label: str, value: str, x: int, y: int, color=(235, 235, 235)) -> None:
    _put_text(panel, label, (x, y), 0.42, (150, 160, 172), 1)
    _put_text(panel, value, (x + 95, y), 0.52, color, 1)


def _draw_badge(panel, text: str, x: int, y: int, color) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.rectangle(panel, (x, y - 17), (x + tw + 16, y + 7), color, -1)
    cv2.rectangle(panel, (x, y - 17), (x + tw + 16, y + 7), (245, 245, 245), 1)
    _put_text(panel, text, (x + 8, y), 0.48, (255, 255, 255), 1)


def _preview_panel(image: np.ndarray, title: str, width: int, height: int) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (18, 20, 24)
    if image.size > 0:
        ih, iw = image.shape[:2]
        scale = min(width / max(iw, 1), (height - 26) / max(ih, 1))
        new_w = max(1, int(iw * scale))
        new_h = max(1, int(ih * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x = (width - new_w) // 2
        y = 26 + (height - 26 - new_h) // 2
        panel[y:y + new_h, x:x + new_w] = resized
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (72, 78, 88), 1)
    _put_text(panel, title, (8, 18), 0.5, (255, 255, 255), 1)
    return panel


def _normalize_map(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    vmin = float(values.min()) if values.size else 0.0
    vmax = float(values.max()) if values.size else 0.0
    if vmax <= vmin:
        return np.zeros_like(values, dtype=np.float32)
    return (values - vmin) / (vmax - vmin)


def _heatmap_overlay(image: np.ndarray, heat: np.ndarray) -> np.ndarray:
    heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.55, heat_bgr, 0.45, 0)


def _arrow_occlusion_panel(
    runtime: car_model.Model,
    arrow_image: np.ndarray,
    predicted_class: str,
    target_class: str,
    width: int,
    height: int,
    patch: int,
    stride: int,
) -> np.ndarray:
    if not runtime.arrow.available or arrow_image.size == 0:
        return _preview_panel(np.zeros((1, 1, 3), dtype=np.uint8), "Arrow occlusion: unavailable", width, height)

    classifier = runtime.arrow
    input_image = cv2.resize(arrow_image, (classifier.width, classifier.height), interpolation=cv2.INTER_AREA)
    target = predicted_class if target_class == "auto" else target_class
    _, baseline_conf, baseline_probs = classifier.predict_with_probabilities(input_image)
    baseline = baseline_probs.get(target, baseline_conf if target == predicted_class else 0.0)

    patch = max(4, min(int(patch), classifier.width, classifier.height))
    stride = max(1, int(stride))
    heat = np.zeros((classifier.height, classifier.width), dtype=np.float32)
    counts = np.zeros_like(heat)

    for y in range(0, classifier.height - patch + 1, stride):
        for x in range(0, classifier.width - patch + 1, stride):
            occluded = input_image.copy()
            occluded[y:y + patch, x:x + patch, :] = 127
            _, _, probs = classifier.predict_with_probabilities(occluded)
            drop = max(0.0, baseline - probs.get(target, 0.0))
            heat[y:y + patch, x:x + patch] += drop
            counts[y:y + patch, x:x + patch] += 1.0

    heat = heat / np.maximum(counts, 1.0)
    overlay = _heatmap_overlay(input_image, _normalize_map(heat))
    title = f"Arrow occlusion: {target} {baseline:.2f}"
    return _preview_panel(overlay, title, width, height)


def _build_status_panel(debug: dict, width: int = 640, height: int = 210):
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (18, 20, 24)

    event = debug["event"]
    safety = debug["safety"]
    final_angle, speed = debug["decision"]
    state = debug.get("controller_state", "lane_keep")

    _put_text(panel, "PiCar Model Debug - No Drive Mode", (14, 24), 0.62, (255, 255, 255), 1)
    state_color = (38, 126, 220) if state == "lane_keep" else (34, 139, 80)
    if state.startswith("arrow"):
        state_color = (160, 105, 30)
    if state == "obstacle_stop":
        state_color = (38, 38, 190)
    _draw_badge(panel, state, 430, 23, state_color)

    _draw_section(panel, "Lane", 10, 42, 300, 74)
    _draw_metric(panel, "ensemble", f"{debug['raw_angle']:.1f}", 24, 73)
    _draw_metric(panel, "mobilenet", f"{debug['mobilenet_raw']:.1f}", 24, 96)
    _draw_metric(panel, "nvidia", f"{debug['nvidia_raw']:.1f}", 166, 96)

    _draw_section(panel, "Control", 330, 42, 300, 74)
    angle_color = (80, 210, 120) if 80 <= final_angle <= 100 else (80, 180, 255)
    _draw_metric(panel, "angle", str(final_angle), 344, 73, angle_color)
    _draw_metric(panel, "speed", str(speed), 484, 73, (80, 210, 120))
    _draw_metric(panel, "requested", f"{debug.get('requested_angle', debug['model_angle']):.1f}", 344, 96)

    _draw_section(panel, "Arrow", 10, 126, 300, 72)
    arrow_color = (90, 190, 90) if event.arrow == "none" else (75, 170, 255)
    if event.arrow == "left":
        arrow_color = (80, 210, 255)
    elif event.arrow == "right":
        arrow_color = (120, 220, 120)
    _draw_metric(panel, "class", event.arrow, 24, 158, arrow_color)
    _draw_metric(panel, "confidence", f"{event.arrow_confidence:.3f}", 24, 181)
    _draw_metric(panel, "pending", event.pending_turn, 166, 158)
    _draw_metric(panel, "frames", str(event.turn_frames_left), 166, 181)

    _draw_section(panel, "Obstacle / Safety", 330, 126, 300, 72)
    obstacle_color = (80, 80, 230) if event.obstacle_seen else (90, 190, 90)
    _draw_metric(panel, "obstacle", str(event.obstacle_seen), 344, 158, obstacle_color)
    _draw_metric(panel, "score", f"{event.obstacle_score:.3f}", 484, 158)
    _draw_metric(panel, "safety", safety.reason, 344, 181)
    _draw_metric(panel, "corr", f"{safety.correction:.1f}", 484, 181)

    return panel


def _draw_debug(frame, debug: dict, runtime: car_model.Model, args: argparse.Namespace):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    event = debug["event"]
    safety = debug["safety"]
    final_angle, speed = debug["decision"]

    if event.arrow_box is not None:
        xmin, ymin, xmax, ymax = event.arrow_box
        p1 = (int(xmin * w), int(ymin * h))
        p2 = (int(xmax * w), int(ymax * h))
        cv2.rectangle(overlay, p1, p2, (255, 255, 0), 2)
        _draw_badge(overlay, f"ARROW {car_model.ARROW_INPUT_MODE.upper()}", 104, 22, (130, 115, 35))
    elif car_model.ARROW_INPUT_MODE == "full_frame" or car_model.ARROW_USE_FULL_FRAME:
        cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (255, 255, 0), 2)
        _draw_badge(overlay, "ARROW FULL FRAME", 104, 22, (130, 115, 35))
    else:
        x1, y1, x2, y2 = car_model.ARROW_ROI
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)
        _draw_badge(overlay, "ARROW ROI", 104, 22, (130, 115, 35))

    if event.obstacle_box is not None:
        xmin, ymin, xmax, ymax = event.obstacle_box
        p1 = (int(xmin * w), int(ymin * h))
        p2 = (int(xmax * w), int(ymax * h))
        cv2.rectangle(overlay, p1, p2, (0, 0, 255), 2)
        cv2.putText(
            overlay,
            f"obstacle {event.obstacle_score:.2f}",
            (p1[0], max(18, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
        )

    crop_y = int(h * car_model.CROP_TOP_RATIO)
    safety_y = int(h * 0.45)
    cv2.line(overlay, (0, crop_y), (w - 1, crop_y), (255, 0, 255), 1)
    cv2.line(overlay, (0, safety_y), (w - 1, safety_y), (0, 165, 255), 1)
    cv2.line(overlay, (w // 2, safety_y), (w // 2, h - 1), (255, 0, 0), 1)

    _draw_badge(overlay, "CAMERA", 8, 22, (55, 89, 160))

    mask = runtime.safety_monitor.mask(frame)
    mask_full = frame.copy()
    mask_full[:] = 0
    mask_full[safety_y:, :] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    _draw_badge(mask_full, "OPENCV MASK", 8, 22, (80, 80, 80))

    arrow_image, _ = runtime._arrow_input_image(frame)
    arrow_panel = _preview_panel(arrow_image, f"Arrow input: {car_model.ARROW_INPUT_MODE}", overlay.shape[1], overlay.shape[0])
    panels = [overlay, mask_full, arrow_panel]
    if args.lane_activations:
        if lane_activation_debugger is None:
            panels.append(_preview_panel(np.zeros((1, 1, 3), dtype=np.uint8), "Lane activations: unavailable", overlay.shape[1], overlay.shape[0]))
        else:
            panels.extend(lane_activation_debugger.render_panels(frame, args.lane_activation_model, overlay.shape[1], overlay.shape[0]))
    if args.arrow_occlusion:
        panels.append(
            _arrow_occlusion_panel(
                runtime,
                arrow_image,
                event.arrow,
                args.arrow_occlusion_class,
                overlay.shape[1],
                overlay.shape[0],
                args.occlusion_patch,
                args.occlusion_stride,
            )
        )
    top = cv2.hconcat(panels)
    panel = _build_status_panel(debug, width=top.shape[1], height=210)
    return cv2.vconcat([top, panel])


def _save_capture_images(capture: dict[str, np.ndarray]) -> list[str]:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    saved = []
    for name, image in capture.items():
        path = CAPTURE_DIR / f"{stamp}_{name}.jpg"
        if cv2.imwrite(str(path), image):
            saved.append(path.name)
    return saved


def _capture_loop(args: argparse.Namespace) -> None:
    global latest_jpeg, latest_capture

    # The standalone debug script owns the web server. Disable model.py's
    # built-in stream so --port controls the only server this process starts.
    car_model.ENABLE_DEBUG_STREAM = False

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Stop autopilot first if it is using the camera.")

    runtime = car_model.Model()
    delay = 1.0 / max(args.fps, 1)

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue

        debug = runtime.predict_debug(frame)
        image = _draw_debug(frame, debug, runtime, args)
        arrow_image, _ = runtime._arrow_input_image(frame)
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
        if ok:
            capture = {
                "raw": frame.copy(),
                "debug": image.copy(),
                "arrow": arrow_image.copy(),
            }
            with latest_lock:
                latest_jpeg = encoded.tobytes()
                latest_capture = capture
            with capture_lock:
                saving = capture_enabled
            if saving:
                _save_capture_images(capture)
        time.sleep(delay)

    cap.release()


class Handler(BaseHTTPRequestHandler):
    def _write_text(self, status: int, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _save_capture(self) -> None:
        with latest_lock:
            capture = {name: image.copy() for name, image in latest_capture.items()}

        if not capture:
            self._write_text(503, "No frame available yet")
            return

        saved = _save_capture_images(capture)
        if not saved:
            self._write_text(500, "Capture failed")
            return
        self._write_text(200, f"Saved {', '.join(saved)}")

    def _toggle_capture(self) -> None:
        global capture_enabled
        with capture_lock:
            capture_enabled = not capture_enabled
            enabled = capture_enabled
        state = "on" if enabled else "off"
        self._write_text(200, f"capture={state}")

    def do_POST(self) -> None:
        if self.path == "/capture":
            self._save_capture()
            return
        if self.path == "/capture_toggle":
            self._toggle_capture()
            return
        if self.path == "/control":
            self._control()
            return
        self.send_error(404)

    def _control(self) -> None:
        global manual_control
        if manual_control is None:
            self._write_text(503, "control unavailable")
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") if length else "{}")
        except json.JSONDecodeError:
            self._write_text(400, "invalid json")
            return
        if payload.get("stop"):
            manual_control.stop()
            self._write_text(200, "stopped")
            return
        angle = payload.get("angle")
        speed = payload.get("speed")
        self._write_text(200, manual_control.command(angle=angle, speed=speed))

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            control_status = manual_control.status if manual_control is not None else "disabled"
            control_enabled = bool(manual_control is not None and manual_control.enabled)
            straight_angle = manual_control.straight_angle if manual_control is not None else 90
            min_angle = manual_control.min_angle if manual_control is not None else 45
            max_angle = manual_control.max_angle if manual_control is not None else 135
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = f"""<!doctype html>
<html>
<head><title>PiCar Model Debug</title></head>
<body style="margin:0;background:#101114;color:#eee;font-family:Arial,sans-serif">
<div style="padding:10px 14px;border-bottom:1px solid #333;background:#16181d;display:flex;gap:10px;align-items:center;flex-wrap:wrap">
  <strong>PiCar Model Debug</strong>
  <span style="color:#9aa4b2;margin-left:12px">{"Manual Control" if control_enabled else "No Drive Mode"}</span>
  <button id="capture" style="margin-left:auto;background:#2d6cdf;color:#fff;border:0;border-radius:4px;padding:7px 10px;cursor:pointer">Capture</button>
  <button id="toggle" style="background:#2a2d35;color:#fff;border:1px solid #555;border-radius:4px;padding:6px 10px;cursor:pointer">Start Capture</button>
  <span id="status" style="color:#9aa4b2;font-size:13px"></span>
</div>
<div id="drive" style="display:{"block" if control_enabled else "none"};padding:10px 14px;border-bottom:1px solid #333;background:#14161b">
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
    <button data-speed="35" style="padding:8px 12px">Forward</button>
    <button data-angle="{min_angle}" style="padding:8px 12px">Left</button>
    <button data-angle="{straight_angle}" data-speed="0" style="padding:8px 12px;background:#b43232;color:white">Stop</button>
    <button data-angle="{max_angle}" style="padding:8px 12px">Right</button>
    <button data-speed="-30" style="padding:8px 12px">Reverse</button>
    <label>Angle <input id="angle" type="range" min="{min_angle}" max="{max_angle}" value="{straight_angle}"></label>
    <label>Speed <input id="speed" type="range" min="-100" max="100" value="0"></label>
    <button id="apply" style="padding:8px 12px">Apply</button>
  </div>
</div>
<div style="display:{"none" if control_enabled else "block"};padding:8px 14px;color:#d0a56a;background:#18130f;border-bottom:1px solid #443">
  Manual car control disabled/unavailable: {control_status}
</div>
<div style="padding:10px">
  <img id="stream" src="/stream" style="max-width:100%;height:auto;border:1px solid #333;display:block">
</div>
<script>
const toggle = document.getElementById("toggle");
const capture = document.getElementById("capture");
const status = document.getElementById("status");
const angle = document.getElementById("angle");
const speed = document.getElementById("speed");
let saving = false;
const angleStep = 3;
const forwardSpeed = 35;
const reverseSpeed = -30;
const minAngle = Number(angle ? angle.min : {min_angle});
const maxAngle = Number(angle ? angle.max : {max_angle});
const straightAngle = {straight_angle};

function clamp(value, minimum, maximum) {{
  return Math.max(minimum, Math.min(maximum, value));
}}

function currentAngle() {{
  return Number(angle ? angle.value : straightAngle);
}}

function currentSpeed() {{
  return Number(speed ? speed.value : 0);
}}

function setControlValues(nextAngle, nextSpeed) {{
  const payload = {{}};
  if (nextAngle !== undefined) {{
    const clampedAngle = clamp(Number(nextAngle), minAngle, maxAngle);
    if (angle) angle.value = clampedAngle;
    payload.angle = clampedAngle;
  }}
  if (nextSpeed !== undefined) {{
    const clampedSpeed = clamp(Number(nextSpeed), -100, 100);
    if (speed) speed.value = clampedSpeed;
    payload.speed = clampedSpeed;
  }}
  sendControl(payload);
}}

async function sendControl(payload) {{
  try {{
    const response = await fetch("/control", {{
      method: "POST",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify(payload)
    }});
    status.textContent = await response.text();
  }} catch (error) {{
    status.textContent = "Control failed";
  }}
}}

document.querySelectorAll("[data-angle], [data-speed]").forEach((button) => {{
  button.addEventListener("click", () => {{
    const nextAngle = button.dataset.angle !== undefined ? Number(button.dataset.angle) : undefined;
    const nextSpeed = button.dataset.speed !== undefined ? Number(button.dataset.speed) : undefined;
    setControlValues(nextAngle, nextSpeed);
  }});
}});

const apply = document.getElementById("apply");
if (apply) {{
  apply.addEventListener("click", () => sendControl({{angle: Number(angle.value), speed: Number(speed.value)}}));
}}

document.addEventListener("keydown", (event) => {{
  if (event.target && ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(event.target.tagName)) {{
    return;
  }}
  if (!["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", " "].includes(event.key)) {{
    return;
  }}
  event.preventDefault();
  if (event.repeat && ["ArrowUp", "ArrowDown", " "].includes(event.key)) {{
    return;
  }}
  if (event.key === "ArrowUp") {{
    setControlValues(currentAngle(), forwardSpeed);
  }} else if (event.key === "ArrowDown") {{
    setControlValues(currentAngle(), reverseSpeed);
  }} else if (event.key === "ArrowLeft") {{
    setControlValues(currentAngle() - angleStep, currentSpeed());
  }} else if (event.key === "ArrowRight") {{
    setControlValues(currentAngle() + angleStep, currentSpeed());
  }} else if (event.key === " ") {{
    setControlValues(straightAngle, 0);
  }}
}});

toggle.addEventListener("click", async () => {{
  try {{
    const response = await fetch("/capture_toggle", {{method: "POST"}});
    const text = await response.text();
    saving = text.includes("capture=on");
    toggle.textContent = saving ? "Stop Capture" : "Start Capture";
    status.textContent = saving ? "Streaming and saving at 5 FPS" : "Streaming at 5 FPS";
  }} catch (error) {{
    status.textContent = "Capture toggle failed";
  }}
}});

capture.addEventListener("click", async () => {{
  status.textContent = "Capturing...";
  try {{
    const response = await fetch("/capture", {{method: "POST"}});
    status.textContent = await response.text();
  }} catch (error) {{
    status.textContent = "Capture failed";
  }}
}});

status.textContent = "Streaming at 5 FPS";
</script>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))
            return

        if self.path == "/capture":
            self._save_capture()
            return

        if not self.path.startswith("/stream"):
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        while not stop_event.is_set():
            with latest_lock:
                jpeg = latest_jpeg
            if jpeg is None:
                time.sleep(0.05)
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                time.sleep(0.2)
            except (BrokenPipeError, ConnectionResetError):
                break

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    global CAPTURE_DIR, manual_control, lane_activation_debugger

    parser = argparse.ArgumentParser(description="Run camera/model debug stream without driving the car.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--quality", type=int, default=75)
    parser.add_argument(
        "--lane-activations",
        action="store_true",
        help="Add live Grad-CAM activation heatmaps for the PyTorch lane model checkpoints.",
    )
    parser.add_argument(
        "--lane-activation-model",
        choices=["mobilenet", "nvidia", "both"],
        default="both",
        help="Which lane model activation heatmap to show when --lane-activations is enabled.",
    )
    parser.add_argument(
        "--mobilenet-checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "runs" / "lane_mobilenetv3_large_right_weighted" / "best.pt",
        help="PyTorch checkpoint for the MobileNet lane model used for Grad-CAM.",
    )
    parser.add_argument(
        "--nvidia-checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "runs" / "lane_nvidia_right_weighted" / "best.pt",
        help="PyTorch checkpoint for the Nvidia lane model used for Grad-CAM.",
    )
    parser.add_argument(
        "--activation-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for PyTorch lane Grad-CAM. Use cpu on the Pi unless CUDA is available.",
    )
    parser.add_argument(
        "--arrow-occlusion",
        action="store_true",
        help="Add a live occlusion-sensitivity panel for the arrow classifier. This is slower than the normal stream.",
    )
    parser.add_argument(
        "--arrow-occlusion-class",
        choices=["auto", "none", "left", "right"],
        default="auto",
        help="Class to visualize in the occlusion panel. auto uses the current predicted arrow class.",
    )
    parser.add_argument(
        "--occlusion-patch",
        type=int,
        default=32,
        help="Patch size for arrow occlusion sensitivity, in arrow-model input pixels.",
    )
    parser.add_argument(
        "--occlusion-stride",
        type=int,
        default=32,
        help="Stride for arrow occlusion sensitivity, in arrow-model input pixels. Smaller is smoother but slower.",
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=None,
        help="Directory for Capture/Start Capture images. Defaults to ./captures beside this script, or PICAR_CAPTURE_DIR.",
    )
    parser.add_argument(
        "--enable-control",
        action="store_true",
        help="Enable manual car controls in the web page. Leave off for camera-only debug.",
    )
    parser.add_argument(
        "--control-path",
        default=os.environ.get("PICAR_CONTROL_PATH"),
        help="Optional path containing the picar package, if it is not already importable.",
    )
    parser.add_argument(
        "--config-file",
        default=os.environ.get("PICAR_CONFIG_FILE"),
        help="Optional SunFounder/PiCar config file path for Front_Wheels/Back_Wheels. If omitted, common local remote_control config paths are tried.",
    )
    args = parser.parse_args()
    if args.capture_dir is not None:
        CAPTURE_DIR = args.capture_dir.expanduser().resolve()
    manual_control = ManualCarControl(args.enable_control, args.config_file, args.control_path)
    lane_activation_debugger = LaneActivationDebugger(
        args.lane_activations,
        args.mobilenet_checkpoint.expanduser().resolve(),
        args.nvidia_checkpoint.expanduser().resolve(),
        args.activation_device,
    )

    thread = threading.Thread(target=_capture_loop, args=(args,), daemon=True)
    thread.start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://<pi-ip-address>:{args.port} in your browser")
    print(f"Capture directory: {CAPTURE_DIR}")
    print(f"Lane activations: {lane_activation_debugger.status}")
    print(
        "Arrow occlusion: "
        f"{'on' if args.arrow_occlusion else 'off'} "
        f"(class={args.arrow_occlusion_class}, patch={args.occlusion_patch}, stride={args.occlusion_stride})"
    )
    print(f"Manual control: {manual_control.status}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if manual_control is not None:
            manual_control.stop()
        stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
