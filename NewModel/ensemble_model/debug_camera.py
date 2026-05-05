"""
Standalone browser debug stream for arrow/obstacle/lane model outputs.

Run this on the Pi from the same folder as model.py:
    python3 debug_camera.py --camera 0 --port 8081

Then open:
    http://<pi-ip-address>:8081

This does not drive the car. It only opens the camera, runs model.predict_debug()
on each frame, and streams the overlay to a browser.
"""
from __future__ import annotations

import argparse
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
CAPTURE_DIR = Path(__file__).resolve().parent / "captures"


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


def _draw_debug(frame, debug: dict, runtime: car_model.Model):
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

    top = cv2.hconcat([overlay, mask_full])
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
        image = _draw_debug(frame, debug, runtime)
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
        self.send_error(404)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"""<!doctype html>
<html>
<head><title>PiCar Model Debug</title></head>
<body style="margin:0;background:#101114;color:#eee;font-family:Arial,sans-serif">
<div style="padding:10px 14px;border-bottom:1px solid #333;background:#16181d;display:flex;gap:10px;align-items:center;flex-wrap:wrap">
  <strong>PiCar Model Debug</strong>
  <span style="color:#9aa4b2;margin-left:12px">No Drive Mode</span>
  <button id="capture" style="margin-left:auto;background:#2d6cdf;color:#fff;border:0;border-radius:4px;padding:7px 10px;cursor:pointer">Capture</button>
  <button id="toggle" style="background:#2a2d35;color:#fff;border:1px solid #555;border-radius:4px;padding:6px 10px;cursor:pointer">Start Capture</button>
  <span id="status" style="color:#9aa4b2;font-size:13px"></span>
</div>
<div style="padding:10px">
  <img id="stream" src="/stream" style="max-width:100%;height:auto;border:1px solid #333;display:block">
</div>
<script>
const toggle = document.getElementById("toggle");
const capture = document.getElementById("capture");
const status = document.getElementById("status");
let saving = false;

toggle.addEventListener("click", async () => {
  try {
    const response = await fetch("/capture_toggle", {method: "POST"});
    const text = await response.text();
    saving = text.includes("capture=on");
    toggle.textContent = saving ? "Stop Capture" : "Start Capture";
    status.textContent = saving ? "Streaming and saving at 5 FPS" : "Streaming at 5 FPS";
  } catch (error) {
    status.textContent = "Capture toggle failed";
  }
});

capture.addEventListener("click", async () => {
  status.textContent = "Capturing...";
  try {
    const response = await fetch("/capture", {method: "POST"});
    status.textContent = await response.text();
  } catch (error) {
    status.textContent = "Capture failed";
  }
});

status.textContent = "Streaming at 5 FPS";
</script>
</body>
</html>"""
            )
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
    parser = argparse.ArgumentParser(description="Run camera/model debug stream without driving the car.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--quality", type=int, default=75)
    args = parser.parse_args()

    thread = threading.Thread(target=_capture_loop, args=(args,), daemon=True)
    thread.start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://<pi-ip-address>:{args.port} in your browser")
    print("Press Ctrl+C to stop. This script does not command the car motors.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
