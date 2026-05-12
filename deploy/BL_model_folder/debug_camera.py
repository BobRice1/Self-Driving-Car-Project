"""
Browser-based OpenCV debug viewer for the Pi car camera.

Run from the same folder as model.py on the Raspberry Pi:
    python3 debug_camera.py --camera 0 --port 8080

Then open this from your laptop:
    http://<pi-ip-address>:8080
"""
from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np

import model as car_model


latest_jpeg: Optional[bytes] = None
latest_lock = threading.Lock()
stop_event = threading.Event()


def _draw_debug(frame: np.ndarray, detector: car_model.LaneGeometryDetector) -> np.ndarray:
    lane = detector.estimate(frame)
    height, width = frame.shape[:2]
    crop_y = int(height * car_model.LANE_CROP_TOP_RATIO)

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

    text = (
        f"angle={lane.angle} target={lane.target_x:.2f} "
        f"ok={lane.ok} reason={lane.reason}"
    )
    cv2.rectangle(overlay, (0, 0), (width - 1, 42), (0, 0, 0), -1)
    cv2.putText(overlay, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(
        overlay,
        "yellow=corridor green=target blue=image-centre red=left orange=dashed",
        (8, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
    )

    crop = frame[crop_y:, :]
    mask = detector._line_mask(crop)
    mask_full = np.zeros_like(frame)
    mask_full[crop_y:, :] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(mask_full, "black-line mask", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return np.hstack([overlay, mask_full])


def _capture_loop(args: argparse.Namespace) -> None:
    global latest_jpeg

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Stop autopilot first if it is already using /dev/video0.")

    detector = car_model.LaneGeometryDetector()
    delay = 1.0 / max(args.fps, 1)

    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue

        debug = _draw_debug(frame, detector)
        ok, encoded = cv2.imencode(".jpg", debug, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
        if ok:
            with latest_lock:
                latest_jpeg = encoded.tobytes()
        time.sleep(delay)

    cap.release()


class DebugHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"""<!doctype html>
<html>
<head><title>Pi Car OpenCV Debug</title></head>
<body style="background:#111;color:#eee;font-family:Arial,sans-serif">
<h2>Pi Car OpenCV Debug</h2>
<p>Left: camera with lane overlay. Right: detected black-line mask.</p>
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
                time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                break

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream Pi car OpenCV lane debug view to a browser.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--quality", type=int, default=80)
    args = parser.parse_args()

    thread = threading.Thread(target=_capture_loop, args=(args,), daemon=True)
    thread.start()

    server = ThreadingHTTPServer((args.host, args.port), DebugHandler)
    print(f"Open http://<pi-ip-address>:{args.port} in your browser")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
