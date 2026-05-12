"""
Live debug interface for the Pi car model.

Usage:
    python -m car.inference.debug_interface --camera 0
    python -m car.inference.debug_interface --image car/data/debug/1023_sign.jpg

Keys:
    q / Esc  quit
    s        save the current annotated frame
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from car.inference.autopilot_model import Model


WINDOW_NAME = "Pi Car Debug View"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show camera feed and model decisions.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--image", type=Path, default=None, help="Run on one image instead of a camera.")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--save_dir", type=Path, default=Path("car/debug_captures"))
    return parser.parse_args()


def draw_status(frame: np.ndarray, debug: dict) -> np.ndarray:
    annotated = frame.copy()
    angle, speed = debug["decision"]
    lane_norm = float(debug["lane_angle_norm"])
    obstacle = bool(debug["obstacle_in_lane"])
    arrow = str(debug["arrow"])

    h, w = annotated.shape[:2]
    steering_x = int(np.interp(angle, [50, 120], [0, w - 1]))

    cv2.line(annotated, (w // 2, h), (steering_x, int(h * 0.55)), (0, 255, 255), 2)
    cv2.rectangle(annotated, (0, 0), (w, 82), (0, 0, 0), -1)
    cv2.putText(annotated, f"angle={angle:3d}  speed={speed:2d}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"lane_norm={lane_norm:.3f}  arrow={arrow}", (8, 47),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"obstacle={obstacle}  frame={debug['frame_count']}", (8, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


def save_frame(frame: np.ndarray, save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = save_dir / f"debug_{stamp}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def run_single_image(model: Model, image_path: Path, args: argparse.Namespace) -> None:
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    frame = cv2.resize(frame, (args.width, args.height))
    annotated = draw_status(frame, model.predict_debug(frame))
    cv2.imshow(WINDOW_NAME, annotated)
    cv2.waitKey(0)


def run_camera(model: Model, args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    print("Debug view running. Press q/Esc to quit, s to save a frame.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera returned no frame")

            frame = cv2.resize(frame, (args.width, args.height))
            annotated = draw_status(frame, model.predict_debug(frame))
            cv2.imshow(WINDOW_NAME, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                print(f"Saved {save_frame(annotated, args.save_dir)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    model = Model()
    if args.image is not None:
        run_single_image(model, args.image, args)
    else:
        run_camera(model, args)


if __name__ == "__main__":
    main()
