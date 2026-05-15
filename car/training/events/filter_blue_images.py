from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find images with visible blue arrow/sign artifacts.")
    parser.add_argument("source", type=Path, help="Folder to scan recursively.")
    parser.add_argument("destination", type=Path, help="Folder for matching images.")
    parser.add_argument("--apply", action="store_true", help="Actually copy/move files. Without this, only prints matches.")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying them.")
    parser.add_argument("--keep-structure", action="store_true", help="Preserve source subfolders under destination.")
    parser.add_argument("--min-blue-pixels", type=int, default=100, help="Minimum blue pixels required to match.")
    parser.add_argument("--min-blue-ratio", type=float, default=0.001, help="Minimum blue pixel fraction required to match.")
    parser.add_argument("--min-component-area", type=int, default=15, help="Minimum connected blue blob area required to match.")
    return parser.parse_args()


def is_blue_artifact(path: Path, min_blue_pixels: int, min_blue_ratio: float, min_component_area: int) -> tuple[bool, float, int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return False, 0.0, 0, 0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Tuned for small visible blue signs in the arrow event dataset.
    # The value threshold avoids treating dark curtains/shadows as blue artifacts.
    lower_blue = np.array([90, 45, 50], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    blue_pixels = int(np.count_nonzero(mask))
    ratio = blue_pixels / float(mask.size)
    largest_component = 0
    if blue_pixels:
        labels_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if labels_count > 1:
            largest_component = int(stats[1:, cv2.CC_STAT_AREA].max())

    matched = (
        blue_pixels >= min_blue_pixels
        and ratio >= min_blue_ratio
        and largest_component >= min_component_area
    )
    return matched, ratio, blue_pixels, largest_component


def make_destination(src: Path, source_root: Path, dest_root: Path, keep_structure: bool) -> Path:
    if keep_structure:
        return dest_root / src.relative_to(source_root)

    candidate = dest_root / src.name
    if not candidate.exists():
        return candidate

    for index in range(1, 100000):
        candidate = dest_root / f"{src.stem}_{index}{src.suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find free destination name for {src.name}")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if not source.is_dir():
        raise SystemExit(f"Source folder does not exist: {source}")

    matches: list[tuple[Path, float, int, int]] = []
    for path in source.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            matched, ratio, pixels, component = is_blue_artifact(
                path,
                args.min_blue_pixels,
                args.min_blue_ratio,
                args.min_component_area,
            )
            if matched:
                matches.append((path, ratio, pixels, component))

    print(f"Found {len(matches)} images with blue artifacts in {source}")
    print(f"Mode: {'MOVE' if args.move else 'COPY'} {'APPLY' if args.apply else 'DRY RUN'}")

    for path, ratio, pixels, component in matches:
        print(f"{ratio:.5f} pixels={pixels:5d} blob={component:4d} {path}")
        if not args.apply:
            continue
        out_path = make_destination(path, source, destination, args.keep_structure)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.move:
            shutil.move(str(path), str(out_path))
        else:
            shutil.copy2(str(path), str(out_path))


if __name__ == "__main__":
    main()
