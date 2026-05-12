"""
Auto-sort training images into event categories.

Arrows:  detected via blue-circle colour segmentation in HSV (no ML needed).
Objects: detected via a COCO-pretrained Faster R-CNN from torchvision.

Usage:
    python -m car.training.auto_sort_images --data_dir data --task arrows
    python -m car.training.auto_sort_images --data_dir data --task sort_signs_auto
    python -m car.training.auto_sort_images --data_dir data --task objects
    python -m car.training.auto_sort_images --data_dir data --task all
    python -m car.training.auto_sort_images --data_dir data --task arrows --debug_dir car/data/debug

Output structure:
    car/data/events/arrows/{sign_present,small_sign_review,left,right,none}/
    car/data/events/objects/{person,none}/
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")


def parse_args():
    p = argparse.ArgumentParser(description="Auto-sort images into event categories.")
    p.add_argument("--data_dir", type=Path, default=Path("data"))
    p.add_argument("--task", choices=["arrows", "objects", "all", "sort_signs", "sort_signs_auto"], default="all")
    p.add_argument("--out_dir", type=Path, default=Path("car/data/events"))
    p.add_argument("--min_blue_area", type=int, default=80,
                   help="Min blue-blob pixel area to count as a sign")
    p.add_argument("--max_blue_area", type=int, default=3000,
                   help="Max blue-blob pixel area (rejects large objects like pillars)")
    p.add_argument("--object_confidence", type=float, default=0.4,
                   help="Min detection score for objects")
    p.add_argument("--max_images", type=int, default=None,
                   help="Limit number of images to process (for testing)")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--debug_dir", type=Path, default=None,
                   help="Save annotated debug images showing detected blue blobs")
    p.add_argument("--review_dir", type=Path, default=None,
                   help="Save uncertain images here for manual review")
    p.add_argument("--auto_confidence", type=float, default=0.80,
                   help="Confidence threshold for auto-sorting sign direction")
    p.add_argument("--auto_min_examples", type=int, default=20,
                   help="Minimum reference examples needed per class for auto sign sorting")
    p.add_argument("--recover_none_confidence", type=float, default=0.995,
                   help="Confidence threshold for recovering missed signs from arrows/none")
    return p.parse_args()


def find_image_dir(data_dir: Path) -> Path:
    for candidate in [data_dir / "train_images", data_dir / "training_data"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image directory found in {data_dir}")


def collect_image_paths(image_dir: Path, max_images: int | None = None) -> List[Path]:
    paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if max_images is not None:
        paths = paths[:max_images]
    return paths


def resolve_device(mode: str) -> torch.device:
    if mode == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(mode)


# ---------------------------------------------------------------------------
# Arrow sorting with blue-circle colour detection (HSV)
# ---------------------------------------------------------------------------

ARROW_LABELS = ["left", "right", "none"]

# HSV range for the blue circular sign — broad enough to catch signs
# under varying indoor lighting.
BLUE_HSV_LOW = np.array([95, 80, 50], dtype=np.uint8)
BLUE_HSV_HIGH = np.array([135, 255, 255], dtype=np.uint8)

MIN_CIRCULARITY = 0.35
MIN_ASPECT = 0.4
MAX_ASPECT = 2.5
AUTO_MIN_CIRCULARITY = 0.2
AUTO_SIGN_CROP_SIZE = 48
SMALL_SIGN_REVIEW_DIR = "small_sign_review"


def find_best_blue_sign(
    bgr: np.ndarray,
    min_area: int = 80,
    max_area: int = 3000,
    min_circularity: float = MIN_CIRCULARITY,
) -> Tuple[np.ndarray | None, float, Tuple[int, int, int, int] | None]:
    """Return the strongest blue sign contour, its area, and bounding box."""
    img_h = bgr.shape[0]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_HSV_LOW, BLUE_HSV_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0, None

    best_cnt = None
    best_area = 0.0
    best_bbox = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue

        cy = y + h / 2
        if cy < img_h * 0.25:
            continue

        score = float(area) * (0.5 + circularity)
        if score > best_score:
            best_score = score
            best_cnt = cnt
            best_area = float(area)
            best_bbox = (x, y, w, h)

    return best_cnt, best_area, best_bbox


def detect_blue_sign(
    bgr: np.ndarray,
    min_area: int = 80,
    max_area: int = 3000,
) -> Tuple[bool, float]:
    """Detect whether a blue circular sign is present in the image."""
    _, area, bbox = find_best_blue_sign(
        bgr,
        min_area=min_area,
        max_area=max_area,
    )
    return bbox is not None, area


def extract_arrow_classifier_features(
    bgr: np.ndarray,
    crop_size: int = AUTO_SIGN_CROP_SIZE,
    min_area: int = 50,
    max_area: int = 4000,
    min_circularity: float = AUTO_MIN_CIRCULARITY,
) -> np.ndarray | None:
    """Extract a compact sign-centered feature vector for left/right/none sorting."""
    _, _, bbox = find_best_blue_sign(
        bgr,
        min_area=min_area,
        max_area=max_area,
        min_circularity=min_circularity,
    )
    if bbox is None:
        return None

    x, y, w, h = bbox
    pad = int(max(w, h) * 0.3)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(bgr.shape[1], x + w + pad)
    y1 = min(bgr.shape[0], y + h + pad)

    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blue = cv2.inRange(hsv, BLUE_HSV_LOW, BLUE_HSV_HIGH)

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    white = ((val > 120) & (sat < 90) & (blue == 0)).astype(np.uint8) * 255

    yy, xx = np.mgrid[:crop_size, :crop_size]
    radius = int(crop_size * 0.375)
    circle_mask = ((xx - crop_size / 2) ** 2 + (yy - crop_size / 2) ** 2) <= radius ** 2

    blue = np.where(circle_mask, blue, 0)
    white = np.where(circle_mask, white, 0)

    return np.concatenate(
        [
            gray.reshape(-1).astype(np.float32) / 255.0,
            blue.reshape(-1).astype(np.float32) / 255.0,
            white.reshape(-1).astype(np.float32) / 255.0,
        ]
    )


def load_logged_sign_present_names(log_csv: Path) -> set[str]:
    names: set[str] = set()
    if not log_csv.exists():
        return names

    with open(log_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("label") == "sign_present":
                names.add(row["image"])
    return names


def find_small_sign_review_candidate(
    bgr: np.ndarray,
) -> Tuple[bool, Tuple[int, int, int, int] | None]:
    """Detect tiny distant blue signs that are better sent to manual review."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_HSV_LOW, BLUE_HSV_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    img_h, img_w = bgr.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    small_parts: List[Tuple[int, int, int, int, float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 15 or area > 120:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.5:
            continue

        aspect = w / max(h, 1)
        if aspect < 0.6 or aspect > 1.8:
            continue

        small_parts.append((x, y, w, h, float(area)))

    small_parts.sort(key=lambda row: (row[1], row[0]))
    if len(small_parts) != 2:
        return False, None

    (x1, y1, w1, h1, _), (x2, y2, w2, h2, _) = small_parts
    center_x = (x1 + x2 + w1 + w2) / 4
    aligned = abs(x1 - x2) <= 4 and abs(w1 - w2) <= 4
    spaced = 4 <= (y2 - y1) <= 18
    positioned = 20 <= y1 <= img_h * 0.65 and 10 <= center_x <= img_w - 10

    if not (aligned and spaced and positioned):
        return False, None

    x0 = min(x1, x2)
    y0 = min(y1, y2)
    x1b = max(x1 + w1, x2 + w2)
    y1b = max(y1 + h1, y2 + h2)
    return True, (x0, y0, x1b - x0, y1b - y0)


def collect_arrow_reference_paths(out_dir: Path) -> List[Tuple[Path, str]]:
    """Collect already-reviewed sign images to train the auto sorter."""
    sign_present_names = load_logged_sign_present_names(out_dir / "arrow_sort_log.csv")
    samples: List[Tuple[Path, str]] = []

    for label in ("left", "right"):
        label_dir = out_dir / label
        if not label_dir.exists():
            continue
        for p in sorted(label_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((p, label))

    none_dir = out_dir / "none"
    if none_dir.exists():
        for p in sorted(none_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if sign_present_names and p.name not in sign_present_names:
                continue
            samples.append((p, "none"))

    return samples


def train_arrow_auto_classifier(
    out_dir: Path,
    min_examples_per_class: int,
):
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for --task sort_signs_auto") from exc

    reference_paths = collect_arrow_reference_paths(out_dir)
    if not reference_paths:
        raise RuntimeError(
            "No reviewed arrow samples found. Add examples to left/right/none first."
        )

    features: List[np.ndarray] = []
    labels: List[str] = []
    class_counts: Counter[str] = Counter()

    for path, label in reference_paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        feature = extract_arrow_classifier_features(img)
        if feature is None:
            continue
        features.append(feature)
        labels.append(label)
        class_counts[label] += 1

    for label in ARROW_LABELS:
        if class_counts[label] < min_examples_per_class:
            raise RuntimeError(
                f"Need at least {min_examples_per_class} usable '{label}' examples "
                f"for auto sign sorting, found {class_counts[label]}."
            )

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )
    clf.fit(np.stack(features), np.array(labels))
    return clf, dict(class_counts)


def sort_arrows(
    image_paths: List[Path],
    out_dir: Path,
    min_blue_area: int = 80,
    max_blue_area: int = 3000,
    debug_dir: Path | None = None,
) -> Dict[str, int]:
    """Phase 1: auto-split into sign_present/ vs none/ using colour detection."""
    for folder in ("sign_present", "none"):
        (out_dir / folder).mkdir(parents=True, exist_ok=True)
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    counts = {"sign_present": 0, "none": 0}
    log_rows = []

    for p in tqdm(image_paths, desc="Detecting signs (colour)"):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue

        found, area = detect_blue_sign(img, min_area=min_blue_area, max_area=max_blue_area)
        label = "sign_present" if found else "none"

        log_rows.append({
            "image": p.name, "label": label, "blue_area": f"{area:.0f}",
        })

        shutil.copy2(p, out_dir / label / p.name)
        counts[label] += 1

    log_csv = out_dir / "arrow_sort_log.csv"
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label", "blue_area"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nPhase 1 done: {counts['sign_present']} images with signs, {counts['none']} without.")
    if counts["sign_present"] > 0:
        print(
            f"Next run the auto or manual direction sorter:\n"
            f"  python -m car.training.auto_sort_images --data_dir data --task sort_signs_auto\n"
            f"  python -m car.training.auto_sort_images --data_dir data --task sort_signs"
        )

    return counts


def manual_sort_signs(out_dir: Path) -> Dict[str, int]:
    """Phase 2: show each sign_present image and let the user press L/R/N."""
    sign_dir = out_dir / "sign_present"
    small_review_dir = out_dir / SMALL_SIGN_REVIEW_DIR
    sign_dir.mkdir(parents=True, exist_ok=True)

    if small_review_dir.exists():
        for p in sorted(small_review_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                shutil.move(str(p), str(sign_dir / p.name))
        if not any(small_review_dir.iterdir()):
            small_review_dir.rmdir()

    for folder in ("left", "right", "none"):
        (out_dir / folder).mkdir(parents=True, exist_ok=True)

    images = sorted(sign_dir.iterdir())
    images = [p for p in images if p.suffix.lower() in IMAGE_EXTENSIONS]
    total = len(images)

    if total == 0:
        print("No images in sign_present/ or small_sign_review/ to sort.")
        return {}

    print(f"\n{'='*60}")
    print(f"  Manual arrow sorter — {total} images to classify")
    print(f"  Keys:  L = left arrow  |  R = right arrow  |  N = no arrow")
    print(f"         B = go back     |  Q = quit (progress is saved)")
    print(f"{'='*60}\n")

    counts = {"left": 0, "right": 0, "none": 0, "skipped": 0}
    i = 0

    while i < total:
        p = images[i]
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            i += 1
            continue

        display = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
        label_text = f"[{i+1}/{total}] L=left  R=right  N=none  B=back  Q=quit"
        cv2.putText(display, label_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, p.name, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Arrow Sorter", display)

        key = cv2.waitKey(0) & 0xFF

        if key == ord("l"):
            shutil.move(str(p), str(out_dir / "left" / p.name))
            counts["left"] += 1
            i += 1
        elif key == ord("r"):
            shutil.move(str(p), str(out_dir / "right" / p.name))
            counts["right"] += 1
            i += 1
        elif key == ord("n"):
            shutil.move(str(p), str(out_dir / "none" / p.name))
            counts["none"] += 1
            i += 1
        elif key == ord("b") and i > 0:
            i -= 1
            prev = images[i]
            for folder in ("left", "right", "none"):
                moved = out_dir / folder / prev.name
                if moved.exists():
                    shutil.move(str(moved), str(sign_dir / prev.name))
                    break
        elif key == ord("q"):
            counts["skipped"] = total - i
            break

    cv2.destroyAllWindows()
    print(f"\nSorting complete: {counts}")

    remaining = list(sign_dir.iterdir())
    if not any(p.suffix.lower() in IMAGE_EXTENSIONS for p in remaining):
        sign_dir.rmdir()
        print("Removed empty sign_present/ folder.")

    return counts


def recover_missed_signs(
    out_dir: Path,
    clf,
    confidence_threshold: float,
) -> Dict[str, int]:
    """Pull likely missed signs out of arrows/none into sign_present for auto-sorting."""
    none_dir = out_dir / "none"
    sign_dir = out_dir / "sign_present"
    sign_dir.mkdir(parents=True, exist_ok=True)

    if not none_dir.exists():
        return {"recovered": 0, "scanned": 0}

    counts = {"recovered": 0, "scanned": 0}
    log_rows = []
    images = sorted(
        p for p in none_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    for p in tqdm(images, desc="Recovering missed signs"):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        feature = extract_arrow_classifier_features(
            img,
            min_area=20,
            max_area=4000,
            min_circularity=0.15,
        )
        if feature is None:
            continue

        counts["scanned"] += 1
        probs = clf.predict_proba(feature.reshape(1, -1))[0]
        best_idx = int(np.argmax(probs))
        predicted_label = str(clf.classes_[best_idx])
        confidence = float(probs[best_idx])
        second_best = float(np.partition(probs, -2)[-2]) if len(probs) > 1 else 0.0
        margin = confidence - second_best
        recovered = predicted_label != "none" and confidence >= confidence_threshold

        if recovered:
            shutil.move(str(p), str(sign_dir / p.name))
            counts["recovered"] += 1

        log_rows.append({
            "image": p.name,
            "predicted_label": predicted_label,
            "confidence": f"{confidence:.4f}",
            "margin": f"{margin:.4f}",
            "status": "recovered" if recovered else "kept_none",
        })

    log_csv = out_dir / "arrow_recovery_log.csv"
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "predicted_label", "confidence", "margin", "status"],
        )
        writer.writeheader()
        writer.writerows(log_rows)

    if counts["recovered"] > 0:
        print(
            f"Recovered {counts['recovered']} likely missed signs from arrows/none "
            f"(confidence >= {confidence_threshold:.3f})."
        )
    else:
        print("No likely missed signs were recovered from arrows/none.")
    print(f"  Recovery log: {log_csv}")
    return counts


def recover_small_sign_review_candidates(out_dir: Path) -> Dict[str, int]:
    """Move tiny sign-like candidates out of none/ into a manual review bucket."""
    none_dir = out_dir / "none"
    review_dir = out_dir / SMALL_SIGN_REVIEW_DIR
    review_dir.mkdir(parents=True, exist_ok=True)

    if not none_dir.exists():
        return {"moved": 0, "scanned": 0}

    counts = {"moved": 0, "scanned": 0}
    log_rows = []

    images = sorted(
        p for p in none_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    for p in tqdm(images, desc="Recovering tiny sign candidates"):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue

        counts["scanned"] += 1
        found, bbox = find_small_sign_review_candidate(img)
        if found:
            shutil.move(str(p), str(review_dir / p.name))
            counts["moved"] += 1

        log_rows.append({
            "image": p.name,
            "status": "moved_to_small_sign_review" if found else "kept_none",
            "bbox": "" if bbox is None else ",".join(str(v) for v in bbox),
        })

    log_csv = out_dir / "arrow_small_sign_review_log.csv"
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "status", "bbox"])
        writer.writeheader()
        writer.writerows(log_rows)

    if counts["moved"] > 0:
        print(f"Moved {counts['moved']} tiny sign candidates into {review_dir.name}/ for manual classification.")
    else:
        print("No tiny sign candidates were moved into manual review.")
    print(f"  Small-sign review log: {log_csv}")
    return counts


def auto_sort_signs(
    out_dir: Path,
    confidence_threshold: float,
    min_examples_per_class: int,
    recovery_confidence_threshold: float,
    review_dir: Path | None = None,
) -> Dict[str, int]:
    """Auto-sort sign_present/ into left/right/none using reviewed in-domain examples."""
    clf, class_counts = train_arrow_auto_classifier(out_dir, min_examples_per_class)
    print(
        "Auto sign sorter trained on reviewed samples: "
        + ", ".join(f"{label}={class_counts[label]}" for label in ARROW_LABELS)
    )

    sign_dir = out_dir / "sign_present"
    sign_dir.mkdir(parents=True, exist_ok=True)
    for folder in ("sign_present", *ARROW_LABELS):
        (out_dir / folder).mkdir(parents=True, exist_ok=True)

    review_out = None
    if review_dir is not None:
        review_out = review_dir / "arrows"
        review_out.mkdir(parents=True, exist_ok=True)

    recovery_counts = recover_missed_signs(
        out_dir=out_dir,
        clf=clf,
        confidence_threshold=recovery_confidence_threshold,
    )
    small_review_counts = recover_small_sign_review_candidates(out_dir)

    images = sorted(
        p for p in sign_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        print("No images in sign_present/ to auto-sort after recovery.")
        return {
            "left": 0,
            "right": 0,
            "none": 0,
            "review": 0,
            "recovered": recovery_counts["recovered"],
            "small_review": small_review_counts["moved"],
        }

    counts = {"left": 0, "right": 0, "none": 0, "review": 0}
    log_rows = []

    for p in tqdm(images, desc="Auto-sorting signs"):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            counts["review"] += 1
            log_rows.append({
                "image": p.name,
                "predicted_label": "unreadable",
                "confidence": "0.0000",
                "margin": "0.0000",
                "status": "review",
            })
            continue

        feature = extract_arrow_classifier_features(img)
        if feature is None:
            counts["review"] += 1
            if review_out is not None:
                shutil.copy2(p, review_out / p.name)
            log_rows.append({
                "image": p.name,
                "predicted_label": "undetected_sign",
                "confidence": "0.0000",
                "margin": "0.0000",
                "status": "review",
            })
            continue

        probs = clf.predict_proba(feature.reshape(1, -1))[0]
        best_idx = int(np.argmax(probs))
        predicted_label = str(clf.classes_[best_idx])
        confidence = float(probs[best_idx])
        second_best = float(np.partition(probs, -2)[-2]) if len(probs) > 1 else 0.0
        margin = confidence - second_best

        if confidence >= confidence_threshold:
            shutil.move(str(p), str(out_dir / predicted_label / p.name))
            counts[predicted_label] += 1
            status = "sorted"
        else:
            counts["review"] += 1
            status = "review"
            if review_out is not None:
                shutil.copy2(p, review_out / p.name)

        log_rows.append({
            "image": p.name,
            "predicted_label": predicted_label,
            "confidence": f"{confidence:.4f}",
            "margin": f"{margin:.4f}",
            "status": status,
        })

    log_csv = out_dir / "arrow_direction_auto_log.csv"
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "predicted_label", "confidence", "margin", "status"],
        )
        writer.writeheader()
        writer.writerows(log_rows)

    remaining = [
        p for p in sign_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not remaining:
        sign_dir.rmdir()
        print("Removed empty sign_present/ folder.")

    print(
        "Auto-sort complete: "
        f"{counts['left']} left, {counts['right']} right, {counts['none']} none, "
        f"{counts['review']} left for review, "
        f"{recovery_counts['recovered']} recovered from none, "
        f"{small_review_counts['moved']} tiny candidates sent to manual review."
    )
    if counts["review"] > 0:
        print(
            "Finish the remainder manually with:\n"
            "  python -m car.training.auto_sort_images --data_dir data --task sort_signs"
        )
    elif small_review_counts["moved"] > 0:
        print(
            "Manual review is still needed for tiny distant signs with:\n"
            "  python -m car.training.auto_sort_images --data_dir data --task sort_signs"
        )
    print(f"  Log: {log_csv}")
    if review_out is not None:
        print(f"  Review copies: {review_out}")

    counts["recovered"] = recovery_counts["recovered"]
    counts["small_review"] = small_review_counts["moved"]
    return counts


# ---------------------------------------------------------------------------
# Object sorting with torchvision detection
# ---------------------------------------------------------------------------

COCO_PERSON_CLASS = 1  # torchvision COCO class IDs are 1-indexed

OBJECT_LABELS = ["person", "none"]


def load_detection_model(device: torch.device):
    from torchvision.models.detection import (
        fasterrcnn_mobilenet_v3_large_fpn,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    )

    print("Loading Faster R-CNN MobileNet V3 detector...")
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights).to(device)
    model.eval()
    return model


def detect_objects_batch(
    images_bgr: List[np.ndarray],
    model,
    device: torch.device,
    score_threshold: float,
) -> List[Tuple[str, float, list]]:
    """Detect objects in a batch. Returns (label, best_score, boxes) per image."""
    tensors = []
    for img in images_bgr:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(rgb.transpose(2, 0, 1)).to(device)
        tensors.append(t)

    with torch.no_grad():
        outputs = model(tensors)

    results = []
    for output in outputs:
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        boxes = output["boxes"].cpu().numpy()

        person_mask = (labels == COCO_PERSON_CLASS) & (scores >= score_threshold)
        if person_mask.any():
            best_score = float(scores[person_mask].max())
            person_boxes = boxes[person_mask].tolist()
            results.append(("person", best_score, person_boxes))
        else:
            results.append(("none", 0.0, []))

    return results


def sort_objects(
    image_paths: List[Path],
    out_dir: Path,
    device: torch.device,
    confidence_threshold: float,
    review_dir: Path | None,
) -> Dict[str, int]:
    model = load_detection_model(device)

    for label in OBJECT_LABELS:
        (out_dir / label).mkdir(parents=True, exist_ok=True)
    if review_dir:
        (review_dir / "objects").mkdir(parents=True, exist_ok=True)

    counts = {label: 0 for label in OBJECT_LABELS}
    counts["uncertain"] = 0
    log_rows = []

    batch_size = 8
    for start in tqdm(range(0, len(image_paths), batch_size), desc="Detecting objects"):
        batch_paths = image_paths[start : start + batch_size]
        batch_images = []
        valid_paths = []
        for p in batch_paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                batch_images.append(img)
                valid_paths.append(p)

        if not batch_images:
            continue

        results = detect_objects_batch(batch_images, model, device, confidence_threshold)

        for path, (label, score, boxes) in zip(valid_paths, results):
            log_rows.append({
                "image": path.name, "label": label,
                "confidence": f"{score:.4f}", "num_detections": len(boxes),
            })

            dest_dir = out_dir / label
            shutil.copy2(path, dest_dir / path.name)
            counts[label] += 1

    log_csv = out_dir / "object_sort_log.csv"
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label", "confidence", "num_detections"])
        writer.writeheader()
        writer.writerows(log_rows)

    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.task == "sort_signs":
        print("\n=== Manual arrow direction sorter ===")
        arrow_out = args.out_dir / "arrows"
        counts = manual_sort_signs(arrow_out)
        print(f"Sort results: {counts}")
        return

    if args.task == "sort_signs_auto":
        print("\n=== Auto arrow direction sorter ===")
        arrow_out = args.out_dir / "arrows"
        try:
            counts = auto_sort_signs(
                arrow_out,
                confidence_threshold=args.auto_confidence,
                min_examples_per_class=args.auto_min_examples,
                recovery_confidence_threshold=args.recover_none_confidence,
                review_dir=args.review_dir,
            )
        except RuntimeError as exc:
            print(f"Auto sorter could not run: {exc}")
            return
        print(f"Auto-sort results: {counts}")
        return

    device = resolve_device(args.device)
    image_dir = find_image_dir(args.data_dir)
    image_paths = collect_image_paths(image_dir, args.max_images)
    print(f"Found {len(image_paths)} images in {image_dir}")
    print(f"Device: {device}")

    if args.task in ("arrows", "all"):
        print("\n=== Phase 1: Detecting signs (colour) ===")
        arrow_out = args.out_dir / "arrows"
        counts = sort_arrows(
            image_paths, arrow_out,
            min_blue_area=args.min_blue_area,
            max_blue_area=args.max_blue_area,
            debug_dir=args.debug_dir,
        )
        print(f"Arrow results: {counts}")
        print(f"  Log: {arrow_out / 'arrow_sort_log.csv'}")

    if args.task in ("objects", "all"):
        print("\n=== Sorting objects ===")
        obj_out = args.out_dir / "objects"
        counts = sort_objects(
            image_paths, obj_out, device,
            confidence_threshold=args.object_confidence,
            review_dir=args.review_dir,
        )
        print(f"Object results: {counts}")
        print(f"  Log: {obj_out / 'object_sort_log.csv'}")

    print("\nDone! Review the sorted folders and logs before training.")
    if args.review_dir:
        print(f"Uncertain images saved to: {args.review_dir}")


if __name__ == "__main__":
    main()
