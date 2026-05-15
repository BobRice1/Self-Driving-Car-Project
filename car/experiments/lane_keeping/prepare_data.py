from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
ANGLE_MIN = 50.0
ANGLE_MAX = 120.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare chunk-based lane keeping splits.")
    parser.add_argument("--csv-data", type=Path, default=Path("data"))
    parser.add_argument("--drive-frames", type=Path, default=Path("car/data/drive_frames"))
    parser.add_argument("--out-dir", type=Path, default=Path("car/data/lane_keeping/splits"))
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--min-steering", type=float, default=ANGLE_MIN)
    parser.add_argument("--max-steering", type=float, default=ANGLE_MAX)
    parser.add_argument("--skip-image-check", action="store_true")
    return parser.parse_args()


def csv_angle_to_car(angle: float, min_angle: float, max_angle: float) -> float:
    angle = float(angle)
    if 0.0 <= angle <= 1.0:
        return min_angle + angle * (max_angle - min_angle)
    return angle


def find_image(stem: str, image_dir: Path) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_csv_data(csv_data: Path, min_angle: float, max_angle: float) -> pd.DataFrame:
    csv_path = csv_data / "train.csv"
    image_dir = csv_data / "training_data"
    if not csv_path.exists() or not image_dir.exists():
        return pd.DataFrame()

    rows = []
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        image_path = find_image(str(int(row["image_id"])), image_dir)
        if image_path is None:
            continue
        rows.append(
            {
                "image_path": str(image_path),
                "steering": csv_angle_to_car(row["angle"], min_angle, max_angle),
                "speed": float(row.get("speed", 0.0)),
                "source": "csv",
                "sequence_key": f"csv_{int(row['image_id']):09d}",
            }
        )
    return pd.DataFrame(rows)


def load_filename_data(drive_frames: Path) -> pd.DataFrame:
    if not drive_frames.exists():
        return pd.DataFrame()

    pattern = re.compile(r"^(.+)_([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)$")
    rows = []
    for path in sorted(drive_frames.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        match = pattern.match(path.stem)
        if not match:
            continue
        rows.append(
            {
                "image_path": str(path),
                "steering": float(match.group(2)),
                "speed": float(match.group(3)),
                "source": "drive_frames",
                "sequence_key": f"drive_{match.group(1)}",
            }
        )
    return pd.DataFrame(rows)


def image_is_valid(path: str) -> bool:
    try:
        fp = Path(path)
        if fp.stat().st_size <= 0:
            return False
        return cv2.imread(str(fp), cv2.IMREAD_COLOR) is not None
    except OSError:
        return False


def add_split_columns(df: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    df = df.sort_values(["source", "sequence_key"]).reset_index(drop=True)
    df["chunk_id"] = np.arange(len(df)) // max(1, int(chunk_size))
    chunks = df["chunk_id"].unique()
    test_chunks = set(chunks[::6])
    val_chunks = set(chunks[3::6])

    def split_for_chunk(chunk_id: int) -> str:
        if chunk_id in test_chunks:
            return "test"
        if chunk_id in val_chunks:
            return "val"
        return "train"

    df["split"] = df["chunk_id"].map(split_for_chunk)
    return df


def write_split(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    cols = ["image_path", "steering", "speed", "source", "sequence_key", "chunk_id"]
    df.loc[df["split"] == split, cols].to_csv(out_dir / f"{split}.csv", index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_df = load_csv_data(args.csv_data, args.min_steering, args.max_steering)
    drive_df = load_filename_data(args.drive_frames)
    df = pd.concat([csv_df, drive_df], ignore_index=True)
    if df.empty:
        raise SystemExit("No training rows found. Check --csv-data and --drive-frames.")

    before = len(df)
    df["steering"] = df["steering"].clip(args.min_steering, args.max_steering)
    if not args.skip_image_check:
        df = df[df["image_path"].map(image_is_valid)].reset_index(drop=True)
    dropped = before - len(df)

    df = add_split_columns(df, args.chunk_size)
    write_split(df, out_dir, "train")
    write_split(df, out_dir, "val")
    write_split(df, out_dir, "test")
    bend_cols = ["image_path", "steering", "speed", "source", "sequence_key", "chunk_id"]
    bend_df = df[df["source"] == "drive_frames"].copy()
    bend_df["source"] = "manual_bend"
    bend_df[bend_cols].to_csv(out_dir / "val_bend.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    summary = (
        df.groupby(["split", "source"])
        .agg(rows=("image_path", "count"), steering_mean=("steering", "mean"), steering_min=("steering", "min"), steering_max=("steering", "max"))
        .reset_index()
    )
    summary.to_csv(out_dir / "dataset_summary.csv", index=False)
    print(f"Prepared {len(df)} rows in {out_dir} (dropped_invalid={dropped}).")
    print(f"Bend/oval validation rows: {len(bend_df)} -> {out_dir / 'val_bend.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
