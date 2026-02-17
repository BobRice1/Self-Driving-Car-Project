from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union


IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")


@dataclass(frozen=True)
class DataPaths:
    train_csv: Path
    sample_submission_csv: Path
    train_images_dir: Path
    test_images_dir: Path


def _first_existing(candidates: Iterable[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    candidate_text = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Unable to locate {label}. Checked:\n{candidate_text}")


def resolve_data_paths(data_dir: Union[str, Path]) -> DataPaths:
    """Resolve CSV and image directories with fallbacks for local project layouts."""
    data_dir = Path(data_dir)
    cwd = Path.cwd()

    train_csv = _first_existing(
        [
            data_dir / "train.csv",
            data_dir / "data" / "train.csv",
            cwd / "train.csv",
            cwd / "data" / "train.csv",
        ],
        "train.csv",
    )

    sample_submission_csv = _first_existing(
        [
            data_dir / "sample_submission.csv",
            data_dir / "data" / "sample_submission.csv",
            cwd / "sample_submission.csv",
            cwd / "data" / "sample_submission.csv",
        ],
        "sample_submission.csv",
    )

    train_images_dir = _first_existing(
        [
            data_dir / "train_images",
            data_dir / "training_data",
            data_dir / "data" / "train_images",
            data_dir / "data" / "training_data",
            cwd / "data" / "train_images",
            cwd / "data" / "training_data",
        ],
        "train image directory",
    )

    test_images_dir = _first_existing(
        [
            data_dir / "test_images",
            data_dir / "test_data",
            data_dir / "data" / "test_images",
            data_dir / "data" / "test_data",
            cwd / "data" / "test_images",
            cwd / "data" / "test_data",
        ],
        "test image directory",
    )

    return DataPaths(
        train_csv=train_csv,
        sample_submission_csv=sample_submission_csv,
        train_images_dir=train_images_dir,
        test_images_dir=test_images_dir,
    )


def normalize_image_id(image_id: Union[int, float, str]) -> str:
    try:
        return str(int(image_id))
    except (TypeError, ValueError):
        return str(image_id)


def resolve_image_file(image_id: Union[int, float, str], image_dir: Path) -> Path:
    stem = normalize_image_id(image_id)
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Image not found for image_id={image_id} in {image_dir}. "
        f"Tried extensions: {IMAGE_EXTENSIONS}"
    )

