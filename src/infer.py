from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

IMPORT_ERROR = None
try:
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from .dataset import DrivingDataset
    from .model import MODEL_IMPLEMENTATION_VERSION, SelfDrivingRegressor, get_architecture_change_log
    from .paths import resolve_data_paths
    from .transforms import build_valid_transforms
    from .utils import (
        ensure_dir,
        load_model_state_dict_flexible,
        load_yaml,
        resolve_runtime_device,
        save_json,
    )
except ModuleNotFoundError as exc:
    IMPORT_ERROR = exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fold-ensemble inference for Kaggle self-driving regression.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument("--kfold", type=int, default=5, help="Number of folds/checkpoints.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Data directory.")
    parser.add_argument("--ckpt_dir", type=Path, required=True, help="Directory containing fold checkpoints.")
    parser.add_argument("--out", type=Path, required=True, help="Output submission CSV path.")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--num_workers", type=int, default=None, help="Optional DataLoader worker override.")
    parser.add_argument(
        "--tta",
        type=str,
        default="hflip",
        choices=["none", "hflip"],
        help="Inference-time augmentation mode.",
    )
    parser.add_argument("--disable_amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device mode. auto uses CUDA if available.",
    )
    parser.add_argument(
        "--multi_gpu",
        type=str,
        default="auto",
        choices=["auto", "off", "dp"],
        help="Multi-GPU mode for inference. auto enables DataParallel when >=2 GPUs are selected.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated CUDA ids (e.g. 0,1). Default selects all visible GPUs.",
    )
    return parser.parse_args()


def predict_fold(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    tta_mode: str,
) -> Dict[str, np.ndarray]:
    model.eval()
    angle_pred = []
    speed_pred = []

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="infer", leave=False):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images)
                angle = outputs["angle_pred"]
                speed = outputs["speed_pred"]

                if tta_mode == "hflip":
                    flipped_images = torch.flip(images, dims=[3])
                    flipped_outputs = model(flipped_images)
                    angle_flipped = 1.0 - flipped_outputs["angle_pred"]
                    speed_flipped = flipped_outputs["speed_pred"]
                    angle = 0.5 * (angle + angle_flipped)
                    speed = 0.5 * (speed + speed_flipped)

            angle_pred.append(angle.detach().cpu().numpy())
            speed_pred.append(speed.detach().cpu().numpy())

    return {
        "angle": np.concatenate(angle_pred).astype(np.float32),
        "speed": np.concatenate(speed_pred).astype(np.float32),
    }


def build_submission_manifest(
    out_path: Path,
    config: Dict[str, Any],
    ckpt_dir: Path,
    kfold: int,
    tta_mode: str,
    run_manifest: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    config_for_manifest = dict(config)
    return {
        "created_for_submission": out_path.name,
        "submission_path": str(out_path),
        "checkpoint_dir": str(ckpt_dir),
        "kfold": int(kfold),
        "tta_mode": tta_mode,
        "implementation_version": (
            run_manifest.get("implementation_version")
            if run_manifest is not None
            else MODEL_IMPLEMENTATION_VERSION
        ),
        "implementation_id": run_manifest.get("implementation_id") if run_manifest is not None else "unknown",
        "architecture_changes": (
            run_manifest.get("architecture_changes")
            if run_manifest is not None
            else get_architecture_change_log()
        ),
        "run_manifest_from_checkpoint": run_manifest is not None,
        "config": config_for_manifest,
    }


def main() -> None:
    args = parse_args()
    if IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            f"Missing dependency for inference: {IMPORT_ERROR}. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from IMPORT_ERROR

    config = load_yaml(args.config)
    data_paths = resolve_data_paths(args.data_dir)

    sample = pd.read_csv(data_paths.sample_submission_csv)
    test_df = sample[["image_id"]].copy()

    data_cfg = config["data"]
    height = int(data_cfg["height"])
    width = int(data_cfg["width"])
    num_workers = int(args.num_workers if args.num_workers is not None else data_cfg.get("num_workers", 4))

    test_ds = DrivingDataset(
        df=test_df,
        image_dir=data_paths.test_images_dir,
        transform=build_valid_transforms(height=height, width=width),
        is_train=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size if args.batch_size is not None else data_cfg.get("val_batch_size", 64)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device, gpu_ids, use_data_parallel = resolve_runtime_device(
        device_mode=args.device,
        multi_gpu_mode=args.multi_gpu,
        gpu_ids_arg=args.gpu_ids,
    )
    use_amp = bool(config.get("training", {}).get("mixed_precision", True)) and (device.type == "cuda") and (not args.disable_amp)

    pred_angle = np.zeros(len(test_ds), dtype=np.float32)
    pred_speed = np.zeros(len(test_ds), dtype=np.float32)
    run_manifest = None

    print(f"Sample CSV: {data_paths.sample_submission_csv}")
    print(f"Test images dir: {data_paths.test_images_dir}")
    print(
        f"Device: {device}, AMP: {use_amp}, TTA: {args.tta}, "
        f"multi_gpu={args.multi_gpu}, use_data_parallel={use_data_parallel}"
    )
    if device.type == "cuda":
        gpu_names = [f"{gid}:{torch.cuda.get_device_name(gid)}" for gid in gpu_ids]
        print(f"Selected CUDA GPUs: {gpu_names}")
    if device.type != "cuda":
        print(
            "CUDA is not available in PyTorch. "
            f"torch.__version__={torch.__version__}, torch.version.cuda={torch.version.cuda}, "
            f"cuda_device_count={torch.cuda.device_count()}."
        )

    for fold in range(args.kfold):
        ckpt_path = args.ckpt_dir / f"fold_{fold}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint for fold {fold}: {ckpt_path}")

        raw_model = SelfDrivingRegressor(
            backbone=str(config["model"].get("backbone", "tf_efficientnetv2_s")),
            pretrained=False,
            hidden_dim=int(config["model"].get("hidden_dim", 256)),
            dropout=float(config["model"].get("dropout", 0.2)),
        ).to(device)

        checkpoint = torch.load(ckpt_path, map_location=device)
        if run_manifest is None and isinstance(checkpoint, dict):
            run_manifest = checkpoint.get("run_manifest")
        load_model_state_dict_flexible(raw_model, checkpoint["model_state_dict"])
        model = raw_model
        if use_data_parallel:
            model = torch.nn.DataParallel(raw_model, device_ids=gpu_ids, output_device=gpu_ids[0])

        print(f"Loaded fold {fold}: {ckpt_path}")
        fold_preds = predict_fold(
            model=model,
            loader=test_loader,
            device=device,
            use_amp=use_amp,
            tta_mode=args.tta,
        )
        pred_angle += fold_preds["angle"]
        pred_speed += fold_preds["speed"]

    pred_angle /= float(args.kfold)
    pred_speed /= float(args.kfold)

    submission = sample.copy()
    submission["angle"] = np.clip(pred_angle, 0.0, 1.0)
    submission["speed"] = np.clip(pred_speed, 0.0, 1.0)

    ensure_dir(args.out.parent)
    submission.to_csv(args.out, index=False)
    print(f"Submission written: {args.out}")

    submission_manifest = build_submission_manifest(
        out_path=args.out,
        config=config,
        ckpt_dir=args.ckpt_dir,
        kfold=args.kfold,
        tta_mode=args.tta,
        run_manifest=run_manifest,
    )
    manifest_path = args.out.parent / f"{args.out.stem}_manifest.json"
    save_json(submission_manifest, manifest_path)
    print(f"Submission manifest: {manifest_path}")


if __name__ == "__main__":
    main()
