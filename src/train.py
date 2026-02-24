from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

IMPORT_ERROR = None
try:
    import cv2
    import pandas as pd
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from .cv import blocked_kfold_split
    from .dataset import DrivingDataset
    from .losses import weighted_mse_loss
    from .model import (
        MODEL_IMPLEMENTATION_VERSION,
        SelfDrivingRegressor,
        get_architecture_change_log,
    )
    from .paths import resolve_data_paths, resolve_image_file
    from .transforms import build_train_transforms, build_valid_transforms
    from .utils import (
        build_run_manifest,
        build_run_name,
        create_warmup_cosine_scheduler,
        ensure_dir,
        load_yaml,
        resolve_runtime_device,
        save_checkpoint,
        save_json,
        save_yaml,
        set_seed,
    )
except ModuleNotFoundError as exc:
    IMPORT_ERROR = exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train blocked K-fold models for Kaggle self-driving regression.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument("--kfold", type=int, default=5, help="Number of blocked folds.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Data directory.")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run name override.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_workers", type=int, default=None, help="Optional DataLoader worker override.")
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
        help="Multi-GPU mode. auto enables DataParallel when >=2 GPUs are selected.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated CUDA ids (e.g. 0,1). Default selects all visible GPUs.",
    )
    return parser.parse_args()


def build_loaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    image_dir: Path,
    config: Dict[str, Any],
    workers_override: int = None,
) -> Dict[str, DataLoader]:
    data_cfg = config["data"]
    height = int(data_cfg["height"])
    width = int(data_cfg["width"])
    num_workers = int(workers_override if workers_override is not None else data_cfg.get("num_workers", 4))

    train_ds = DrivingDataset(
        df=train_df,
        image_dir=image_dir,
        transform=build_train_transforms(height=height, width=width),
        is_train=True,
        hflip_p=float(data_cfg.get("hflip_p", 0.0)),
    )
    valid_ds = DrivingDataset(
        df=valid_df,
        image_dir=image_dir,
        transform=build_valid_transforms(height=height, width=width),
        is_train=False,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=int(data_cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(data_cfg.get("val_batch_size", data_cfg.get("batch_size", 32))),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return {"train": train_loader, "valid": valid_loader}


def get_amp_autocast(use_amp: bool):
    return torch.amp.autocast(device_type="cuda", enabled=use_amp)


def make_grad_scaler(use_amp: bool):
    return torch.amp.GradScaler("cuda", enabled=use_amp)


def filter_invalid_training_rows(df: pd.DataFrame, image_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep_mask = []
    invalid_rows = []

    for image_id in df["image_id"].tolist():
        reason = None
        file_path = None
        try:
            file_path = resolve_image_file(image_id, image_dir)
        except FileNotFoundError:
            reason = "missing_file"

        if reason is None and file_path is not None:
            if file_path.stat().st_size <= 0:
                reason = "empty_file"
            else:
                img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
                if img is None:
                    reason = "unreadable_file"

        is_valid = reason is None
        keep_mask.append(is_valid)
        if not is_valid:
            invalid_rows.append(
                {
                    "image_id": int(image_id),
                    "reason": reason,
                    "path": str(file_path) if file_path is not None else "",
                }
            )

    filtered_df = df.loc[keep_mask].reset_index(drop=True)
    invalid_df = pd.DataFrame(invalid_rows)
    return filtered_df, invalid_df


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler,
    device: torch.device,
    config: Dict[str, Any],
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    train_cfg = config["training"]

    running = {"loss": 0.0, "kaggle_mse": 0.0, "mse_angle": 0.0, "mse_speed": 0.0, "speed_bce": 0.0}
    seen = 0

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        angle_true = targets["angle"].to(device, non_blocking=True)
        speed_true = targets["speed"].to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)
        with get_amp_autocast(use_amp):
            outputs = model(images)
            loss_dict = weighted_mse_loss(
                angle_pred=outputs["angle_pred"],
                speed_pred=outputs["speed_pred"],
                speed_logit=outputs["speed_logit"],
                angle_true=angle_true,
                speed_true=speed_true,
                angle_loss_weight=float(train_cfg.get("angle_loss_weight", 1.0)),
                speed_loss_weight=float(train_cfg.get("speed_loss_weight", 1.0)),
                angle_sample_weight_alpha=float(train_cfg.get("angle_sample_weight_alpha", 0.0)),
                speed_loss_type=str(train_cfg.get("speed_loss_type", "mse")),
                speed_pos_weight=float(train_cfg.get("speed_pos_weight", 1.0)),
                speed_neg_weight=float(train_cfg.get("speed_neg_weight", 1.0)),
                speed_bce_weight=float(train_cfg.get("speed_bce_weight", 1.0)),
                speed_mse_weight=float(train_cfg.get("speed_mse_weight", 0.0)),
            )
            loss = loss_dict["loss"]

        scaler.scale(loss).backward()

        grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))
        if grad_clip_norm > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        seen += batch_size
        running["loss"] += float(loss_dict["loss"].detach().item()) * batch_size
        running["kaggle_mse"] += float(loss_dict["kaggle_mse"].detach().item()) * batch_size
        running["mse_angle"] += float(loss_dict["mse_angle"].detach().item()) * batch_size
        running["mse_speed"] += float(loss_dict["mse_speed"].detach().item()) * batch_size
        running["speed_bce"] += float(loss_dict["speed_bce"].detach().item()) * batch_size

    if seen == 0:
        return {"loss": 0.0, "kaggle_mse": 0.0, "mse_angle": 0.0, "mse_speed": 0.0, "speed_bce": 0.0}
    return {k: v / seen for k, v in running.items()}


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    train_cfg = config["training"]

    running = {"loss": 0.0, "kaggle_mse": 0.0, "mse_angle": 0.0, "mse_speed": 0.0, "speed_bce": 0.0}
    seen = 0

    for images, targets in tqdm(loader, desc="valid", leave=False):
        images = images.to(device, non_blocking=True)
        angle_true = targets["angle"].to(device, non_blocking=True)
        speed_true = targets["speed"].to(device, non_blocking=True)
        batch_size = images.size(0)

        with get_amp_autocast(use_amp):
            outputs = model(images)
            loss_dict = weighted_mse_loss(
                angle_pred=outputs["angle_pred"],
                speed_pred=outputs["speed_pred"],
                speed_logit=outputs["speed_logit"],
                angle_true=angle_true,
                speed_true=speed_true,
                angle_loss_weight=float(train_cfg.get("angle_loss_weight", 1.0)),
                speed_loss_weight=float(train_cfg.get("speed_loss_weight", 1.0)),
                angle_sample_weight_alpha=float(train_cfg.get("angle_sample_weight_alpha", 0.0)),
                speed_loss_type=str(train_cfg.get("speed_loss_type", "mse")),
                speed_pos_weight=float(train_cfg.get("speed_pos_weight", 1.0)),
                speed_neg_weight=float(train_cfg.get("speed_neg_weight", 1.0)),
                speed_bce_weight=float(train_cfg.get("speed_bce_weight", 1.0)),
                speed_mse_weight=float(train_cfg.get("speed_mse_weight", 0.0)),
            )

        seen += batch_size
        running["loss"] += float(loss_dict["loss"].detach().item()) * batch_size
        running["kaggle_mse"] += float(loss_dict["kaggle_mse"].detach().item()) * batch_size
        running["mse_angle"] += float(loss_dict["mse_angle"].detach().item()) * batch_size
        running["mse_speed"] += float(loss_dict["mse_speed"].detach().item()) * batch_size
        running["speed_bce"] += float(loss_dict["speed_bce"].detach().item()) * batch_size

    if seen == 0:
        return {"loss": 0.0, "kaggle_mse": 0.0, "mse_angle": 0.0, "mse_speed": 0.0, "speed_bce": 0.0}
    return {k: v / seen for k, v in running.items()}


def main() -> None:
    args = parse_args()
    if IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            f"Missing dependency for training: {IMPORT_ERROR}. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from IMPORT_ERROR

    config = load_yaml(args.config)

    run_name = build_run_name(config, args.run_name)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints" / run_name
    log_dir = out_dir / "logs"
    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)

    set_seed(args.seed)
    data_paths = resolve_data_paths(args.data_dir)
    df = pd.read_csv(data_paths.train_csv).sort_values("image_id").reset_index(drop=True)
    df, invalid_df = filter_invalid_training_rows(df, data_paths.train_images_dir)
    if not invalid_df.empty:
        invalid_path = log_dir / f"{run_name}_invalid_train_rows.csv"
        invalid_df.to_csv(invalid_path, index=False)
        print(
            f"Dropped {len(invalid_df)} invalid training rows before CV split. "
            f"Details: {invalid_path}"
        )

    folds = blocked_kfold_split(df, args.kfold)

    config_dump_path = ckpt_dir / "config_used.yaml"
    save_yaml(config, config_dump_path)
    run_manifest = build_run_manifest(
        run_name=run_name,
        config=config,
        implementation_version=MODEL_IMPLEMENTATION_VERSION,
        architecture_change_log=get_architecture_change_log(),
    )
    run_manifest_path = ckpt_dir / "run_manifest.json"
    save_json(run_manifest, run_manifest_path)
    save_json(run_manifest, log_dir / f"{run_name}_run_manifest.json")
    architecture_log_text = "\n".join(f"- {line}" for line in run_manifest["architecture_changes"])
    (ckpt_dir / "architecture_changes.txt").write_text(architecture_log_text + "\n", encoding="utf-8")
    (log_dir / f"{run_name}_architecture_changes.txt").write_text(architecture_log_text + "\n", encoding="utf-8")

    device, gpu_ids, use_data_parallel = resolve_runtime_device(
        device_mode=args.device,
        multi_gpu_mode=args.multi_gpu,
        gpu_ids_arg=args.gpu_ids,
    )
    train_cfg = config["training"]
    use_amp = bool(train_cfg.get("mixed_precision", True)) and (device.type == "cuda") and (not args.disable_amp)

    all_logs: List[Dict[str, Any]] = []
    print(f"Run: {run_name}")
    print(f"Implementation ID: {run_manifest['implementation_id']}")
    print(f"Implementation version: {run_manifest['implementation_version']}")
    print(f"Train CSV: {data_paths.train_csv}")
    print(f"Train images dir: {data_paths.train_images_dir}")
    print(f"Device: {device}, AMP: {use_amp}, multi_gpu={args.multi_gpu}, use_data_parallel={use_data_parallel}")
    if device.type == "cuda":
        gpu_names = [f"{gid}:{torch.cuda.get_device_name(gid)}" for gid in gpu_ids]
        print(f"Selected CUDA GPUs: {gpu_names}")
    if device.type != "cuda":
        print(
            "CUDA is not available in PyTorch. "
            f"torch.__version__={torch.__version__}, torch.version.cuda={torch.version.cuda}, "
            f"cuda_device_count={torch.cuda.device_count()}. "
            "Install a CUDA-enabled PyTorch build to train on GPU."
        )

    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        print(f"\n=== Fold {fold_idx + 1}/{args.kfold} ===")
        fold_train_df = df.iloc[train_idx].reset_index(drop=True)
        fold_valid_df = df.iloc[valid_idx].reset_index(drop=True)
        loaders = build_loaders(
            train_df=fold_train_df,
            valid_df=fold_valid_df,
            image_dir=data_paths.train_images_dir,
            config=config,
            workers_override=args.num_workers,
        )

        raw_model = SelfDrivingRegressor(
            backbone=str(config["model"].get("backbone", "tf_efficientnetv2_s")),
            pretrained=bool(config["model"].get("pretrained", True)),
            hidden_dim=int(config["model"].get("hidden_dim", 256)),
            dropout=float(config["model"].get("dropout", 0.2)),
        ).to(device)
        model = raw_model
        if use_data_parallel:
            model = torch.nn.DataParallel(raw_model, device_ids=gpu_ids, output_device=gpu_ids[0])

        optimizer = AdamW(
            model.parameters(),
            lr=float(train_cfg.get("lr", 3e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )

        epochs = int(train_cfg.get("epochs", 20))
        steps_per_epoch = max(1, len(loaders["train"]))
        total_steps = max(1, epochs * steps_per_epoch)
        warmup_steps = int(train_cfg.get("warmup_epochs", 1)) * steps_per_epoch
        scheduler = create_warmup_cosine_scheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )
        scaler = make_grad_scaler(use_amp)

        best_val = float("inf")
        patience = int(train_cfg.get("early_stopping_patience", 5))
        select_metric = str(train_cfg.get("model_selection_metric", "kaggle_mse")).strip().lower()
        valid_selection_metrics = {"loss", "kaggle_mse", "mse_angle", "mse_speed", "speed_bce"}
        if select_metric not in valid_selection_metrics:
            raise ValueError(
                f"Unsupported model_selection_metric='{select_metric}'. "
                f"Choose one of: {sorted(valid_selection_metrics)}"
            )
        bad_epochs = 0
        fold_ckpt_path = ckpt_dir / f"fold_{fold_idx}.pt"

        for epoch in range(1, epochs + 1):
            train_metrics = train_one_epoch(
                model=model,
                loader=loaders["train"],
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                config=config,
                use_amp=use_amp,
            )
            valid_metrics = validate_one_epoch(
                model=model,
                loader=loaders["valid"],
                device=device,
                config=config,
                use_amp=use_amp,
            )

            row = {
                "run_name": run_name,
                "fold": fold_idx,
                "epoch": epoch,
                "implementation_id": run_manifest["implementation_id"],
                "implementation_version": run_manifest["implementation_version"],
                "train_mse_total": train_metrics["loss"],
                "train_kaggle_mse": train_metrics["kaggle_mse"],
                "train_mse_angle": train_metrics["mse_angle"],
                "train_mse_speed": train_metrics["mse_speed"],
                "train_speed_bce": train_metrics["speed_bce"],
                "val_mse_total": valid_metrics["loss"],
                "val_kaggle_mse": valid_metrics["kaggle_mse"],
                "val_mse_angle": valid_metrics["mse_angle"],
                "val_mse_speed": valid_metrics["mse_speed"],
                "val_speed_bce": valid_metrics["speed_bce"],
                "lr": optimizer.param_groups[0]["lr"],
                "selection_metric": select_metric,
                "selection_value": valid_metrics[select_metric],
            }
            all_logs.append(row)

            print(
                "epoch={epoch:03d} "
                "train_loss={train_total:.6f} val_loss={val_total:.6f} "
                "val_kaggle={val_kaggle:.6f} val_angle={val_angle:.6f} val_speed={val_speed:.6f}".format(
                    epoch=epoch,
                    train_total=train_metrics["loss"],
                    val_total=valid_metrics["loss"],
                    val_kaggle=valid_metrics["kaggle_mse"],
                    val_angle=valid_metrics["mse_angle"],
                    val_speed=valid_metrics["mse_speed"],
                )
            )

            current_val = valid_metrics[select_metric]
            if current_val < best_val:
                best_val = current_val
                bad_epochs = 0
                save_checkpoint(
                    path=fold_ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_score=best_val,
                    fold=fold_idx,
                    config=config,
                    run_manifest=run_manifest,
                )
                print(f"Saved best checkpoint ({select_metric}={best_val:.6f}): {fold_ckpt_path}")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch} (patience {patience}).")
                    break

    log_path = log_dir / f"{run_name}_train_log.csv"
    pd.DataFrame(all_logs).to_csv(log_path, index=False)
    print(f"\nTraining complete. Logs: {log_path}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == "__main__":
    main()
