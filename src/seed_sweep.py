from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

IMPORT_ERROR = None
try:
    import pandas as pd

    from .utils import ensure_dir, load_yaml, save_json
except ModuleNotFoundError as exc:
    IMPORT_ERROR = exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a sequential seed sweep: train multiple seeds, rank by mean best-fold "
            "val_kaggle_mse, infer the top runs, and write blend CSVs."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="Seeds to train sequentially.")
    parser.add_argument("--kfold", type=int, default=5, help="Number of folds/checkpoints.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Data directory.")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs"), help="Output directory.")
    parser.add_argument(
        "--run_prefix",
        type=str,
        default=None,
        help="Run name prefix. Final run names become <run_prefix>_seed<seed>.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=3,
        help="Number of top-ranked runs to infer and consider for blends.",
    )
    parser.add_argument(
        "--tta",
        type=str,
        default="hflip",
        choices=["none", "hflip"],
        help="Inference-time augmentation mode for generated submissions.",
    )
    parser.add_argument(
        "--blend_sizes",
        type=int,
        nargs="*",
        default=[2, 3],
        help="Blend sizes to write from the top ranked inferred runs.",
    )
    parser.add_argument("--num_workers", type=int, default=None, help="Optional DataLoader worker override.")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional inference batch size override.")
    parser.add_argument("--disable_amp", action="store_true", help="Disable mixed precision for train and infer.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device mode forwarded to train and infer.",
    )
    parser.add_argument(
        "--multi_gpu",
        type=str,
        default="auto",
        choices=["auto", "off", "dp"],
        help="Multi-GPU mode forwarded to train and infer.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated CUDA ids forwarded to train and infer.",
    )
    parser.add_argument("--skip_train", action="store_true", help="Skip training and rank existing runs only.")
    parser.add_argument("--skip_infer", action="store_true", help="Skip inference and blending.")
    parser.add_argument(
        "--summary_name",
        type=str,
        default="seed_sweep_summary.json",
        help="Summary filename written under outputs/submissions.",
    )
    return parser.parse_args()


def build_run_names(run_prefix: str, seeds: Sequence[int]) -> List[str]:
    return [f"{run_prefix}_seed{seed}" for seed in seeds]


def append_common_runtime_args(command: List[str], args: argparse.Namespace, *, include_batch_size: bool) -> None:
    if args.num_workers is not None:
        command.extend(["--num_workers", str(args.num_workers)])
    if include_batch_size and args.batch_size is not None:
        command.extend(["--batch_size", str(args.batch_size)])
    if args.disable_amp:
        command.append("--disable_amp")
    command.extend(["--device", args.device, "--multi_gpu", args.multi_gpu])
    if args.gpu_ids:
        command.extend(["--gpu_ids", args.gpu_ids])


def run_command(command: Sequence[str], cwd: Path) -> None:
    printable = " ".join(command)
    print(f"\n>>> {printable}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def train_run(run_name: str, seed: int, args: argparse.Namespace, repo_root: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "src.train",
        "--config",
        str(args.config),
        "--kfold",
        str(args.kfold),
        "--data_dir",
        str(args.data_dir),
        "--out_dir",
        str(args.out_dir),
        "--run_name",
        run_name,
        "--seed",
        str(seed),
    ]
    append_common_runtime_args(command, args, include_batch_size=False)
    run_command(command, cwd=repo_root)


def summarize_run(log_path: Path, run_name: str) -> Dict[str, object]:
    if not log_path.exists():
        raise FileNotFoundError(f"Missing training log for run '{run_name}': {log_path}")

    df = pd.read_csv(log_path)
    if df.empty:
        raise ValueError(f"Training log is empty for run '{run_name}': {log_path}")

    fold_best = df.groupby("fold")["val_kaggle_mse"].min().sort_index()
    best_epoch_rows = df.loc[df.groupby("fold")["val_kaggle_mse"].idxmin()].sort_values("fold")
    return {
        "run_name": run_name,
        "log_path": str(log_path),
        "mean_best_fold_val_kaggle_mse": float(fold_best.mean()),
        "fold_best_val_kaggle_mse": {str(int(k)): float(v) for k, v in fold_best.items()},
        "fold_best_epoch": {
            str(int(row["fold"])): int(row["epoch"])
            for _, row in best_epoch_rows[["fold", "epoch"]].iterrows()
        },
    }


def rank_runs(run_names: Sequence[str], out_dir: Path) -> List[Dict[str, object]]:
    logs_dir = out_dir / "logs"
    summaries = [
        summarize_run(log_path=logs_dir / f"{run_name}_train_log.csv", run_name=run_name)
        for run_name in run_names
    ]
    return sorted(summaries, key=lambda item: float(item["mean_best_fold_val_kaggle_mse"]))


def infer_run(run_name: str, args: argparse.Namespace, repo_root: Path) -> Path:
    submissions_dir = args.out_dir / "submissions"
    ensure_dir(submissions_dir)
    out_path = submissions_dir / f"{run_name}_tta_{args.tta}.csv"
    command = [
        sys.executable,
        "-m",
        "src.infer",
        "--config",
        str(args.config),
        "--kfold",
        str(args.kfold),
        "--data_dir",
        str(args.data_dir),
        "--ckpt_dir",
        str(args.out_dir / "checkpoints" / run_name),
        "--out",
        str(out_path),
        "--tta",
        args.tta,
    ]
    append_common_runtime_args(command, args, include_batch_size=True)
    run_command(command, cwd=repo_root)
    return out_path


def blend_submissions(input_paths: Sequence[Path], out_path: Path) -> None:
    if not input_paths:
        raise ValueError("No submission CSVs were provided for blending.")

    frames = [pd.read_csv(path) for path in input_paths]
    base = frames[0].copy()

    for frame in frames[1:]:
        if not base["image_id"].equals(frame["image_id"]):
            raise ValueError("Cannot blend submissions with mismatched image_id order.")

    base["angle"] = sum(frame["angle"] for frame in frames) / float(len(frames))
    base["speed"] = sum(frame["speed"] for frame in frames) / float(len(frames))
    base.to_csv(out_path, index=False)


def choose_blend_sizes(blend_sizes: Iterable[int], num_available: int) -> List[int]:
    cleaned = []
    for size in blend_sizes:
        if size < 2 or size > num_available:
            continue
        if size not in cleaned:
            cleaned.append(size)
    return cleaned


def main() -> None:
    args = parse_args()
    if IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            f"Missing dependency for seed sweep: {IMPORT_ERROR}. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from IMPORT_ERROR

    repo_root = Path(__file__).resolve().parent.parent
    args.config = args.config.resolve()
    args.data_dir = args.data_dir.resolve()
    args.out_dir = args.out_dir.resolve()

    config = load_yaml(args.config)
    run_prefix = args.run_prefix or str(config.get("experiment", {}).get("run_name", "seed_sweep"))
    run_names = build_run_names(run_prefix=run_prefix, seeds=args.seeds)

    print("Seed sweep plan:", flush=True)
    for seed, run_name in zip(args.seeds, run_names):
        print(f"- seed={seed} -> run_name={run_name}", flush=True)

    if not args.skip_train:
        for seed, run_name in zip(args.seeds, run_names):
            train_run(run_name=run_name, seed=seed, args=args, repo_root=repo_root)

    ranked = rank_runs(run_names=run_names, out_dir=args.out_dir)
    top_runs = ranked[: max(1, min(args.top_n, len(ranked)))]

    print("\nRanked runs by mean best-fold val_kaggle_mse:", flush=True)
    for idx, item in enumerate(ranked, start=1):
        print(
            f"{idx}. {item['run_name']} -> {float(item['mean_best_fold_val_kaggle_mse']):.6f}",
            flush=True,
        )

    summary: Dict[str, object] = {
        "config": str(args.config),
        "seeds": [int(seed) for seed in args.seeds],
        "run_prefix": run_prefix,
        "tta": args.tta,
        "ranked_runs": ranked,
        "top_runs": [item["run_name"] for item in top_runs],
        "submission_csvs": {},
        "blend_csvs": {},
    }

    if args.skip_infer:
        summary_path = args.out_dir / "submissions" / args.summary_name
        ensure_dir(summary_path.parent)
        save_json(summary, summary_path)
        print(f"\nSummary written: {summary_path}", flush=True)
        return

    inferred_paths: Dict[str, Path] = {}
    for item in top_runs:
        run_name = str(item["run_name"])
        inferred_paths[run_name] = infer_run(run_name=run_name, args=args, repo_root=repo_root)

    summary["submission_csvs"] = {run_name: str(path) for run_name, path in inferred_paths.items()}

    blend_outputs: Dict[str, str] = {}
    submissions_dir = args.out_dir / "submissions"
    selected_run_names = [str(item["run_name"]) for item in top_runs]
    for blend_size in choose_blend_sizes(args.blend_sizes, num_available=len(selected_run_names)):
        blend_run_names = selected_run_names[:blend_size]
        blend_name = f"blend_top{blend_size}_{run_prefix}_tta_{args.tta}.csv"
        blend_path = submissions_dir / blend_name
        blend_submissions([inferred_paths[name] for name in blend_run_names], blend_path)
        blend_outputs[",".join(blend_run_names)] = str(blend_path)
        print(f"Blend written ({blend_size}-way): {blend_path}", flush=True)

    summary["blend_csvs"] = blend_outputs

    summary_path = submissions_dir / args.summary_name
    save_json(summary, summary_path)
    print(f"\nSummary written: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
