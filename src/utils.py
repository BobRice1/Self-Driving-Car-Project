import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> LambdaLR:
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_run_name(config: Dict[str, Any], override: Optional[str] = None) -> str:
    if override:
        return override
    base = config.get("experiment", {}).get("run_name", "run")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}"


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_score: float,
    fold: int,
    config: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "epoch": epoch,
            "best_score": best_score,
            "fold": fold,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
        },
        path,
    )


def load_model_state_dict_flexible(model: torch.nn.Module, state_dict: Dict[str, Any]) -> None:
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        pass

    if all(k.startswith("module.") for k in state_dict.keys()):
        stripped = {k[len("module."):]: v for k, v in state_dict.items()}
        model.load_state_dict(stripped, strict=True)
        return

    prefixed = {f"module.{k}": v for k, v in state_dict.items()}
    model.load_state_dict(prefixed, strict=True)


def parse_gpu_ids(gpu_ids_arg: Optional[str], cuda_device_count: int) -> List[int]:
    if cuda_device_count <= 0:
        return []

    if gpu_ids_arg is None or gpu_ids_arg.strip() == "":
        return list(range(cuda_device_count))

    parsed: List[int] = []
    for token in gpu_ids_arg.split(","):
        token = token.strip()
        if not token:
            continue
        gpu_id = int(token)
        if gpu_id < 0 or gpu_id >= cuda_device_count:
            raise ValueError(
                f"Invalid gpu id '{gpu_id}'. Detected {cuda_device_count} CUDA device(s): "
                f"valid range is [0, {cuda_device_count - 1}]"
            )
        if gpu_id not in parsed:
            parsed.append(gpu_id)

    if not parsed:
        raise ValueError("No valid GPU ids were parsed from --gpu_ids.")
    return parsed


def resolve_runtime_device(
    device_mode: str = "auto",
    multi_gpu_mode: str = "auto",
    gpu_ids_arg: Optional[str] = None,
) -> Tuple[torch.device, List[int], bool]:
    device_mode = device_mode.lower()
    multi_gpu_mode = multi_gpu_mode.lower()

    if device_mode not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Unknown device mode: {device_mode}")
    if multi_gpu_mode not in {"auto", "off", "dp"}:
        raise ValueError(f"Unknown multi_gpu mode: {multi_gpu_mode}")

    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count()

    if device_mode == "cpu":
        if multi_gpu_mode == "dp":
            raise RuntimeError("Cannot use --multi_gpu dp with --device cpu.")
        return torch.device("cpu"), [], False

    if not cuda_available:
        if device_mode == "cuda":
            raise RuntimeError(
                "CUDA requested but not available in PyTorch. "
                f"torch.__version__={torch.__version__}, torch.version.cuda={torch.version.cuda}, "
                f"cuda_device_count={cuda_device_count}"
            )
        if multi_gpu_mode == "dp":
            raise RuntimeError("Cannot use --multi_gpu dp because CUDA is unavailable.")
        return torch.device("cpu"), [], False

    gpu_ids = parse_gpu_ids(gpu_ids_arg, cuda_device_count)
    if not gpu_ids:
        raise RuntimeError("CUDA is available but no GPU ids were selected.")

    use_data_parallel = False
    if multi_gpu_mode == "dp":
        if len(gpu_ids) < 2:
            raise RuntimeError(
                f"--multi_gpu dp requires at least 2 GPUs, but gpu_ids={gpu_ids}."
            )
        use_data_parallel = True
    elif multi_gpu_mode == "auto":
        use_data_parallel = len(gpu_ids) >= 2

    torch.cuda.set_device(gpu_ids[0])
    return torch.device(f"cuda:{gpu_ids[0]}"), gpu_ids, use_data_parallel
