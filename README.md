# Kaggle Self-Driving Regression

PyTorch + timm pipeline for predicting:
- `angle` in `[0, 1]`
- `speed` in `[0, 1]`

Submission format:
- `image_id,angle,speed`

## Setup

1. Install PyTorch for your CUDA/CPU from:
- https://pytorch.org/get-started/locally/

2. Install project dependencies:
```bash
pip install -r requirements.txt
```

## Data Layout

The loader supports both folder conventions:
- `data/train_images`, `data/test_images`
- `data/training_data`, `data/test_data`

CSV files are resolved from:
- `data/train.csv` or project-root `train.csv`
- `data/sample_submission.csv` or project-root `sample_submission.csv`

## Model Architecture

Implemented in `src/model.py` as a multitask regressor:

1. Backbone:
- timm model (`tf_efficientnetv2_s` by default; configurable)
- global average pooled feature vector

2. Shared head:
- `Linear(num_features -> 256)`
- `GELU`
- `Dropout(0.2)`

3. Angle head:
- `Linear(256 -> 17)` logits
- softmax over 17 bins
- bins are fixed values `[0/16, 1/16, ..., 16/16]`
- final angle is expected value:
  - `angle_pred = sum_i p_i * bin_value_i`

4. Speed head:
- `Linear(256 -> 1)` logit
- `speed_pred = sigmoid(speed_logit)`

Loss in `src/losses.py`:
- `loss = 2.5 * mse_angle + 1.0 * mse_speed`
- optional angle sample weighting:
  - `w = 1 + alpha * |angle_true - 0.5|`

Training uses blocked contiguous K-fold CV (`src/cv.py`), with fold ensembling at inference.

## Training

Default:
```bash
python -m src.train --config configs/effnetv2s_320x240.yaml --kfold 5 --data_dir data --out_dir outputs
```

Key options:
- `--run_name <name>`
- `--seed 42`
- `--disable_amp`
- `--device auto|cuda|cpu`
- `--multi_gpu auto|off|dp`
- `--gpu_ids 0,1`

Examples:
- single GPU:
```bash
python -m src.train --config configs/effnetv2s_320x240.yaml --kfold 5 --data_dir data --out_dir outputs --device cuda --multi_gpu off --gpu_ids 0
```
- dual GPU (DataParallel):
```bash
python -m src.train --config configs/effnetv2s_320x240.yaml --kfold 5 --data_dir data --out_dir outputs --device cuda --multi_gpu dp --gpu_ids 0,1
```

## Inference

```bash
python -m src.infer --config configs/effnetv2s_320x240.yaml --kfold 5 --data_dir data --ckpt_dir outputs/checkpoints/<run_name> --out outputs/submissions/submission.csv
```

Multi-GPU inference example:
```bash
python -m src.infer --config configs/effnetv2s_320x240.yaml --kfold 5 --data_dir data --ckpt_dir outputs/checkpoints/<run_name> --out outputs/submissions/submission.csv --device cuda --multi_gpu dp --gpu_ids 0,1
```

Inference behavior:
- test order comes from `sample_submission.csv`
- loads `fold_0.pt ... fold_{k-1}.pt`
- averages fold predictions
- clamps `angle` and `speed` to `[0, 1]`

## Seed Sweep Automation

To train several seeds sequentially, rank them by mean best-fold `val_kaggle_mse`, run inference for the top seeds, and write blend CSVs in one unattended run:

```bash
python -m src.seed_sweep --config configs/ablation_loss_plus_select.yaml --seeds 42 1337 2026 31415 777 --kfold 5 --data_dir data --out_dir outputs --run_prefix ablation_loss_plus_select --top_n 3 --tta hflip
```

This writes:
- training logs under `outputs/logs`
- checkpoints under `outputs/checkpoints/<run_name>`
- top-seed submissions under `outputs/submissions`
- blend CSVs for the top 2 and top 3 inferred runs
- a summary JSON at `outputs/submissions/seed_sweep_summary.json`

Useful options:
- `--skip_train` to rank and infer existing runs only
- `--skip_infer` to train/rank without writing submissions
- `--device auto|cuda|cpu`
- `--multi_gpu auto|off|dp`
- `--gpu_ids 0,1`
- `--num_workers 4`
- `--batch_size 128`

## Outputs

- checkpoints:
  - `outputs/checkpoints/<run_name>/fold_<k>.pt`
- training log:
  - `outputs/logs/<run_name>_train_log.csv`
- invalid/corrupt training image report:
  - `outputs/logs/<run_name>_invalid_train_rows.csv`
- submission:
  - `outputs/submissions/submission.csv`

`speed` predictions are continuous probabilities in `[0,1]` by design.
