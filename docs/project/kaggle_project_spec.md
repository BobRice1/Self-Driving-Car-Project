# Kaggle Self‑Driving Regression — Project Specification (Windows‑compatible)

This document is **the project brief + implementation context** you can paste into **Codex** to start building the Kaggle training/inference pipeline.

## Goal

Train a computer-vision model to predict two **normalised** continuous targets from images:

- `angle` in **[0, 1]** (observed to be **discretised into 17 levels**: 0/16, 1/16, …, 16/16 in the provided `train.csv`)
- `speed` in **[0, 1]** (observed to be **binary**: 0 or 1 in `train.csv`)

Kaggle evaluation metric: **Mean Squared Error (MSE)** across the two columns (public and private leaderboards each score on 50% of test data).

**Constraint:** Must run on **Windows** (development) and be easy to run on a single GPU (e.g., 2080 Ti).  
**Note:** Kaggle model does **not** need to be lightweight for edge deployment.

---

## Why this approach

Because `angle` takes only **17 discrete values**, a strong approach is:

- Predict a **17‑class distribution** over angle bins (softmax), then convert to a continuous value via **expected value**.
- Predict `speed` as a probability in [0,1] via sigmoid.

This stabilises training, reduces “big misses” on turns, and still outputs continuous values compatible with Kaggle’s MSE.

---

## Files and data contract

### Input files (provided by user)
- `train.csv` (labelled): columns `image_id, angle, speed`
- `sample_submission.csv`: contains `image_id` and placeholder `angle, speed` (defines required test order)
- Images:
  - Train images in `data/training_data/`
  - Test images in `data/test_data/`

### Image naming
Assume test/train image files are named by `image_id` with common extensions. Implement robust loading:
- Prefer `.png`, else try `.jpb`, else `.jpeg`.

### Output files
- `outputs/submissions/submission.csv` with columns exactly:
  - `image_id, angle, speed`
- The `image_id` order **must match `sample_submission.csv` exactly** (do not rely on filesystem ordering).

---

## Windows‑compatible tech stack

### Recommended framework: **PyTorch + timm**
Reasons:
- Fast iteration and easy ensembling on GPU
- Strong pretrained backbones (EfficientNetV2, ConvNeXt)
- Works well on Windows

References:
- Install PyTorch for Windows via the official selector. (https://pytorch.org/get-started/locally/) citeturn0search4
- `timm` installation is `pip install timm`. (https://timm.fast.ai/) citeturn0search1
- Albumentations is available via pip and supports modern Python. (https://pypi.org/project/albumentations/) citeturn0search12

### Python version
Use **Python 3.10 or 3.11** (recommended for current PyTorch + libraries). Avoid very new Python versions if PyTorch wheels lag behind.

### Environment
Prefer a virtual environment:
- `python -m venv .venv`
- `.\.venv\Scripts\activate`

---

## Repository layout (create exactly)

```
kaggle-selfdriving/
  README.md
  requirements.txt
  configs/
    effnetv2s_320x240.yaml
    effnetv2s_384x288.yaml
    convnext_tiny_320x240.yaml
  data/
    train.csv
    sample_submission.csv
    train_images/
    test_images/
  src/
    __init__.py
    paths.py
    dataset.py
    transforms.py
    model.py
    losses.py
    cv.py
    train.py
    infer.py
    utils.py
  outputs/
    logs/
    checkpoints/
    submissions/
```

---

## Core modelling specification

### Angle bins
Define fixed bin values:
- `bin_values = [0/16, 1/16, ..., 16/16]` (float32)

Angle head:
- Output `angle_logits` shape `[B, 17]`
- Convert to distribution: `p = softmax(angle_logits)`
- Convert to continuous prediction:
  - `angle_pred = sum(p_i * bin_values_i)` for each sample

### Speed head
Speed head:
- Output `speed_logit` shape `[B, 1]`
- `speed_pred = sigmoid(speed_logit)` (float in [0,1])

### Loss function (matches Kaggle metric)
Compute:
- `mse_angle = mean((angle_pred - angle_true)^2)`
- `mse_speed = mean((speed_pred - speed_true)^2)`

Weighted total:
- `loss = 2.5*mse_angle + 1.0*mse_speed`

Optional (recommended) sample weighting for angle to address imbalance:
- `w = 1 + alpha * abs(angle_true - 0.5)` with `alpha` in [2, 4]
- Apply weights only to angle MSE term.

---

## Cross‑validation (private‑LB safe)

Implement **blocked K‑fold**:
- Sort `train.csv` by `image_id` ascending
- Split into K contiguous folds (K=5 default)
- Train K models; validate on the held-out fold
- Save best checkpoint per fold by **val weighted MSE**

Test inference uses **fold ensemble**:
- Average predictions across all K fold models

This reduces overfitting to sequential correlation and is safer for private leaderboard.

---

## Augmentations (moderate, realistic)

Use Albumentations:

Train augmentations:
- RandomBrightnessContrast (moderate)
- RandomGamma
- MotionBlur or GaussianBlur (low probability)
- GaussNoise (low probability)
- ImageCompression (low probability)
- Affine / ShiftScaleRotate (very mild)

Optional flip:
- HorizontalFlip (p=0.2)
- If flip, adjust angle bins: bin index `i -> 16 - i` (equivalently, angle value transforms `angle -> 1 - angle`)

Validation augmentations:
- Only resize + normalisation (no stochastic aug)

---

## Backbones and configs

### Primary model (strong baseline)
- `tf_efficientnetv2_s` (timm name)
- Image size: start 320×240; also try 384×288

### Diversity model (optional for stronger ensemble)
- `convnext_tiny` (timm)

Head:
- Global pooling from backbone features
- MLP: Linear(feats -> 256) + GELU/ReLU + Dropout(0.2)
- Then:
  - angle_logits: Linear(256 -> 17)
  - speed_logit: Linear(256 -> 1)

---

## Training loop requirements

- Mixed precision (torch.cuda.amp) if CUDA available
- Optimiser: AdamW
- Scheduler: cosine annealing with warmup OR OneCycleLR
- Early stopping on validation weighted MSE
- Deterministic seeds for reproducibility
- Save:
  - `outputs/checkpoints/<run_name>/fold_<k>.pt`
  - logs to `outputs/logs/`

Metrics to log:
- `val_mse_angle`, `val_mse_speed`, `val_mse_total` (weighted)

---

## Inference requirements

- Load all K fold checkpoints
- Iterate test ids from `sample_submission.csv['image_id']`
- Predict per model, average across folds
- Clamp final predictions to [0, 1]
- Write submission CSV exactly like sample

---

## CLI contract (Codex should implement)

### Train
```
python -m src.train --config configs/effnetv2s_320x240.yaml --kfold 5 --data_dir data --out_dir outputs
```

### Infer (make submission)
```
python -m src.infer --config configs/effnetv2s_320x240.yaml --kfold 5 --data_dir data --ckpt_dir outputs/checkpoints/<run_name> --out outputs/submissions/submission.csv
```

---

## requirements.txt (Codex should generate)

Minimum:
- torch, torchvision (installed via official PyTorch command for Windows) citeturn0search4
- timm citeturn0search1
- albumentations citeturn0search12
- opencv-python
- pandas
- numpy
- pyyaml
- tqdm
- scikit-learn

---

## Acceptance checklist (must pass)

1. `python -m src.train --help` runs on Windows.
2. Training can run 1 fold end-to-end and saves a checkpoint.
3. Inference produces a CSV with correct columns and correct `image_id` ordering from `sample_submission.csv`.
4. All predictions are clamped to [0,1].
5. Fold ensemble inference uses exactly K checkpoints and averages outputs.

---

## Notes for Codex (implementation details)

- Always derive test order from `sample_submission.csv` (never OS directory listing).
- Use Windows-safe paths via `pathlib.Path`.
- Implement robust image extension fallback (`.jpg`, `.png`, `.jpeg`).
- Keep the code “Kaggle-friendly” (no exotic dependencies; avoid Linux-only assumptions).
