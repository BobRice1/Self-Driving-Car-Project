# Ablation Guide: Isolate What Improved the Kaggle Score

This guide gives a reproducible, minimal ablation for your EfficientNetV2-S setup.
The goal is to isolate three effects:

1. Inference TTA effect (`none` vs `hflip`)
2. Checkpoint selection metric effect (`loss` vs `kaggle_mse`)
3. Loss recipe effect (old MSE-style vs new BCE+MSE speed recipe)



Think of your model setup like a recipe you changed in three ways:

1. You changed how you judge the "best" model checkpoint.
2. You changed the loss used during training.
3. You changed how predictions are made at test time with TTA.

If the final Kaggle score improved, ablation is how you work out which change actually helped.
The rule is simple: change one thing at a time, keep everything else the same, and compare the score.

This guide does that in stages:

- Step 1 checks whether TTA alone helped, without retraining.
- Step 2 checks whether choosing checkpoints by `kaggle_mse` helped, while keeping the old loss.
- Step 3 checks whether the new loss helps on top of the new checkpoint selection.

So the ablation answers:

- "Was the gain mostly from TTA?"
- "Was the gain mostly from better checkpoint selection?"
- "Did the new loss really help, or did it just come along for the ride?"

## Prerequisites

1. Use the same machine, seed, folds, and data paths for all runs.
2. Keep `kfold=5` across all experiments.
3. Record both:
  1. Mean fold `val_kaggle_mse` from training logs
  2. Kaggle LB score from submitted CSV

## Config files

Two ablation configs are already prepared:

- `configs/ablation_select_only.yaml`
  - Old loss recipe
  - Selection metric changed to `kaggle_mse`
- `configs/ablation_loss_plus_select.yaml`
  - New loss recipe
  - Selection metric `kaggle_mse`

## Step 0: Baseline reference

Use your existing baseline submission (`0.0126`) as the reference point.
If needed, regenerate a baseline later with old settings and `--tta none`.

## Step 1: TTA-only ablation (no retraining)

Run inference twice on the same checkpoints (your improved run, e.g. `iteration2`):

```powershell
python -m src.infer --config configs/ablation_loss_plus_select.yaml --kfold 5 --data_dir data --ckpt_dir outputs/checkpoints/iteration2 --out outputs/submissions/ablation_iteration2_tta_none.csv --tta none
python -m src.infer --config configs/ablation_loss_plus_select.yaml --kfold 5 --data_dir data --ckpt_dir outputs/checkpoints/iteration2 --out outputs/submissions/ablation_iteration2_tta_hflip.csv --tta hflip
```

Interpretation:

- Difference between these two submissions isolates pure TTA impact.

## Step 2: Selection-only ablation (train + infer)

Train with old loss, but select checkpoints by `kaggle_mse`:

```powershell
python -m src.train --config configs/ablation_select_only.yaml --kfold 5 --data_dir data --out_dir outputs --run_name ablation_select_only_seed42 --seed 42
```

Infer without TTA first:

```powershell
python -m src.infer --config configs/ablation_select_only.yaml --kfold 5 --data_dir data --ckpt_dir outputs/checkpoints/ablation_select_only_seed42 --out outputs/submissions/ablation_select_only_tta_none.csv --tta none
```

Optional TTA variant:

```powershell
python -m src.infer --config configs/ablation_select_only.yaml --kfold 5 --data_dir data --ckpt_dir outputs/checkpoints/ablation_select_only_seed42 --out outputs/submissions/ablation_select_only_tta_hflip.csv --tta hflip
```

Interpretation:

- Compare to baseline to estimate selection-metric contribution.

## Step 3: Loss+selection ablation (train + infer)

Train with new loss recipe + `kaggle_mse` selection:

```powershell
python -m src.train --config configs/ablation_loss_plus_select.yaml --kfold 5 --data_dir data --out_dir outputs --run_name ablation_loss_plus_select_seed42 --seed 42
```

Infer without TTA first:

```powershell
python -m src.infer --config configs/ablation_loss_plus_select.yaml --kfold 5 --data_dir data --ckpt_dir outputs/checkpoints/ablation_loss_plus_select_seed42 --out outputs/submissions/ablation_loss_plus_select_tta_none.csv --tta none
```

Optional TTA variant:

```powershell
python -m src.infer --config configs/ablation_loss_plus_select.yaml --kfold 5 --data_dir data --ckpt_dir outputs/checkpoints/ablation_loss_plus_select_seed42 --out outputs/submissions/ablation_loss_plus_select_tta_hflip.csv --tta hflip
```

Interpretation:

- Compare Step 3 vs Step 2 to isolate loss-recipe contribution.

## Step 4: Use manifests for traceability

Each run/submission writes a manifest. Keep these linked with your Kaggle submissions:

1. `outputs/checkpoints/<run_name>/run_manifest.json`
2. `outputs/checkpoints/<run_name>/architecture_changes.txt`
3. `outputs/submissions/<submission_name>_manifest.json`

## Step 5: Suggested results table

Create a simple table in your notes:


| Experiment  | Config                    | TTA   | Public LB | Mean val_kaggle_mse | Notes              |
| ----------- | ------------------------- | ----- | --------- | ------------------- | ------------------ |
| Baseline    | old                       | none  | 0.01260   | ...                 | reference          |
| Select-only | ablation_select_only      | none  | ...       | ...                 | isolate selection  |
| Loss+Select | ablation_loss_plus_select | none  | ...       | ...                 | isolate loss       |
| Loss+Select | ablation_loss_plus_select | hflip | ...       | ...                 | isolate TTA on top |


## Decision rule

Keep a change only if:

1. Mean fold `val_kaggle_mse` improves consistently.
2. Kaggle LB does not regress.
3. Improvement repeats across at least one extra seed.

