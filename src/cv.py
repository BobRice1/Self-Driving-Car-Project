from typing import List, Tuple

import numpy as np
import pandas as pd


def blocked_kfold_split(df: pd.DataFrame, kfold: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if kfold < 2:
        raise ValueError("kfold must be >= 2 for cross-validation.")
    if "image_id" not in df.columns:
        raise ValueError("Expected 'image_id' column in training dataframe.")

    sorted_df = df.sort_values("image_id").reset_index(drop=True)
    all_indices = np.arange(len(sorted_df))
    val_splits = np.array_split(all_indices, kfold)

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for val_idx in val_splits:
        train_mask = np.ones(len(sorted_df), dtype=bool)
        train_mask[val_idx] = False
        train_idx = all_indices[train_mask]
        folds.append((train_idx, val_idx))
    return folds

