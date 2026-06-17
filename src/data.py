"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMNS
from .features import add_engineered_features


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a concrete dataset and add engineered features."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)
    data = add_engineered_features(data)

    required = FEATURE_COLUMNS + TARGET_COLUMNS
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return data


def split_features_targets(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train and test features/targets."""
    x = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMNS]

    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )
