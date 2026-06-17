"""Feature engineering utilities."""

from __future__ import annotations

import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add binder content and water-to-binder ratio.

    Parameters
    ----------
    df:
        Input dataframe containing at least cement, water, silica fume and fly ash columns.

    Returns
    -------
    pd.DataFrame
        A copy of the dataframe with engineered features added.
    """
    data = df.copy()

    required = ["cement_kg_m3", "water_kg_m3", "silica_fume_pct", "fly_ash_pct"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {missing}")

    data["silica_fume_kg_m3"] = data["cement_kg_m3"] * data["silica_fume_pct"] / 100.0
    data["fly_ash_kg_m3"] = data["cement_kg_m3"] * data["fly_ash_pct"] / 100.0
    data["binder_kg_m3"] = data["cement_kg_m3"] + data["silica_fume_kg_m3"] + data["fly_ash_kg_m3"]
    data["water_binder_ratio"] = data["water_kg_m3"] / data["binder_kg_m3"]

    return data
