"""Evaluation and plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    target_names: Iterable[str],
    model_name: str,
) -> pd.DataFrame:
    """Compute MAE, RMSE and R² for each target."""
    rows = []
    pred = pd.DataFrame(y_pred, columns=list(target_names), index=y_true.index)

    for target in target_names:
        mae = mean_absolute_error(y_true[target], pred[target])
        rmse = float(np.sqrt(mean_squared_error(y_true[target], pred[target])))
        r2 = r2_score(y_true[target], pred[target])
        rows.append(
            {
                "model": model_name,
                "target": target,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        )

    return pd.DataFrame(rows)


def plot_predicted_vs_actual(
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    target_names: Iterable[str],
    model_name: str,
    output_dir: str | Path,
) -> None:
    """Save predicted-vs-actual scatter plots for each target."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.DataFrame(y_pred, columns=list(target_names), index=y_true.index)

    for target in target_names:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_true[target], pred[target], alpha=0.75)

        min_value = min(y_true[target].min(), pred[target].min())
        max_value = max(y_true[target].max(), pred[target].max())
        ax.plot([min_value, max_value], [min_value, max_value], linestyle="--")

        ax.set_xlabel(f"Actual {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title(f"{model_name}: {target}")
        fig.tight_layout()
        fig.savefig(output_dir / f"{model_name}_{target}_predicted_vs_actual.png", dpi=300)
        plt.close(fig)


def plot_feature_importance(model, feature_names: list[str], output_path: str | Path) -> None:
    """Save feature importance plot when the model exposes feature_importances_."""
    if not hasattr(model, "feature_importances_"):
        return

    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    importance.plot(kind="barh", ax=ax)
    ax.set_xlabel("Feature importance")
    ax.set_title("Random Forest feature importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
