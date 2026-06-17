"""Train machine learning models for concrete performance prediction.

Example
-------
python -m src.train --data data/sample_concrete_dataset.csv --output results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMNS
from .data import load_dataset, split_features_targets
from .evaluate import plot_feature_importance, plot_predicted_vs_actual, regression_metrics
from .models import build_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML models for concrete performance prediction.")
    parser.add_argument("--data", type=str, default="data/sample_concrete_dataset.csv", help="Path to CSV dataset.")
    parser.add_argument("--output", type=str, default="results", help="Directory for results.")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory for trained model files.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    figures_dir = output_dir / "figures"
    models_dir = Path(args.models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.data)
    x_train, x_test, y_train, y_test = split_features_targets(
        data,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    all_metrics = []
    models = build_models(random_state=args.random_state)

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        metrics = regression_metrics(y_test, y_pred, TARGET_COLUMNS, model_name)
        all_metrics.append(metrics)

        model_path = models_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)

        plot_predicted_vs_actual(
            y_true=y_test,
            y_pred=y_pred,
            target_names=TARGET_COLUMNS,
            model_name=model_name,
            output_dir=figures_dir,
        )

        if model_name == "random_forest":
            plot_feature_importance(
                model=model,
                feature_names=FEATURE_COLUMNS,
                output_path=figures_dir / "random_forest_feature_importance.png",
            )

    metrics_table = pd.concat(all_metrics, ignore_index=True)
    metrics_table.to_csv(output_dir / "model_metrics.csv", index=False)

    summary = (
        metrics_table.groupby("model")[["mae", "rmse", "r2"]]
        .mean()
        .sort_values("r2", ascending=False)
    )
    summary.to_csv(output_dir / "model_summary.csv")

    print("\nModel summary, averaged across all targets:")
    print(summary.round(4))
    print(f"\nSaved metrics to: {output_dir / 'model_metrics.csv'}")
    print(f"Saved trained models to: {models_dir}")


if __name__ == "__main__":
    main()
