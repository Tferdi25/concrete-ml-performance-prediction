"""Use a trained model to predict concrete properties for new mixtures.

Example
-------
python -m src.predict --model models/random_forest.joblib --input data/sample_concrete_dataset.csv --output results/predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import FEATURE_COLUMNS, TARGET_COLUMNS
from .features import add_engineered_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict concrete performance with a trained model.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained .joblib model.")
    parser.add_argument("--input", type=str, required=True, help="Path to new mixture CSV file.")
    parser.add_argument("--output", type=str, default="results/predictions.csv", help="Path to save predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model)
    data = pd.read_csv(args.input)
    data = add_engineered_features(data)

    missing = [col for col in FEATURE_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    predictions = model.predict(data[FEATURE_COLUMNS])
    predictions = pd.DataFrame(predictions, columns=[f"predicted_{col}" for col in TARGET_COLUMNS])

    output = pd.concat([data.reset_index(drop=True), predictions], axis=1)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
