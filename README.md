# Machine Learning Models for Predicting Concrete Mechanical and Durability Performance

This repository provides a reproducible Python workflow for predicting the mechanical, durability, and environmental performance of concrete mixtures using machine learning.

The project is designed as a research-oriented portfolio repository for sustainable construction materials, green concrete, recycled aggregates, silica fume, durability assessment, and data-driven concrete design.

## Research objective

The objective is to compare several machine learning models for predicting multiple concrete performance indicators from mixture design parameters:

- Compressive strength
- Splitting tensile strength
- Porosity
- Electrical resistivity
- Water absorption
- CO₂ emission index

The workflow includes data preprocessing, feature engineering, model training, model comparison, prediction plots, and feature importance analysis.

## Machine learning models

The following models are implemented:

- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regression
- Gaussian Process Regression with a Matérn kernel

The models are trained using a multi-output regression strategy, allowing the prediction of several concrete properties at the same time.

## Repository structure

```text
concrete-ml-performance-prediction/
│
├── data/
│   └── sample_concrete_dataset.csv
├── notebooks/
│   └── 01_quick_start.ipynb
├── src/
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── models.py
│   ├── predict.py
│   └── train.py
├── results/
│   └── figures/
├── models/
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## Dataset

The repository includes a synthetic sample dataset to demonstrate the full machine learning workflow.

The input features are:

- Cement content, kg/m³
- Water content, kg/m³
- Fine aggregate content, kg/m³
- Coarse aggregate content, kg/m³
- Recycled coarse aggregate replacement, %
- Silica fume content, %
- Fly ash content, %
- Superplasticizer content, %
- Curing age, days
- Saturation ratio, %
- Engineered binder content, kg/m³
- Engineered water-to-binder ratio

The target variables are:

- Compressive strength, MPa
- Splitting tensile strength, MPa
- Porosity, %
- Electrical resistivity, kΩ·cm
- Water absorption, %
- CO₂ emission, kg/m³

> Important: the included dataset is synthetic and is provided only for demonstration. For publication or scientific reporting, replace it with experimental data and clearly cite the data source.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\\Scripts\\activate       # Windows
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the training workflow

From the repository root, run:

```bash
python -m src.train --data data/sample_concrete_dataset.csv --output results
```

This will:

1. Load the dataset
2. Add engineered features
3. Split the data into training and test sets
4. Train four machine learning models
5. Save trained models in `models/`
6. Save metrics in `results/model_metrics.csv`
7. Save prediction figures in `results/figures/`

## Use a trained model for prediction

Example using the Random Forest model:

```bash
python -m src.predict \
  --model models/random_forest.joblib \
  --input data/sample_concrete_dataset.csv \
  --output results/predictions.csv
```
## Future improvements

- Experimental concrete data from your own research
- Cross-validation and hyperparameter optimization
- SHAP explainability analysis
- CO₂ efficiency score combining emissions and performance
- Uncertainty quantification using Gaussian Process Regression
- A scientific manuscript-style methodology section

## Disclaimer

This project is intended for research training, portfolio demonstration, and reproducible workflow development. The sample dataset is synthetic and should not be interpreted as experimental evidence.
