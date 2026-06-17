# MATLAB Version

This folder contains a MATLAB implementation of the machine learning workflow for predicting concrete mechanical, durability, and environmental performance indicators.

The MATLAB version is intended to complement the Python workflow and demonstrate reproducible engineering-oriented modelling using MATLAB Statistics and Machine Learning Toolbox.

## Required toolbox

- MATLAB
- Statistics and Machine Learning Toolbox

## Main files

```text
matlab_version/
├── main_train_models.m
├── predict_concrete_properties.m
└── README_MATLAB.md
```

## How to run

Open MATLAB, set the current folder to the repository root or to `matlab_version`, and run:

```matlab
main_train_models
```

The script will:

1. Load `data/sample_concrete_dataset.csv`
2. Split the data into training and testing sets
3. Train machine learning models for each target property
4. Save model files in `models/matlab/`
5. Save metrics and figures in `results/matlab/`

## Models included

- Random Forest-style bagged regression ensemble
- Gradient Boosting regression ensemble
- Support Vector Regression
- Gaussian Process Regression with Matérn 5/2 kernel

## Predicted concrete performance indicators

- Compressive strength
- Splitting tensile strength
- Porosity
- Electrical resistivity
- Water absorption
- CO₂ emission index

## Prediction example

After running `main_train_models.m`, run:

```matlab
predict_concrete_properties
```

This script loads a trained model and predicts the performance of a new concrete mixture.
