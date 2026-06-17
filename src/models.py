"""Model definitions for multi-output concrete performance prediction."""

from __future__ import annotations

from typing import Dict

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def build_models(random_state: int = 42) -> Dict[str, object]:
    """Create a dictionary of machine learning models.

    Returns
    -------
    dict
        Model name mapped to a scikit-learn compatible estimator.
    """
    matern_kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * Matern(length_scale=1.0, nu=2.5)
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2))
    )

    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=3,
                random_state=random_state,
            )
        ),
        "support_vector_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", MultiOutputRegressor(SVR(kernel="rbf", C=100.0, epsilon=0.1, gamma="scale"))),
            ]
        ),
        "gaussian_process_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    MultiOutputRegressor(
                        GaussianProcessRegressor(
                            kernel=matern_kernel,
                            normalize_y=True,
                            random_state=random_state,
                            n_restarts_optimizer=0,
                            optimizer=None,
                        )
                    ),
                ),
            ]
        ),
    }

    return models
