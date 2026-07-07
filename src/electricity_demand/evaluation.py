"""
evaluation.py
-------------
Forecast evaluation metrics for the electricity demand study.

MASE (mean absolute scaled error) is the headline metric: it scales the error
by the in-sample seasonal-naive error, so a value below 1 means the model
outperforms a seasonal-naive forecast.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def mase(y_true: pd.Series, y_pred: pd.Series,
         y_train: pd.Series, seasonality: int = 52) -> float:
    """
    Mean Absolute Scaled Error, scaled by the in-sample seasonal-naive error.

    Parameters
    ----------
    y_true, y_pred : pd.Series
        Actual and predicted test values.
    y_train : pd.Series
        Training series (used to compute the scaling factor).
    seasonality : int, default 52
        Season length used by the seasonal-naive scaling.

    Returns
    -------
    float
        MASE. Below 1.0 indicates the forecast beats seasonal naive.
    """
    naive_errors = np.abs(y_train.iloc[seasonality:].values
                          - y_train.iloc[:-seasonality].values)
    scale = naive_errors.mean()
    return np.mean(np.abs(y_true.values - y_pred.values)) / scale


def evaluate_forecast(name: str, y_true: pd.Series, y_pred: pd.Series,
                      y_train: pd.Series) -> dict:
    """
    Compute MAE, RMSE, MASE, and Bias for a single forecast.

    Parameters
    ----------
    name : str
        Model name (for the results table).
    y_true, y_pred : pd.Series
        Actual and predicted test values.
    y_train : pd.Series
        Training series (needed for MASE scaling).

    Returns
    -------
    dict
        Dictionary of metrics for this model.
    """
    y_pred = y_pred.reindex(y_true.index)   # align on the test index
    errors = y_pred.values - y_true.values

    return {
        "model": name,
        "MAE":  np.mean(np.abs(errors)),
        "RMSE": np.sqrt(np.mean(errors ** 2)),
        "MASE": mase(y_true, y_pred, y_train),
        "Bias": np.mean(errors),
    }
