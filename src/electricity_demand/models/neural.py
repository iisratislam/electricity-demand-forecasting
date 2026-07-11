"""
neural.py
---------
LSTM sequence preparation and model construction for hourly electricity demand.

The LSTM is trained on the hourly load series using a sliding-window formulation:
each input is a fixed-length window of past hours, and the target is the next hour.
"""

from __future__ import annotations
import numpy as np


def make_sequences(series: np.ndarray, lookback: int = 168):
    """
    Convert a 1-D series into overlapping (input window, next value) pairs.

    Parameters
    ----------
    series : np.ndarray
        Scaled 1-D array of load values.
    lookback : int, default 168
        Number of past hours used as input (168 = one week).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        X of shape (n_samples, lookback, 1) and y of shape (n_samples,).
    """
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i])
        y.append(series[i])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    return X, y
