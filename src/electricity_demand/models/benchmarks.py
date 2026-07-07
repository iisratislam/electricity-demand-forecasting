"""
benchmarks.py
-------------
Simple benchmark forecasting models for weekly electricity demand.

These provide the baselines that all advanced models must beat. The
seasonal-naive model is the most important reference for strongly seasonal
series such as electricity load.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def mean_forecast(train: pd.Series, horizon: int, index: pd.Index) -> pd.Series:
    """
    Forecast every future point as the mean of the training data.

    Parameters
    ----------
    train : pd.Series
        Training series.
    horizon : int
        Number of steps to forecast.
    index : pd.Index
        Datetime index to assign to the forecast (the test index).

    Returns
    -------
    pd.Series
        Constant forecast equal to the training mean.
    """
    return pd.Series(train.mean(), index=index, name="mean")


def naive_forecast(train: pd.Series, horizon: int, index: pd.Index) -> pd.Series:
    """
    Forecast every future point as the last observed training value.

    Parameters
    ----------
    train : pd.Series
        Training series.
    horizon : int
        Number of steps to forecast.
    index : pd.Index
        Datetime index to assign to the forecast.

    Returns
    -------
    pd.Series
        Constant forecast equal to the final training observation.
    """
    return pd.Series(train.iloc[-1], index=index, name="naive")


def seasonal_naive_forecast(train: pd.Series, horizon: int,
                            seasonality: int, index: pd.Index) -> pd.Series:
    """
    Forecast each point as the value observed one season earlier.

    For weekly data with annual seasonality, this predicts each week using the
    same week of the previous year. It is the key benchmark for this study.

    Parameters
    ----------
    train : pd.Series
        Training series.
    horizon : int
        Number of steps to forecast.
    seasonality : int
        Season length in steps (52 for weekly annual seasonality).
    index : pd.Index
        Datetime index to assign to the forecast.

    Returns
    -------
    pd.Series
        Seasonal-naive forecast.
    """
    last_season = train.iloc[-seasonality:].values
    # Repeat the last observed season forward to cover the whole horizon
    reps = int(np.ceil(horizon / seasonality))
    values = np.tile(last_season, reps)[:horizon]
    return pd.Series(values, index=index, name="seasonal_naive")


def drift_forecast(train: pd.Series, horizon: int, index: pd.Index) -> pd.Series:
    """
    Forecast by extending the average trend (drift) from the training data.

    The drift is the average step change between the first and last training
    points; the forecast continues this straight line into the future.

    Parameters
    ----------
    train : pd.Series
        Training series.
    horizon : int
        Number of steps to forecast.
    index : pd.Index
        Datetime index to assign to the forecast.

    Returns
    -------
    pd.Series
        Linear drift forecast.
    """
    slope = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
    values = train.iloc[-1] + slope * np.arange(1, horizon + 1)
    return pd.Series(values, index=index, name="drift")
