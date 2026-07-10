"""
sarimax.py
----------
SARIMA / SARIMAX model fitting, order selection, and forecasting.

Order selection uses the Akaike Information Criterion (AIC), which balances
model fit against complexity. Fits that fail to converge are skipped rather
than raising, so the search is robust across the full parameter grid.
"""

from __future__ import annotations
import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


def search_sarima_orders(
    y_train: pd.Series,
    p_range=range(0, 7),
    d_range=range(0, 3),
    q_range=range(0, 7),
    seasonal_order=(1, 1, 1, 52),
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Grid-search non-seasonal (p, d, q) orders by AIC, with the seasonal
    component held fixed.

    Parameters
    ----------
    y_train : pd.Series
        Training series.
    p_range, d_range, q_range : iterable of int
        Candidate values for the AR order, differencing order, and MA order.
    seasonal_order : tuple
        Fixed seasonal order (P, D, Q, s).
    verbose : bool
        If True, print progress and each successful fit's AIC.

    Returns
    -------
    pd.DataFrame
        One row per successful fit, with columns (p, d, q, aic), sorted by AIC.
    """
    results = []
    combos = list(itertools.product(p_range, d_range, q_range))
    total = len(combos)

    for i, (p, d, q) in enumerate(combos, start=1):
        try:
            model = SARIMAX(
                y_train,
                order=(p, d, q),
                seasonal_order=seasonal_order,
                trend="c" if d == 0 else None,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False)
            results.append({"p": p, "d": d, "q": q, "aic": fit.aic})

            if verbose:
                print(f"[{i:3d}/{total}] SARIMA({p},{d},{q}) AIC={fit.aic:.2f}")

        except Exception:
            # Non-convergent or invalid parameterisations are skipped
            if verbose:
                print(f"[{i:3d}/{total}] SARIMA({p},{d},{q}) failed - skipped")
            continue

    return pd.DataFrame(results).sort_values("aic").reset_index(drop=True)


def fit_sarimax(y_train: pd.Series, order: tuple, seasonal_order: tuple,
                X_train: pd.DataFrame | None = None):
    """
    Fit a SARIMAX model with the given orders and optional exogenous regressors.

    Parameters
    ----------
    y_train : pd.Series
        Training target series.
    order : tuple
        Non-seasonal order (p, d, q).
    seasonal_order : tuple
        Seasonal order (P, D, Q, s).
    X_train : pd.DataFrame, optional
        Exogenous regressors aligned to y_train (used in Part 4).

    Returns
    -------
    statsmodels results object
        The fitted model.
    """
    model = SARIMAX(
        y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        trend="c" if order[1] == 0 else None,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def forecast_sarimax(model_fit, horizon: int, index: pd.Index,
                     X_test: pd.DataFrame | None = None, alpha: float = 0.05):
    """
    Produce a forecast with confidence intervals from a fitted SARIMAX model.

    Parameters
    ----------
    model_fit : statsmodels results object
        A fitted SARIMAX model.
    horizon : int
        Number of steps to forecast.
    index : pd.Index
        Datetime index to assign to the forecast.
    X_test : pd.DataFrame, optional
        Exogenous regressors for the forecast period.
    alpha : float, default 0.05
        Significance level; 0.05 gives a 95% confidence interval.

    Returns
    -------
    tuple of (pd.Series, pd.DataFrame)
        (point forecast, confidence interval with lower/upper columns)
    """
    fc = model_fit.get_forecast(steps=horizon, exog=X_test)
    mean = fc.predicted_mean
    conf = fc.conf_int(alpha=alpha)
    mean.index = index
    conf.index = index
    return mean, conf


def search_sarima_orders_resumable(
    y_train: pd.Series,
    checkpoint_path: str,
    p_range=range(0, 7),
    d_range=range(0, 3),
    q_range=range(0, 7),
    seasonal_order=(1, 1, 1, 52),
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Grid-search (p, d, q) by AIC, saving progress after every fit.

    Results are appended to a CSV checkpoint file. If the search is interrupted,
    re-running this function skips any combination already recorded, allowing
    the search to resume rather than restart.

    Parameters
    ----------
    y_train : pd.Series
        Training series.
    checkpoint_path : str
        CSV file used to persist results between runs.
    p_range, d_range, q_range : iterable of int
        Candidate orders to search.
    seasonal_order : tuple
        Fixed seasonal order (P, D, Q, s).
    verbose : bool
        If True, print progress for each fit.

    Returns
    -------
    pd.DataFrame
        All results so far, sorted by AIC (best first).
    """
    import os

    # Resume from any previously completed fits
    if os.path.exists(checkpoint_path):
        done = pd.read_csv(checkpoint_path)
        completed = {tuple(r) for r in done[["p", "d", "q"]].values}
        if verbose:
            print(f"Resuming: {len(completed)} combinations already done.\n")
    else:
        completed = set()

    combos = list(itertools.product(p_range, d_range, q_range))
    total = len(combos)

    for i, (p, d, q) in enumerate(combos, start=1):
        if (p, d, q) in completed:
            continue                       # already fitted in an earlier run

        try:
            model = SARIMAX(
                y_train,
                order=(p, d, q),
                seasonal_order=seasonal_order,
                trend="c" if d == 0 else None,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False)
            row = {"p": p, "d": d, "q": q, "aic": fit.aic}

            if verbose:
                print(f"[{i:3d}/{total}] SARIMA({p},{d},{q}) AIC={fit.aic:.2f}")

        except Exception:
            if verbose:
                print(f"[{i:3d}/{total}] SARIMA({p},{d},{q}) failed - skipped")
            continue

        # Persist immediately so nothing is lost on disconnect
        pd.DataFrame([row]).to_csv(
            checkpoint_path,
            mode="a",
            header=not os.path.exists(checkpoint_path),
            index=False,
        )

    results = pd.read_csv(checkpoint_path)
    return results.sort_values("aic").reset_index(drop=True)
