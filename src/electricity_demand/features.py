"""
features.py
-----------
Temperature feature engineering for the electricity demand study.

Daily mean temperature for a representative location (Berlin) is downloaded from
the Open-Meteo historical archive API and converted into weekly features,
including heating and cooling degree days.
"""

from __future__ import annotations
import requests
import numpy as np
import pandas as pd


def get_open_meteo_temperature(
    latitude: float = 52.52,
    longitude: float = 13.41,
    start_date: str = "2015-01-01",
    end_date: str = "2020-12-31",
) -> pd.DataFrame:
    """
    Download daily mean temperature from the Open-Meteo archive API.

    Parameters
    ----------
    latitude, longitude : float
        Location coordinates. Defaults to Berlin (52.52 N, 13.41 E).
    start_date, end_date : str
        Inclusive date range in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        Daily mean temperature (deg C), indexed by timezone-naive date.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean",
        "timezone": "Europe/Berlin",
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()["daily"]

    temp = pd.DataFrame({
        "date": pd.to_datetime(data["time"]),
        "temperature_2m_mean": data["temperature_2m_mean"],
    }).set_index("date")

    return temp


def build_weekly_temperature_features(
    temp_daily: pd.DataFrame,
    weekly_index: pd.Index,
    base_heat: float = 15.5,
    base_cool: float = 22.0,
) -> pd.DataFrame:
    """
    Convert daily temperature into weekly temperature features.

    The daily temperature index is aligned to the timezone of the weekly load
    index before resampling, so the resulting features match the load series.

    Parameters
    ----------
    temp_daily : pd.DataFrame
        Daily mean temperature, indexed by date (from get_open_meteo_temperature).
    weekly_index : pd.Index
        The weekly (W-SUN) index of the load series, used to align features.
    base_heat : float, default 15.5
        Threshold below which heating demand accrues (deg C).
    base_cool : float, default 22.0
        Threshold above which cooling demand accrues (deg C).

    Returns
    -------
    pd.DataFrame
        Weekly temperature features aligned to weekly_index:
        temp_mean, temp_min, temp_max, heating_degree, cooling_degree.
    """
    t = temp_daily["temperature_2m_mean"].copy()

    # Match the timezone of the load index so resampling aligns correctly
    if weekly_index.tz is not None and t.index.tz is None:
        t.index = t.index.tz_localize(weekly_index.tz)
    elif weekly_index.tz is None and t.index.tz is not None:
        t.index = t.index.tz_localize(None)

    # Resample daily temperature to weekly (W-SUN, matching the load series)
    weekly_mean = t.resample("W").mean()
    weekly_min  = t.resample("W").min()
    weekly_max  = t.resample("W").max()
    hdd = np.maximum(base_heat - t, 0).resample("W").sum()
    cdd = np.maximum(t - base_cool, 0).resample("W").sum()

    feats = pd.DataFrame({
        "temp_mean": weekly_mean,
        "temp_min":  weekly_min,
        "temp_max":  weekly_max,
        "heating_degree": hdd,
        "cooling_degree": cdd,
    })

    # Reindex onto the exact load weekly index; interpolate any small gaps
    feats = feats.reindex(weekly_index).interpolate("time")
    return feats


def make_supervised_table(
    data: pd.DataFrame,
    target: str = "load_gw",
    lags=(1, 2, 3, 4, 52),
    roll_window: int = 4,
) -> pd.DataFrame:
    """
    Build a leakage-free supervised-learning table for demand forecasting.

    Each row predicts the target at week t using only information available before
    week t. Lag and rolling features are shifted by one step so the current
    target never leaks into its own predictors.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain the target column and any temperature feature columns.
    target : str, default "load_gw"
        Name of the column to be forecast.
    lags : tuple of int
        Which past weeks of the target to include as lag features.
    roll_window : int, default 4
        Window length (in weeks) for the rolling mean and standard deviation.

    Returns
    -------
    pd.DataFrame
        Feature table including the target, with early rows containing NaNs
        (from the longest lag) dropped.
    """
    df = data.copy()

    # Lag features: demand at previous weeks
    for lag in lags:
        df[f"lag_{lag}"] = df[target].shift(lag)

    # Rolling features: computed on PAST values only (shift(1) before rolling)
    past = df[target].shift(1)
    df[f"roll_mean_{roll_window}"] = past.rolling(roll_window).mean()
    df[f"roll_std_{roll_window}"]  = past.rolling(roll_window).std()

    # Calendar features, encoded cyclically so week 52 and week 1 are adjacent
    week = df.index.isocalendar().week.astype(int).values
    month = df.index.month.values
    df["week_sin"]  = np.sin(2 * np.pi * week / 52)
    df["week_cos"]  = np.cos(2 * np.pi * week / 52)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Drop rows with NaNs introduced by the longest lag / rolling window
    df = df.dropna()
    return df
