"""
data.py
-------
Data loading and preparation for the German electricity demand study.

* Load the raw OPSD 60-minute single-index file.
* Extract the German actual-load series (DE_load_actual_entsoe_transparency).
* Restrict to the modelling window (2015-01-01 onwards).
* Aggregate hourly load to daily and weekly average demand, in gigawatts (GW).
"""

from __future__ import annotations
import pandas as pd

RAW_LOAD_COLUMN = "DE_load_actual_entsoe_transparency"


def load_raw_load(path: str, start: str = "2015-01-01") -> pd.Series:
    """
    Load the German hourly electricity load series from the OPSD file.

    Parameters
    ----------
    path : str
        Path (local file or URL) to the OPSD 60-minute single-index CSV.
    start : str, default "2015-01-01"
        Inclusive start date; earlier observations are discarded.

    Returns
    -------
    pd.Series
        Hourly load in megawatts (MW), indexed by timestamp, gaps dropped.
    """
    df = pd.read_csv(
        path,
        usecols=["utc_timestamp", RAW_LOAD_COLUMN],
        parse_dates=["utc_timestamp"],
    )
    df = df.rename(columns={"utc_timestamp": "date", RAW_LOAD_COLUMN: "load_mw"})
    df = df.set_index("date").sort_index()

    load = df["load_mw"].astype(float)
    load = load[load.notna()]
    load = load[start:]
    return load


def to_weekly_gw(load_mw: pd.Series) -> pd.Series:
    """
    Aggregate hourly load (MW) to weekly mean load (GW).

    Weekly averaging removes the within-day and within-week cycles, leaving the
    annual seasonality and slow level changes that we model.

    Parameters
    ----------
    load_mw : pd.Series
        Hourly load in megawatts.

    Returns
    -------
    pd.Series
        Weekly mean load in gigawatts, named 'load_gw' (W-SUN frequency).
    """
    weekly = load_mw.resample("W").mean() / 1000.0
    weekly = weekly.asfreq("W")
    weekly = weekly.interpolate("time")
    weekly.name = "load_gw"
    return weekly


def to_daily_gw(load_mw: pd.Series) -> pd.Series:
    """
    Aggregate hourly load (MW) to daily mean load (GW).

    Parameters
    ----------
    load_mw : pd.Series
        Hourly load in megawatts.

    Returns
    -------
    pd.Series
        Daily mean load in gigawatts, named 'load_gw'.
    """
    daily = load_mw.resample("D").mean() / 1000.0
    daily.name = "load_gw"
    return daily
