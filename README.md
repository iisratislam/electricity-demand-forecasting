# Forecasting Weekly German Electricity Demand

A reproducible time-series forecasting pipeline that models and forecasts weekly
German electricity demand, comparing simple benchmarks against SARIMAX,
feature-based machine learning, and LSTM neural models.

Data source: Open Power System Data (OPSD), German actual load (`DE`),
hourly observations Jan 2015 – Oct 2020, aggregated to weekly average load (GW).

## Research questions

1. How well do simple benchmarks forecast weekly electricity demand?
2. Does a SARIMAX model meaningfully improve on the seasonal-naive benchmark?
3. Do temperature covariates improve forecast accuracy, and are they known at
   the forecast origin?
4. Do feature-based and neural models justify their additional complexity?
5. Which model is most suitable for operational use?

## Project structure

```text
electricity-demand-forecasting/
├── data/            # raw / interim / processed (raw not committed)
├── src/electricity_demand/   # reusable pipeline package
│   ├── data.py      # load & clean OPSD data, resample to weekly
│   ├── features.py  # temperature, calendar, lag & rolling features
│   ├── evaluation.py# MAE, RMSE, MASE, Bias
│   ├── plotting.py  # forecast & diagnostic figures
│   └── models/      # benchmarks, sarimax, feature_models, neural
├── scripts/         # download_data.py, make_features.py, run_pipeline.py
├── notebooks/       # exploration & analysis (EDA, stationarity, each model)
├── outputs/         # figures, forecasts, metrics (generated)
├── reports/         # final report + figures
└── tests/           # unit tests for key functions
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the analysis

```bash
python scripts/download_data.py   # fetch raw OPSD data (not stored in repo)
python scripts/make_features.py   # build the processed weekly dataset
python scripts/run_pipeline.py    # fit all models, save metrics & figures
```

## Models

| Model | Type | Notes |
|-------|------|-------|
| Mean / Naive / Seasonal naive / Drift | Benchmark | Seasonal naive is the key baseline |
| SARIMAX | Statistical | p,d,q x P,D,Q tuned by AIC; optional temperature exog |
| Gradient Boosting | Feature-based ML | lag, rolling, calendar, Fourier features |
| LSTM | Neural | trained on hourly data |

## Evaluation

All models are evaluated on the final **104 weeks** (two-year horizon) using
MAE, RMSE, MASE and Bias. A time-ordered split is used (no random shuffling).
Every advanced model is compared against the seasonal-naive benchmark.

## Author

Built as part of the Advanced Research Topics module, Ulster University.
