# Forecasting Weekly German Electricity Demand

A reproducible time-series forecasting pipeline that models and forecasts weekly German electricity demand, comparing simple benchmarks against SARIMA, SARIMAX, feature-based machine learning, and an LSTM neural network.

**Dataset:** Open Power System Data (OPSD), German actual load (`DE_load_actual_entsoe_transparency`), hourly observations January 2015 – October 2020, aggregated to weekly average load (GW).

- Hourly observations: 50,400
- Weekly observations: 301
- Weekly load: mean 55.48 GW, min 46.51 GW, max 63.59 GW
- Train / test split: 197 weeks training, 104 weeks test (two-year horizon)

---

## Research Questions

1. Do complex models meaningfully improve on the seasonal-naïve benchmark?
2. Does adding temperature as an exogenous variable improve forecast accuracy?
3. Which model is most suitable for operational use?

---

## Results Summary

| Model | MASE | Beats Seasonal Naïve? |
|---|---|---|
| Seasonal Naïve | 1.732 | Reference |
| Mean | 2.831 | No |
| Naïve | 2.827 | No |
| Drift | 3.243 | No |
| SARIMA(0,1,6)(0,1,1,52) | 2.295 | No |
| SARIMA(0,1,1)(0,1,1,52) | 2.462 | No |
| SARIMAX + Temperature | 2.114 | No (conditional) |
| Gradient Boosting | 1.421 | Yes (18% improvement) |
| LSTM (one-step-ahead) | 0.097 | Not comparable |

**Key finding:** Only gradient boosting meaningfully outperformed the seasonal-naïve benchmark. All models failed to anticipate the COVID-19 demand collapse (March 2020) as this structural break was absent from training data.

---

## Project Structure
electricity-demand-forecasting/
├── data/
│ └── raw/ # raw OPSD CSV (not committed, download below)
├── src/electricity_demand/ # reusable pipeline package
│ ├── data.py # load, clean, resample OPSD data
│ ├── features.py # temperature, lag, rolling, calendar features
│ ├── evaluation.py # MAE, RMSE, MASE, Bias metrics
│ └── models/
│ ├── benchmarks.py # Mean, Naive, Seasonal Naive, Drift
│ ├── sarimax.py # SARIMA/SARIMAX with AIC grid search
│ └── neural.py # LSTM model definition and training
├── notebooks/
│ ├── 01_data_and_eda.ipynb # data preparation, EDA, stationarity tests
│ ├── 02_benchmarks.ipynb # benchmark forecasts
│ ├── 03_sarimax.ipynb # SARIMA model selection and forecasting
│ ├── 04_temperature_sarimax.ipynb # SARIMAX with temperature covariates
│ ├── 05_feature_model.ipynb # gradient boosting regressor
│ └── 06_lstm.ipynb # LSTM on hourly data
├── outputs/
│ ├── figures/ # all forecast and diagnostic plots
│ ├── forecasts/ # model forecast CSVs
│ └── metrics/ # evaluation metric CSVs
├── requirements.txt
└── README.md

---

## Installation

```bash
git clone https://github.com/iisratislam/electricity-demand-forecasting.git
cd electricity-demand-forecasting
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Reproducing the Analysis

### Step 1 — Download the data

Download manually from OPSD and save to `data/raw/`:
https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv

### Step 2 — Run notebooks in order
01_data_and_eda.ipynb → EDA, stationarity tests, ACF/PACF plots
02_benchmarks.ipynb → Mean, Naive, Seasonal Naive, Drift forecasts
03_sarimax.ipynb → SARIMA AIC grid search, residual diagnostics
04_temperature_sarimax.ipynb → SARIMAX with Berlin temperature covariates
05_feature_model.ipynb → Gradient Boosting with engineered features
06_lstm.ipynb → LSTM on hourly data

All outputs (figures, forecasts, metrics) are saved to `outputs/`.

---

## Data Leakage Prevention

- All lag and rolling features use `.shift(1)` so target week demand is never its own predictor
- MinMaxScaler fitted on training data only, then applied to test set
- TimeSeriesSplit cross-validation always trains on past and validates on future
- Temperature features aligned to load index with no forward-looking information

---

## Author

Israt Islam | Student ID: 23082056
Advanced Research Topics in Data Science
University of Hertfordshire
Click the copy button in the top right corner of the code block and paste directly into GitHub. 🌙


