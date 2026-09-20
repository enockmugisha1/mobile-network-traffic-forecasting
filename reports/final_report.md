# Comparative Analysis of Sequential Models for Mobile Network Traffic Forecasting — Milan

**Author:** Enock Mugisha · African Leadership University, Kigali · ML Formative 1, September 2026

> This markdown file mirrors the submitted report. The full formatted report with all
> figures is the Google Doc / PDF submission. Figures referenced below live in
> `outputs/figures/`.

## Abstract
One-step-ahead internet-traffic forecasting is studied on the Telecom Italia Big Data
Challenge (Milan) dataset. Three sequential deep-learning models — an LSTM, a Temporal
Convolutional Network (TCN), and an encoder-only Transformer — are implemented, tuned, and
evaluated on the three highest-traffic grid areas during the test week 16–22 December 2013.
The TCN gives the best overall accuracy (MAPE 9.01% on the busiest area, 7.21% on the third),
with the fewest parameters and fastest inference; the Transformer is weakest on every area,
consistent with attention models being data-hungry on short univariate series.

## 1. Research question
*How do different sequential models compare for one-step-ahead mobile network traffic
forecasting, and how does their performance vary across geographical areas with different
traffic characteristics?*

## 2. Dataset and data preparation
62 daily tab-separated files, 10-minute resolution, 10,000 grid squares, 1 Nov 2013 – 1 Jan
2014. Each raw row is `square_id, timestamp_ms, country_code, sms_in, sms_out, call_in,
call_out, internet`. Rows with country code 39 (domestic Italian SIMs — the dominant share)
are retained and their per-interval activity fields aggregated into a single traffic-intensity
value per `(square_id, timestamp)`; internet activity dominates this quantity, so the series is
effectively an internet-traffic-intensity measure. Aggregation is streamed file-by-file with
column selection and dtype downcasting to keep peak RAM under ~3 GB (89,127,473 aggregated
rows). See `src/data_pipeline.py`.

## 3. Exploratory analysis (`src/eda.py`)
- **Spatial distribution (Fig 1):** heavy right-tailed / approximately log-normal. Top areas
  are 5161 (12,740,060), 5059 (11,170,854), 5259 (10,485,780).
- **Temporal dynamics (Fig 2):** squares 5161/5059/5259 show strong daily cycles and a
  weekday/weekend pattern; reference squares 4159 and 4556 are lower and more irregular.
- **Stationarity:** ADF statistic −14.8151 (p ≈ 0), stationary at 1% → no differencing needed
  (`outputs/adf_results.txt`).
- **STL (Fig 3):** seasonal strength 0.8592 (dominant daily cycle); slow downward trend into the
  Christmas holiday.
- **Autocorrelation (Fig 4):** ACF spike at lag 144 (1 day) and 288 (2 days); short-range PACF
  → motivates a 144-step (24 h) input window.

## 4. Methodology (`src/models.py`, `src/forecast_experiment.py`)
One-step-ahead task with sequence length L = 144. Strictly chronological split: train
1 Nov–8 Dec, validation 9–15 Dec, test 16–22 Dec. Per-area MinMax scaling fitted on training
only (no leakage). Rolling one-step forecast with the true value fed back as the next input.
Adam, MSE loss, ReduceLROnPlateau, gradient clipping, 30 epochs, best-validation checkpoint,
seed 42.

| Model | Key config | Params |
|---|---|---|
| LSTM | hidden 64, 2 layers, dropout 0.1 | 50,497 |
| TCN | 32 ch, 4 levels, kernel 3, dilations 1-2-4-8 | 21,953 |
| Transformer | d_model 32, 4 heads, 2 layers, ff 128 | 25,505 |

Five configurations per model were tuned on square 5161 (`src/tune_experiment.py`,
`outputs/tuning_results.csv`); the compact baseline generalised best in every case — larger
variants overfit the ~5,600 training windows.

## 5. Results (`outputs/forecast_results.csv`, `outputs/forecast_timing.csv`)

**Square 5161** — TCN 80.52 / 117.57 / 9.01% · LSTM 93.66 / 134.83 / 11.79% · Transformer 105.19 / 148.68 / 14.21%
**Square 5059** — LSTM 75.65 / 104.18 / 8.58% · TCN 82.56 / 113.71 / 9.88% · Transformer 89.07 / 124.76 / 9.06%
**Square 5259** — TCN 64.38 / 91.99 / 7.21% · LSTM 71.69 / 99.88 / 8.56% · Transformer 85.44 / 115.60 / 10.99%
*(MAE / RMSE / MAPE)*

TCN is best overall (lowest error on 2 of 3 areas, fewest params, fastest inference). LSTM wins
on 5059. The Transformer trails everywhere and costs ~4× the training time. All models are worst
on the busiest area (5161) and best on 5259 — forecasting difficulty tracks traffic complexity.
Failure cases: all models underestimate the Monday 16 Dec morning surge and overestimate the
pre-Christmas weekend drop (a once-in-dataset calendar anomaly).

## 6. Conclusion
TCN offers the best accuracy/cost trade-off for this problem. Limitations: univariate, no spatial
or calendar features, two-month window, CPU-only tuning. Future work: multivariate/spatial
models, calendar features, longer history, probabilistic forecasts.

## References
See the full report for the complete IEEE reference list [1]–[7].
