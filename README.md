# Mobile Network Traffic Forecasting — Milan

A comparative empirical study of sequential deep learning models for one-step-ahead internet traffic forecasting using the Telecom Italia Big Data Challenge dataset.

**Author:** Enock Mugisha
**Institution:** African Leadership University (ALU), Kigali, Rwanda
**Programme:** BSc Software Engineering (Machine Learning specialisation)
**Assignment:** Formative 1 — Machine Learning, September 2026

---

## Overview

This project investigates how different sequential models compare for one-step-ahead mobile network traffic forecasting across geographical areas with different traffic characteristics. The study uses 62 days of internet traffic data recorded at 10-minute intervals across 10,000 geographical grid areas in Milan, Italy.

Three fundamentally different sequential deep learning architectures are implemented, tuned, and evaluated:

| Model | Architecture type |
|---|---|
| LSTM | Recurrent (gated memory) |
| TCN | Dilated causal convolutions |
| Transformer | Self-attention encoder |

---

## Research Question

How do different sequential models compare for one-step-ahead mobile network traffic forecasting, and how does their performance vary across geographical areas with different traffic characteristics?

---

## Key Results

Evaluation on the test week December 16-22, 2013:

**Square 5161 - Highest traffic area**

| Model | MAE | RMSE | MAPE | Train Time |
|---|---|---|---|---|
| TCN | 80.52 | 117.57 | 9.01% | 116s |
| LSTM | 93.66 | 134.83 | 11.79% | 106s |
| Transformer | 105.19 | 148.68 | 14.21% | 466s |

**Square 5059 - Second highest traffic**

| Model | MAE | RMSE | MAPE | Train Time |
|---|---|---|---|---|
| LSTM | 75.65 | 104.18 | 8.58% | 106s |
| TCN | 82.56 | 113.71 | 9.88% | 116s |
| Transformer | 89.07 | 124.76 | 9.06% | 466s |

**Square 5259 - Third highest traffic**

| Model | MAE | RMSE | MAPE | Train Time |
|---|---|---|---|---|
| TCN | 64.38 | 91.99 | 7.21% | 116s |
| LSTM | 71.69 | 99.88 | 8.56% | 106s |
| Transformer | 85.44 | 115.60 | 10.99% | 466s |

TCN is the best overall model. Lowest error on 2 of 3 areas, fastest inference, fewest parameters. The Transformer underperformed on all areas.

---

## Dataset

Source: Telecom Italia Big Data Challenge - Milan Grid Dataset
Download: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV

| Property | Value |
|---|---|
| Time range | Nov 1, 2013 to Jan 1, 2014 |
| Temporal resolution | 10 minutes |
| Grid areas | 10,000 |
| Daily files | 62 |
| Raw rows total | ~319 million |
| Aggregated rows | 89,127,473 |

Column 2 in the raw files is the country code (39 = Italy, others = foreign roaming), not an activity type. Internet activity is column 7. Rows with country code 39 (domestic traffic — the dominant share) are retained, and the per-interval activity fields are aggregated into a single traffic-intensity value per (square_id, time slot). Because internet activity dominates this quantity by one to two orders of magnitude, the resulting series is effectively an internet-traffic-intensity measure, which is the forecasting target.

The raw dataset files are not committed to this repository due to their size (~15 GB). Place the 62 daily .txt files inside a folder called mobile-network/ at the root of this repository.

---

## Repository Structure

mobile-network-traffic-forecasting/
- mobile-network/            <- Raw dataset (download separately)
- src/
  - data_pipeline.py         <- Raw file parser and aggregator
  - eda.py                   <- Exploratory analysis (fig1-4 + ADF stationarity test)
  - tune_experiment.py       <- Hyperparameter tuning (5 configs/model on square 5161)
  - models.py                <- LSTM, TCN, Transformer definitions
  - forecast_experiment.py   <- Training, evaluation, plots
  - save_best_model.py       <- Save best model weights
- outputs/
  - internet_traffic_dataset.csv
  - forecast_results.csv
  - forecast_timing.csv
  - tuning_results.csv
  - best_model_tcn.pt
  - figures/                 <- All EDA and forecast plots
- reports/
  - final_report.md          <- Concise markdown mirror of the submitted report
- requirements.txt
- README.md

---

## Setup

Requirements: Python 3.10 or higher, Linux or macOS

    git clone https://github.com/enockmugisha1/mobile-network-traffic-forecasting.git
    cd mobile-network-traffic-forecasting
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install torch --index-url https://download.pytorch.org/whl/cpu

---

## How to Reproduce

Run in this exact order:

Step 1 - Parse raw dataset (~4 minutes)
    python src/data_pipeline.py

Step 2 - Exploratory analysis (~2 minutes)
    python src/eda.py

Step 3 - Hyperparameter tuning (optional, documents the 5 configs/model)
    python src/tune_experiment.py

Step 4 - Forecasting experiment: train + evaluate LSTM/TCN/Transformer
         on squares 5161/5059/5259 (~45 minutes on CPU)
    python src/forecast_experiment.py

Step 5 - Save best model (optional, ~3 minutes)
    python src/save_best_model.py

---

## Hardware

| Component | Specification |
|---|---|
| Machine | Dell Latitude 7430 |
| Processor | Intel Core i7-1265U (12 threads) |
| RAM | 16 GB |
| GPU | None (CPU-only) |
| OS | Ubuntu 24.04 LTS |

---

## Reproducibility

- All experiments use random seed 42
- Scaler fitted on training data only (no data leakage)
- Train/val/test split is strictly chronological

---

## References

1. S. Hochreiter and J. Schmidhuber, Long short-term memory, Neural Computation, vol. 9, no. 8, pp. 1735-1780, 1997.
2. S. Bai, J. Z. Kolter, and V. Koltun, An empirical evaluation of generic convolutional and recurrent networks for sequence modeling, arXiv:1803.01271, 2018.
3. A. Vaswani et al., Attention is all you need, NeurIPS, 2017.
4. G. Barlacchi et al., A multi-source dataset of urban life in the city of Milan, Scientific Data, vol. 2, p. 150055, 2015.
