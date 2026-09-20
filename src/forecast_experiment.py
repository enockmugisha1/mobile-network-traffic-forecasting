"""One-step-ahead forecasting experiment for LSTM, TCN and Transformer.

Trains and evaluates the three sequential models on the three highest-traffic
Milan grid squares (5161, 5059, 5259) using a strictly chronological
train / validation / test split. For each (square, model) pair the script
records MAE / RMSE / MAPE, training and inference timing, and writes the
per-area forecast figures used in the report.

Reproduces:
    outputs/forecast_results.csv
    outputs/forecast_timing.csv
    outputs/figures/forecast_sq{square}_{model}.png
    outputs/figures/forecast_sq{square}_all_models.png

Note: this is a training script (~45 min on CPU). It is deterministic
(seed 42) but does not need to be re-run to read the committed results.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from models import LSTMForecaster, TCNForecaster, TransformerForecaster

# ── configuration ──────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "outputs" / "internet_traffic_dataset.csv"
FIG_DIR = REPO / "outputs" / "figures"
OUT_DIR = REPO / "outputs"

SQUARES = [5161, 5059, 5259]          # three highest-traffic areas (see eda.py)
SEQ_LEN = 144                         # 24 h of history at 10-minute resolution
BATCH = 64
EPOCHS = 30
LR = 1e-3
SEED = 42

TRAIN_END = "2013-12-08 23:50:00"
VAL_START = "2013-12-09 00:00:00"
VAL_END = "2013-12-15 23:50:00"
TEST_START = "2013-12-16 00:00:00"
TEST_END = "2013-12-22 23:50:00"

# Best configuration per model, selected during hyperparameter tuning
# (see src/tune_experiment.py and outputs/tuning_results.csv). For all three
# models the compact "baseline" configuration generalised best.
MODEL_BUILDERS = {
    "LSTM": lambda: LSTMForecaster(SEQ_LEN, hidden_size=64, num_layers=2, dropout=0.1),
    "TCN": lambda: TCNForecaster(SEQ_LEN, num_channels=32, num_levels=4, kernel_size=3, dropout=0.1),
    "Transformer": lambda: TransformerForecaster(
        SEQ_LEN, d_model=32, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1
    ),
}

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_square_series(df: pd.DataFrame, square_id: int) -> pd.Series:
    """Return the regular 10-minute series for one square (gaps forward-filled)."""
    s = (
        df[df["square_id"] == square_id]
        .set_index("timestamp")["internet_traffic"]
        .sort_index()
    )
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="10min")
    return s.reindex(full_index).ffill().fillna(0.0)


def make_windows(values: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i : i + seq_len])
        y.append(values[i + seq_len])
    return np.array(X, np.float32), np.array(y, np.float32)


def to_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    Xt = torch.tensor(X).unsqueeze(-1)
    yt = torch.tensor(y).unsqueeze(-1)
    return DataLoader(TensorDataset(Xt, yt), batch_size=BATCH, shuffle=shuffle)


def prepare_split(series: pd.Series):
    """Chronological split + MinMax scaling fitted on the training data only."""
    train_raw = series[:TRAIN_END].values.reshape(-1, 1)
    val_raw = series[VAL_START:VAL_END].values.reshape(-1, 1)
    test_raw = series[TEST_START:TEST_END].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    train_s = scaler.fit_transform(train_raw).ravel()
    val_s = scaler.transform(val_raw).ravel()
    test_s = scaler.transform(test_raw).ravel()

    Xtr, ytr = make_windows(train_s, SEQ_LEN)
    Xva, yva = make_windows(val_s, SEQ_LEN)

    # Context for the rolling forecast: the last SEQ_LEN scaled points before the
    # test window, followed by the scaled test window itself.
    pre_s = scaler.transform(series[:VAL_END].values.reshape(-1, 1)).ravel()
    ctx_s = np.concatenate([pre_s[-SEQ_LEN:], test_s])

    return (
        to_loader(Xtr, ytr, True),
        to_loader(Xva, yva, False),
        ctx_s,
        test_raw.ravel(),
        scaler,
    )


def train_and_forecast(model, train_loader, val_loader, ctx_s, test_actual, scaler):
    """Train with early-stopping-by-checkpoint, then produce a rolling forecast."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val, best_state = float("inf"), None
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    train_time = time.perf_counter() - t0
    model.load_state_dict(best_state)

    # Rolling one-step-ahead forecast (teacher forcing: the true value becomes
    # the next input, matching the assignment definition of x_a(t+1)).
    model.eval()
    buffer = list(ctx_s[:SEQ_LEN])
    preds_scaled = []
    t1 = time.perf_counter()
    with torch.no_grad():
        for i in range(len(test_actual)):
            x = torch.tensor(buffer[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            preds_scaled.append(model(x).item())
            buffer.append(float(ctx_s[SEQ_LEN + i]))
    infer_time = time.perf_counter() - t1

    preds = scaler.inverse_transform(np.array(preds_scaled, np.float32).reshape(-1, 1)).ravel()

    mask = test_actual > 0
    metrics = {
        "MAE": float(mean_absolute_error(test_actual, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(test_actual, preds))),
        "MAPE": float(np.mean(np.abs((test_actual[mask] - preds[mask]) / test_actual[mask])) * 100),
    }
    timing = {
        "train_time_s": round(train_time, 1),
        "infer_time_s": round(infer_time, 4),
        "ms_per_step": round(infer_time / len(test_actual) * 1000, 4),
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    return preds, metrics, timing


def plot_single(square_id, model_name, actual, preds, index):
    plt.figure(figsize=(12, 4))
    plt.plot(index, actual, label="Actual", color="#1f3b5b", linewidth=1.2)
    plt.plot(index, preds, label=f"{model_name} forecast", color="#c0392b", linewidth=1.0, alpha=0.85)
    plt.title(f"Square {square_id} — {model_name} one-step forecast (Dec 16–22, 2013)")
    plt.xlabel("Time")
    plt.ylabel("Internet traffic intensity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"forecast_sq{square_id}_{model_name.lower()}.png", dpi=150)
    plt.close()


def plot_all(square_id, actual, preds_by_model, index):
    plt.figure(figsize=(13, 5))
    plt.plot(index, actual, label="Actual", color="black", linewidth=1.4)
    colors = {"LSTM": "#2e86de", "TCN": "#c0392b", "Transformer": "#27ae60"}
    for name, preds in preds_by_model.items():
        plt.plot(index, preds, label=name, linewidth=1.0, alpha=0.8, color=colors.get(name))
    plt.title(f"Square {square_id} — all models (Dec 16–22, 2013)")
    plt.xlabel("Time")
    plt.ylabel("Internet traffic intensity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"forecast_sq{square_id}_all_models.png", dpi=150)
    plt.close()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading dataset...")
    df = pd.read_csv(CSV, parse_dates=["timestamp"], usecols=["square_id", "timestamp", "internet_traffic"])

    result_rows, timing_rows = [], []
    test_index = pd.date_range(TEST_START, TEST_END, freq="10min")

    for square_id in SQUARES:
        print(f"\n=== Square {square_id} ===")
        series = load_square_series(df, square_id)
        train_loader, val_loader, ctx_s, test_actual, scaler = prepare_split(series)
        index = test_index[: len(test_actual)]

        preds_by_model = {}
        for model_name, build in MODEL_BUILDERS.items():
            torch.manual_seed(SEED)
            model = build()
            preds, metrics, timing = train_and_forecast(
                model, train_loader, val_loader, ctx_s, test_actual, scaler
            )
            preds_by_model[model_name] = preds
            print(
                f"  {model_name:12s} MAE={metrics['MAE']:.2f} "
                f"RMSE={metrics['RMSE']:.2f} MAPE={metrics['MAPE']:.2f}% "
                f"train={timing['train_time_s']}s"
            )
            result_rows.append({"square_id": square_id, "model": model_name, **metrics})
            timing_rows.append({"square_id": square_id, "model": model_name, **timing})
            plot_single(square_id, model_name, test_actual, preds, index)

        plot_all(square_id, test_actual, preds_by_model, index)

    pd.DataFrame(result_rows).to_csv(OUT_DIR / "forecast_results.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(OUT_DIR / "forecast_timing.csv", index=False)
    print("\nSaved forecast_results.csv, forecast_timing.csv and figures to outputs/.")


if __name__ == "__main__":
    main()
