"""Shared time-series utilities for the forecasting experiments.

Centralises the dataset reload, per-square series construction, windowing, and
the chronological MinMax split so that the tuning, evaluation, and model-saving
scripts all use one definition instead of duplicating it. Keeping this logic in
one place guarantees the splits and scaling are identical across experiments.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Chronological split boundaries shared by every experiment (no data leakage).
TRAIN_END = "2013-12-08 23:50:00"
VAL_START = "2013-12-09 00:00:00"
VAL_END = "2013-12-15 23:50:00"
TEST_START = "2013-12-16 00:00:00"
TEST_END = "2013-12-22 23:50:00"


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Reload the aggregated dataset with column selection + dtype downcasting.

    Only the three required columns are read (``usecols``) and the traffic column
    is held as ``float32`` (identifiers as ``int32``), matching the memory-reduction
    strategy described in the report.
    """
    df = pd.read_csv(
        csv_path,
        parse_dates=["timestamp"],
        usecols=["square_id", "timestamp", "internet_traffic"],
    )
    df["square_id"] = df["square_id"].astype("int32")
    df["internet_traffic"] = df["internet_traffic"].astype("float32")
    return df


def load_square_series(df: pd.DataFrame, square_id: int, freq: str = "10min") -> pd.Series:
    """Return the regular series for one square, gaps forward-filled."""
    s = df[df["square_id"] == square_id].set_index("timestamp")["internet_traffic"].sort_index()
    full_index = pd.date_range(s.index.min(), s.index.max(), freq=freq)
    return s.reindex(full_index).ffill().fillna(0.0)


def make_windows(values: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Slide a length-``seq_len`` window over ``values`` to build (X, y) pairs."""
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i : i + seq_len])
        y.append(values[i + seq_len])
    return np.array(X, np.float32), np.array(y, np.float32)


def to_loader(X: np.ndarray, y: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    Xt = torch.tensor(X).unsqueeze(-1)
    yt = torch.tensor(y).unsqueeze(-1)
    return DataLoader(TensorDataset(Xt, yt), batch_size=batch, shuffle=shuffle)


def prepare_minmax_split(series: pd.Series, seq_len: int, batch: int):
    """Chronological split + MinMax scaling fitted on the training data only.

    Returns ``(train_loader, val_loader, ctx_scaled, test_actual, scaler)`` where
    ``ctx_scaled`` is the last ``seq_len`` scaled points before the test window
    concatenated with the scaled test window, ready for the rolling one-step forecast.
    """
    train_raw = series[:TRAIN_END].values.reshape(-1, 1)
    val_raw = series[VAL_START:VAL_END].values.reshape(-1, 1)
    test_raw = series[TEST_START:TEST_END].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    train_s = scaler.fit_transform(train_raw).ravel()
    val_s = scaler.transform(val_raw).ravel()
    test_s = scaler.transform(test_raw).ravel()

    Xtr, ytr = make_windows(train_s, seq_len)
    Xva, yva = make_windows(val_s, seq_len)

    pre_s = scaler.transform(series[:VAL_END].values.reshape(-1, 1)).ravel()
    ctx_s = np.concatenate([pre_s[-seq_len:], test_s])

    return (
        to_loader(Xtr, ytr, batch, True),
        to_loader(Xva, yva, batch, False),
        ctx_s,
        test_raw.ravel(),
        scaler,
    )
