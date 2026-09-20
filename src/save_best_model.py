"""
Trains the best model (TCN, baseline config) on square 5161
and saves it to outputs/best_model_tcn.pt

This can be loaded later for inference without retraining.
"""
from __future__ import annotations
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from models import TCNForecaster

REPO    = Path(__file__).resolve().parents[1]
CSV     = REPO / "outputs" / "internet_traffic_dataset.csv"
OUT_DIR = REPO / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SEQ_LEN   = 144
BATCH     = 64
EPOCHS    = 30
LR        = 1e-3
SEED      = 42
TOP_SQ    = 5161
TRAIN_END = "2013-12-08 23:50:00"
VAL_END   = "2013-12-15 23:50:00"

torch.manual_seed(SEED)
np.random.seed(SEED)


def make_windows(values, seq_len):
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i: i + seq_len])
        y.append(values[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    print("Loading dataset...")
    df = pd.read_csv(CSV, parse_dates=["timestamp"])

    series = (
        df[df["square_id"] == TOP_SQ]
        .set_index("timestamp")["internet_traffic"]
        .sort_index()
    )

    train_vals = series[:TRAIN_END].values.reshape(-1, 1)
    val_vals   = series[TRAIN_END:VAL_END].values.reshape(-1, 1)

    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_vals).ravel()
    val_s   = scaler.transform(val_vals).ravel()

    Xtr, ytr = make_windows(train_s, SEQ_LEN)
    Xva, yva = make_windows(val_s,   SEQ_LEN)

    def to_loader(X, y, shuffle):
        Xt = torch.tensor(X).unsqueeze(-1)
        yt = torch.tensor(y).unsqueeze(-1)
        return DataLoader(TensorDataset(Xt, yt),
                          batch_size=BATCH, shuffle=shuffle)

    train_loader = to_loader(Xtr, ytr, True)
    val_loader   = to_loader(Xva, yva, False)

    model     = TCNForecaster(SEQ_LEN, num_channels=32, num_levels=4,
                               kernel_size=3, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=3, factor=0.5)

    best_val_loss = float("inf")
    best_state    = None

    print(f"Training TCN on square {TOP_SQ}...")
    for epoch in range(1, EPOCHS + 1):
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

        # Save checkpoint whenever validation improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone()
                             for k, v in model.state_dict().items()}
            print(f"  epoch {epoch:3d}  val_loss={val_loss:.6f}  ← best")
        else:
            if epoch % 5 == 0:
                print(f"  epoch {epoch:3d}  val_loss={val_loss:.6f}")

    # Load the best weights back into the model
    model.load_state_dict(best_state)

    # Save everything needed to reload the model later
    save_path = OUT_DIR / "best_model_tcn.pt"
    torch.save({
        "model_name":    "TCN",
        "square_id":     TOP_SQ,
        "seq_len":       SEQ_LEN,
        "architecture": {
            "num_channels": 32,
            "num_levels":   4,
            "kernel_size":  3,
            "dropout":      0.1,
        },
        "model_state_dict":  best_state,
        "scaler_mean":       scaler.mean_.tolist(),
        "scaler_scale":      scaler.scale_.tolist(),
        "best_val_loss":     best_val_loss,
        "train_end":         TRAIN_END,
        "val_end":           VAL_END,
    }, save_path)

    print(f"\nModel saved to {save_path}")
    print(f"Best val loss : {best_val_loss:.6f}")
    print(f"File size     : {save_path.stat().st_size / 1024:.1f} KB")
    print("\nTo reload this model later:")
    print("  checkpoint = torch.load('outputs/best_model_tcn.pt')")
    print("  model = TCNForecaster(seq_len=144, num_channels=32,")
    print("                        num_levels=4, kernel_size=3, dropout=0.1)")
    print("  model.load_state_dict(checkpoint['model_state_dict'])")
    print("  model.eval()")


if __name__ == "__main__":
    main()
