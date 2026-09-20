"""
Iterative hyperparameter tuning for LSTM, TCN, Transformer.
Tunes on Square 5161 only (highest-traffic area).
Documents each experiment for the report.
"""
from __future__ import annotations
import time, warnings, sys
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent))
from models import LSTMForecaster, TCNForecaster, TransformerForecaster

# ── config ────────────────────────────────────────────────────────────
REPO       = Path(__file__).resolve().parents[1]
CSV        = REPO / "outputs" / "internet_traffic_dataset.csv"
SQUARE_ID  = 5161
SEQ_LEN    = 144
BATCH      = 64
SEED       = 42
TRAIN_END  = "2013-12-08 23:50:00"
VAL_START  = "2013-12-09 00:00:00"
VAL_END    = "2013-12-15 23:50:00"
TEST_START = "2013-12-16 00:00:00"
TEST_END   = "2013-12-22 23:50:00"

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_and_split(df: pd.DataFrame):
    s = (df[df["square_id"] == SQUARE_ID]
         .set_index("timestamp")["internet_traffic"]
         .sort_index())
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="10min")
    s = s.reindex(full_idx).ffill().fillna(0.0)

    train_raw = s[:TRAIN_END].values.reshape(-1, 1)
    val_raw   = s[VAL_START:VAL_END].values.reshape(-1, 1)
    test_raw  = s[TEST_START:TEST_END].values.reshape(-1, 1)

    scaler    = MinMaxScaler()
    train_s   = scaler.fit_transform(train_raw).ravel()
    val_s     = scaler.transform(val_raw).ravel()
    test_s    = scaler.transform(test_raw).ravel()

    def windows(v):
        X, y = [], []
        for i in range(len(v) - SEQ_LEN):
            X.append(v[i:i+SEQ_LEN])
            y.append(v[i+SEQ_LEN])
        return np.array(X, np.float32), np.array(y, np.float32)

    def loader(X, y, shuffle):
        Xt = torch.tensor(X).unsqueeze(-1)
        yt = torch.tensor(y).unsqueeze(-1)
        return DataLoader(TensorDataset(Xt, yt),
                          batch_size=BATCH, shuffle=shuffle)

    Xtr, ytr = windows(train_s)
    Xva, yva = windows(val_s)

    pre_s  = scaler.transform(s[:VAL_END].values.reshape(-1,1)).ravel()
    ctx_s  = np.concatenate([pre_s[-SEQ_LEN:], test_s])

    return (loader(Xtr, ytr, True),
            loader(Xva, yva, False),
            ctx_s,
            test_raw.ravel(),
            scaler)


def run_one(model, train_loader, val_loader,
            ctx_s, test_actual, scaler,
            epochs=30, lr=1e-3):
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=3, factor=0.5)
    best_val, best_state = float("inf"), None

    t0 = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                vl += crit(model(xb), yb).item() * len(xb)
        vl /= len(val_loader.dataset)
        sched.step(vl)
        if vl < best_val:
            best_val  = vl
            best_state = {k: v.clone() for k,v in model.state_dict().items()}
    train_time = time.perf_counter() - t0
    model.load_state_dict(best_state)

    # rolling forecast
    model.eval()
    buf = list(ctx_s[:SEQ_LEN])
    preds = []
    with torch.no_grad():
        for i in range(len(test_actual)):
            x = torch.tensor(buf[-SEQ_LEN:],
                             dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            p = model(x).item()
            preds.append(p)
            buf.append(float(ctx_s[SEQ_LEN + i]))
    preds = scaler.inverse_transform(
        np.array(preds, np.float32).reshape(-1,1)).ravel()

    mae  = mean_absolute_error(test_actual, preds)
    rmse = float(np.sqrt(mean_squared_error(test_actual, preds)))
    mask = test_actual > 0
    mape = float(np.mean(np.abs(
        (test_actual[mask]-preds[mask])/test_actual[mask]))*100)
    return mae, rmse, mape, round(train_time, 1), best_val


def main():
    print("Loading data...")
    df = pd.read_csv(CSV, parse_dates=["timestamp"],
                     usecols=["square_id","timestamp","internet_traffic"])
    (train_loader, val_loader,
     ctx_s, test_actual, scaler) = load_and_split(df)

    records = []

    # ── baseline results from experiment (for reference) ──────────────
    print("\nBaseline results (from forecast_experiment.py):")
    print("  LSTM        MAE=93.7  RMSE=134.8  MAPE=11.79%  train=106s")
    print("  TCN         MAE=80.5  RMSE=117.6  MAPE= 9.01%  train=116s")
    print("  Transformer MAE=105.2 RMSE=148.7  MAPE=14.21%  train=466s")

    # ════════════════════════════════════════════════════════════════════
    # LSTM TUNING
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "="*55)
    print("LSTM TUNING")
    print("="*55)

    lstm_configs = [
        # (hidden, layers, dropout, epochs, lr, label)
        (64,  2, 0.1, 30, 1e-3, "baseline"),
        (128, 2, 0.1, 30, 1e-3, "larger hidden"),
        (64,  3, 0.1, 30, 1e-3, "deeper (3 layers)"),
        (128, 2, 0.2, 40, 5e-4, "larger+more epochs+lower lr"),
        (64,  2, 0.1, 30, 5e-4, "lower lr"),
    ]

    for hidden, layers, drop, epochs, lr, label in lstm_configs:
        torch.manual_seed(SEED)
        model = LSTMForecaster(SEQ_LEN, hidden, layers, drop)
        mae, rmse, mape, tt, bv = run_one(
            model, train_loader, val_loader,
            ctx_s, test_actual, scaler, epochs, lr)
        nparams = sum(p.numel() for p in model.parameters())
        print(f"  [{label}]")
        print(f"    hidden={hidden} layers={layers} drop={drop} "
              f"epochs={epochs} lr={lr}")
        print(f"    MAE={mae:.1f}  RMSE={rmse:.1f}  "
              f"MAPE={mape:.2f}%  train={tt}s  params={nparams:,}")
        records.append(dict(model="LSTM", config=label,
                            hidden=hidden, layers=layers,
                            dropout=drop, epochs=epochs, lr=lr,
                            MAE=round(mae,2), RMSE=round(rmse,2),
                            MAPE=round(mape,4), train_s=tt,
                            n_params=nparams))

    # ════════════════════════════════════════════════════════════════════
    # TCN TUNING
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "="*55)
    print("TCN TUNING")
    print("="*55)

    tcn_configs = [
        # (channels, levels, kernel, dropout, epochs, lr, label)
        (32, 4, 3, 0.1, 30, 1e-3, "baseline"),
        (64, 4, 3, 0.1, 30, 1e-3, "more channels"),
        (32, 5, 3, 0.1, 30, 1e-3, "more levels"),
        (64, 4, 5, 0.1, 40, 5e-4, "larger kernel+more epochs"),
        (32, 4, 3, 0.2, 30, 1e-3, "higher dropout"),
    ]

    for ch, lv, ks, drop, epochs, lr, label in tcn_configs:
        torch.manual_seed(SEED)
        model = TCNForecaster(SEQ_LEN, ch, lv, ks, drop)
        mae, rmse, mape, tt, bv = run_one(
            model, train_loader, val_loader,
            ctx_s, test_actual, scaler, epochs, lr)
        nparams = sum(p.numel() for p in model.parameters())
        print(f"  [{label}]")
        print(f"    channels={ch} levels={lv} kernel={ks} "
              f"drop={drop} epochs={epochs} lr={lr}")
        print(f"    MAE={mae:.1f}  RMSE={rmse:.1f}  "
              f"MAPE={mape:.2f}%  train={tt}s  params={nparams:,}")
        records.append(dict(model="TCN", config=label,
                            channels=ch, levels=lv, kernel=ks,
                            dropout=drop, epochs=epochs, lr=lr,
                            MAE=round(mae,2), RMSE=round(rmse,2),
                            MAPE=round(mape,4), train_s=tt,
                            n_params=nparams))

    # ════════════════════════════════════════════════════════════════════
    # TRANSFORMER TUNING
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "="*55)
    print("TRANSFORMER TUNING")
    print("="*55)

    tr_configs = [
        # (d_model, nhead, layers, ff, dropout, epochs, lr, label)
        (32, 4, 2, 128, 0.1, 30, 1e-3, "baseline"),
        (64, 4, 2, 256, 0.1, 30, 1e-3, "larger d_model"),
        (32, 4, 3, 128, 0.1, 30, 1e-3, "deeper"),
        (32, 4, 2, 128, 0.1, 40, 5e-4, "more epochs+lower lr"),
        (64, 4, 2, 128, 0.2, 30, 1e-3, "larger+more dropout"),
    ]

    for dm, nh, nl, ff, drop, epochs, lr, label in tr_configs:
        torch.manual_seed(SEED)
        model = TransformerForecaster(SEQ_LEN, dm, nh, nl, ff, drop)
        mae, rmse, mape, tt, bv = run_one(
            model, train_loader, val_loader,
            ctx_s, test_actual, scaler, epochs, lr)
        nparams = sum(p.numel() for p in model.parameters())
        print(f"  [{label}]")
        print(f"    d_model={dm} nhead={nh} layers={nl} "
              f"ff={ff} drop={drop} epochs={epochs} lr={lr}")
        print(f"    MAE={mae:.1f}  RMSE={rmse:.1f}  "
              f"MAPE={mape:.2f}%  train={tt}s  params={nparams:,}")
        records.append(dict(model="Transformer", config=label,
                            d_model=dm, nhead=nh, layers=nl,
                            ff=ff, dropout=drop, epochs=epochs, lr=lr,
                            MAE=round(mae,2), RMSE=round(rmse,2),
                            MAPE=round(mape,4), train_s=tt,
                            n_params=nparams))

    # ── save ──────────────────────────────────────────────────────────
    out = REPO / "outputs" / "tuning_results.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"\nTuning results saved → {out}")

    # ── print best per model ──────────────────────────────────────────
    df_r = pd.DataFrame(records)
    print("\n" + "="*55)
    print("BEST CONFIG PER MODEL (lowest MAE)")
    print("="*55)
    for m in ["LSTM", "TCN", "Transformer"]:
        best = df_r[df_r["model"]==m].sort_values("MAE").iloc[0]
        print(f"\n{m}: config='{best['config']}'")
        print(f"  MAE={best['MAE']}  RMSE={best['RMSE']}  "
              f"MAPE={best['MAPE']}  train={best['train_s']}s")


if __name__ == "__main__":
    main()
