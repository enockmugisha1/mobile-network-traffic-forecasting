"""Exploratory data analysis for the Milan internet-traffic dataset.

Produces the four figures and the stationarity result referenced in the report:

    outputs/figures/fig1_traffic_distribution.png   (spatial distribution)
    outputs/figures/fig2_two_week_series.png         (temporal dynamics, 5 areas)
    outputs/figures/fig3_stl_decomposition.png       (seasonal / trend / residual)
    outputs/figures/fig4_acf_pacf.png                (autocorrelation structure)
    outputs/adf_results.txt                          (Augmented Dickey-Fuller test)

All analyses focus on the highest-traffic square (5161) except the spatial
distribution (all 10,000 areas) and the two-week comparison (five areas).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "outputs" / "internet_traffic_dataset.csv"
FIG_DIR = REPO / "outputs" / "figures"
OUT_DIR = REPO / "outputs"

TOP_SQUARE = 5161
REFERENCE_SQUARES = [4159, 4556]      # additional areas required by the brief
TWO_WEEK_START = "2013-11-01"
TWO_WEEK_END = "2013-11-14 23:50:00"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(CSV, parse_dates=["timestamp"], usecols=["square_id", "timestamp", "internet_traffic"])
    return df


def square_series(df: pd.DataFrame, square_id: int) -> pd.Series:
    s = df[df["square_id"] == square_id].set_index("timestamp")["internet_traffic"].sort_index()
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="10min")
    return s.reindex(full_index).ffill().fillna(0.0)


def totals_by_square(df: pd.DataFrame) -> pd.Series:
    return df.groupby("square_id")["internet_traffic"].sum().sort_values(ascending=False)


def fig1_distribution(totals: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(totals.values, bins=80, color="#2e86de", edgecolor="white")
    axes[0].set_title("Total internet traffic per area (linear)")
    axes[0].set_xlabel("Total traffic")
    axes[0].set_ylabel("Number of areas")

    axes[1].hist(np.log10(totals.values + 1), bins=80, color="#8e44ad", edgecolor="white")
    axes[1].set_title("Total internet traffic per area (log10)")
    axes[1].set_xlabel("log10(total traffic)")
    axes[1].set_ylabel("Number of areas")

    fig.suptitle("Figure 1 — Distribution of total internet traffic across 10,000 grid areas")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_traffic_distribution.png", dpi=150)
    plt.close(fig)


def fig2_two_week(df: pd.DataFrame, top_three: list[int]) -> None:
    squares = top_three + REFERENCE_SQUARES
    plt.figure(figsize=(14, 6))
    for square_id in squares:
        s = square_series(df, square_id)[TWO_WEEK_START:TWO_WEEK_END]
        plt.plot(s.index, s.values, label=f"Square {square_id}", linewidth=0.9)
    plt.title("Figure 2 — Internet traffic, first two weeks of Nov 2013 (10-minute resolution)")
    plt.xlabel("Time")
    plt.ylabel("Internet traffic intensity")
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_two_week_series.png", dpi=150)
    plt.close()


def fig3_stl(series: pd.Series) -> float:
    hourly = series.resample("1h").sum()
    stl = STL(hourly, period=24, robust=True).fit()
    seasonal_strength = max(
        0.0, 1.0 - stl.resid.var() / (stl.seasonal + stl.resid).var()
    )

    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(hourly.index, hourly.values, color="#1f3b5b"); axes[0].set_ylabel("Observed")
    axes[1].plot(stl.trend.index, stl.trend.values, color="#c0392b"); axes[1].set_ylabel("Trend")
    axes[2].plot(stl.seasonal.index, stl.seasonal.values, color="#27ae60"); axes[2].set_ylabel("Seasonal")
    axes[3].plot(stl.resid.index, stl.resid.values, color="#7f8c8d"); axes[3].set_ylabel("Residual")
    fig.suptitle(f"Figure 3 — STL decomposition of square {TOP_SQUARE} "
                 f"(hourly, period=24, seasonal strength={seasonal_strength:.4f})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_stl_decomposition.png", dpi=150)
    plt.close(fig)
    return seasonal_strength


def fig4_acf_pacf(series: pd.Series) -> None:
    one_week = series.iloc[: 144 * 7]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    plot_acf(one_week, lags=300, ax=axes[0])
    axes[0].axvline(144, color="red", ls="--", alpha=0.6)
    axes[0].axvline(288, color="red", ls="--", alpha=0.6)
    axes[0].set_title("ACF (lag 144 = 1 day, lag 288 = 2 days marked)")
    plot_pacf(one_week, lags=50, ax=axes[1], method="ywm")
    axes[1].set_title("PACF (first 50 lags)")
    fig.suptitle(f"Figure 4 — Autocorrelation of square {TOP_SQUARE} (10-minute resolution)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_acf_pacf.png", dpi=150)
    plt.close(fig)


def adf_test(series: pd.Series) -> None:
    sample = series.iloc[:5000]
    result = adfuller(sample, autolag="AIC")
    stat, pvalue, used_lag, nobs, crit, _ = result
    lines = [
        "Augmented Dickey-Fuller test — square 5161 (first 5,000 observations)",
        "=" * 60,
        f"ADF statistic : {stat:.4f}",
        f"p-value       : {pvalue:.6f}",
        f"Lags used     : {used_lag}",
        f"Observations  : {nobs}",
        "Critical values:",
    ]
    for level, value in crit.items():
        lines.append(f"  {level:>4} : {value:.4f}")
    verdict = "STATIONARY (reject H0 at 1%)" if stat < crit["1%"] else "NON-STATIONARY"
    lines.append(f"Result        : {verdict}")
    (OUT_DIR / "adf_results.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading dataset...")
    df = load_dataset()

    totals = totals_by_square(df)
    top_three = totals.head(3).index.tolist()
    print("\nTop 10 areas by total internet traffic:")
    print(totals.head(10).to_string())

    fig1_distribution(totals)
    fig2_two_week(df, top_three)

    series = square_series(df, TOP_SQUARE)
    strength = fig3_stl(series)
    fig4_acf_pacf(series)
    adf_test(series)

    print(f"\nSeasonal strength (STL): {strength:.4f}")
    print("Saved EDA figures and adf_results.txt to outputs/.")


if __name__ == "__main__":
    main()
