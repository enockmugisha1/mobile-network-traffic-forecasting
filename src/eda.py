from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    dataset = pd.read_csv(dataset_path)
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"])
    return dataset


def plot_top_areas(dataset: pd.DataFrame, top_n: int = 10, output_dir: str | Path = "outputs/figures") -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top = (
        dataset.groupby("square_id", as_index=False)["internet_traffic"]
        .sum()
        .rename(columns={"internet_traffic": "total_internet_traffic"})
        .sort_values("total_internet_traffic", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top,
        x="square_id",
        y="total_internet_traffic",
        hue="square_id",
        palette="viridis",
        dodge=False,
        legend=False,
    )
    plt.title(f"Top {top_n} squares by total internet traffic")
    plt.xlabel("Square ID")
    plt.ylabel("Total traffic")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "top_areas_by_total_traffic.png", dpi=200)
    plt.close()

    return top


def plot_selected_series(dataset: pd.DataFrame, squares: list[int], output_dir: str | Path = "outputs/figures") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered = dataset[dataset["square_id"].isin(squares)].copy()
    filtered["timestamp"] = pd.to_datetime(filtered["timestamp"])
    daily = (
        filtered.groupby(["square_id", pd.Grouper(key="timestamp", freq="D")], as_index=False)["internet_traffic"]
        .sum()
        .rename(columns={"timestamp": "date"})
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    for square_id in squares:
        square_daily = daily[daily["square_id"] == square_id]
        plt.plot(square_daily["date"], square_daily["internet_traffic"], label=f"Square {square_id}")

    plt.title("Daily internet traffic for selected squares")
    plt.xlabel("Date")
    plt.ylabel("Daily traffic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "selected_square_daily_traffic.png", dpi=200)
    plt.close()


def main() -> None:
    dataset_path = Path(__file__).resolve().parents[1] / "outputs" / "internet_traffic_dataset.csv"
    dataset = load_dataset(dataset_path)
    top_areas = plot_top_areas(dataset, top_n=10)
    print(top_areas.head(10).to_string(index=False))

    relevant_squares = [4159, 4556, int(top_areas.iloc[0]["square_id"])]
    plot_selected_series(dataset, relevant_squares)
    print("Saved EDA figures to outputs/figures/")


if __name__ == "__main__":
    main()
