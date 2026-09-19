from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

ACTIVITY_CODE_INTERNET = 39


def parse_daily_file(file_path: str | Path) -> pd.DataFrame:
    """Parse a single daily txt file from the Milan telecom dataset.

    The source files are sparse and contain several activity types. We retain only
    rows related to internet traffic (activity code 39) and collapse each row to a
    single traffic value by summing all numeric observations on the row.
    """
    rows: list[dict[str, float | int]] = []
    file_path = Path(file_path)

    with file_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            parts = [value.strip() for value in line.split("\t")]
            if len(parts) < 3:
                continue

            try:
                square_id = int(parts[0])
                timestamp_ms = int(parts[1])
                activity_code = int(parts[2])
            except ValueError:
                continue

            if activity_code != ACTIVITY_CODE_INTERNET:
                continue

            numeric_values = []
            for value in parts[3:]:
                if not value:
                    continue
                try:
                    numeric_values.append(float(value))
                except ValueError:
                    continue

            if not numeric_values:
                continue

            rows.append(
                {
                    "square_id": square_id,
                    "timestamp_ms": timestamp_ms,
                    "internet_traffic": sum(numeric_values),
                }
            )

    return pd.DataFrame(rows)


def build_internet_dataset(data_dir: str | Path, output_csv: str | Path | None = None) -> pd.DataFrame:
    """Build the aggregated square-level internet traffic dataset."""
    data_dir = Path(data_dir)
    frames: list[pd.DataFrame] = []

    for file_path in sorted(data_dir.glob("*.txt")):
        daily_df = parse_daily_file(file_path)
        if not daily_df.empty:
            frames.append(daily_df)

    if not frames:
        raise FileNotFoundError(f"No dataset files found in {data_dir}")

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.groupby(["square_id", "timestamp_ms"], as_index=False)["internet_traffic"].sum()
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp_ms"], unit="ms")
    dataset = dataset[["square_id", "timestamp", "timestamp_ms", "internet_traffic"]].sort_values(["square_id", "timestamp"])

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_csv, index=False)

    return dataset


def summarize_top_areas(dataset: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Summarize total internet traffic by spatial square."""
    top_areas = (
        dataset.groupby("square_id", as_index=False)["internet_traffic"]
        .sum()
        .rename(columns={"internet_traffic": "total_internet_traffic"})
        .sort_values("total_internet_traffic", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return top_areas


def main() -> None:
    dataset_dir = Path(__file__).resolve().parents[1] / "mobile-network"
    output_csv = Path(__file__).resolve().parents[1] / "outputs" / "internet_traffic_dataset.csv"

    dataset = build_internet_dataset(dataset_dir, output_csv)
    top_areas = summarize_top_areas(dataset, top_n=10)
    print("Rows:", len(dataset))
    print("Squares:", dataset["square_id"].nunique())
    print(top_areas.to_string(index=False))


if __name__ == "__main__":
    main()
