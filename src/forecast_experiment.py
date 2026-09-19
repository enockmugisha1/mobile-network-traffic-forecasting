from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid


def load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    dataset = pd.read_csv(dataset_path)
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"])
    return dataset


def build_daily_series(dataset: pd.DataFrame, square_id: int) -> pd.DataFrame:
    series = dataset[dataset["square_id"] == square_id].copy()
    series = series.sort_values("timestamp")
    daily = (
        series.groupby(pd.Grouper(key="timestamp", freq="D"), as_index=False)["internet_traffic"]
        .sum()
        .rename(columns={"timestamp": "date", "internet_traffic": "traffic"})
    )
    return daily


def make_feature_frame(series: pd.DataFrame, lag_values: tuple[int, ...] = (1, 2, 3, 7, 14, 30)) -> pd.DataFrame:
    feature_frame = series.copy()
    for lag in lag_values:
        feature_frame[f"lag_{lag}"] = feature_frame["traffic"].shift(lag)

    feature_frame["day_of_week"] = feature_frame["date"].dt.dayofweek
    feature_frame["month"] = feature_frame["date"].dt.month
    feature_frame["day_of_month"] = feature_frame["date"].dt.day
    feature_frame = feature_frame.dropna().reset_index(drop=True)
    return feature_frame


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        mape = np.nan
    else:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mape),
    }


def run_experiment(series: pd.DataFrame, model_name: str, params: dict[str, object]) -> dict[str, object]:
    feature_frame = make_feature_frame(series)
    last_index = len(feature_frame)
    split_index = int(last_index * 0.8)

    X_train = feature_frame.iloc[:split_index].drop(columns=["date", "traffic"])
    y_train = feature_frame.iloc[:split_index]["traffic"]
    X_test = feature_frame.iloc[split_index:].drop(columns=["date", "traffic"])
    y_test = feature_frame.iloc[split_index:]["traffic"]

    if model_name == "linear_regression":
        model = LinearRegression(**params)
    elif model_name == "random_forest":
        model = RandomForestRegressor(random_state=42, **params)
    elif model_name == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(random_state=42, **params)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = evaluate_predictions(y_test, predictions)

    return {
        "model": model_name,
        "params": params,
        "metrics": metrics,
        "prediction": predictions,
        "actual": y_test.reset_index(drop=True),
    }


def run_grid_search(dataset: pd.DataFrame) -> pd.DataFrame:
    top_square = (
        dataset.groupby("square_id", as_index=False)["internet_traffic"]
        .sum()
        .sort_values("internet_traffic", ascending=False)
        .iloc[0]["square_id"]
    )
    series = build_daily_series(dataset, int(top_square))

    model_params = {
        "linear_regression": [{},],
        "random_forest": [
            {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
            {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 1},
            {"n_estimators": 400, "max_depth": 10, "min_samples_leaf": 2},
        ],
        "hist_gradient_boosting": [
            {"max_depth": 3, "learning_rate": 0.05, "max_iter": 250},
            {"max_depth": 6, "learning_rate": 0.03, "max_iter": 400},
            {"max_depth": 8, "learning_rate": 0.05, "max_iter": 500},
        ],
    }

    rows = []
    for model_name, param_grid in model_params.items():
        for params in param_grid:
            result = run_experiment(series, model_name, params)
            record = {
                "model": model_name,
                "params": str(params),
                "mae": result["metrics"]["mae"],
                "rmse": result["metrics"]["rmse"],
                "mape": result["metrics"]["mape"],
                "square_id": int(top_square),
            }
            rows.append(record)

    results = pd.DataFrame(rows)
    results = results.sort_values(["mape", "rmse", "mae"], ascending=True).reset_index(drop=True)
    return results


def main() -> None:
    dataset_path = Path(__file__).resolve().parents[1] / "outputs" / "internet_traffic_dataset.csv"
    dataset = load_dataset(dataset_path)
    results = run_grid_search(dataset)
    print(results.to_string(index=False))

    output_path = Path(__file__).resolve().parents[1] / "outputs" / "forecast_experiments.csv"
    results.to_csv(output_path, index=False)
    print(f"Saved experiment results to {output_path}")


if __name__ == "__main__":
    main()
