# Milan Telecom Traffic Forecasting

## Introduction
This project studies urban telecommunication demand using the Milan Telecom Traffic 2013 dataset. The objective is to forecast Internet traffic intensity for the most active spatial cells and explain how traffic patterns evolve over time. The problem is practically relevant because accurate demand forecasting supports traffic engineering, capacity planning, and anomaly detection in mobile networks.

## Related Work
Prior network-focused forecasting studies often use time-series and machine-learning models that combine trend, weekly seasonality, and lag features. This project follows that convention by using a daily aggregation and lag-based regression features, while comparing a simple benchmark linear model with nonlinear tree-based learners. The chosen models reflect a common trade-off between interpretability and predictive flexibility for high-variance telecommunication data.

## Dataset and Data Preparation
The raw dataset is stored in the `mobile-network/` directory as daily sparse TXT files. Each row includes a square identifier, a millisecond timestamp, an activity code, and numeric traffic features. Because the assignment focuses on Internet demand, only rows with `activity_code = 39` were retained. Numeric values in each retained row were summed to obtain a single traffic measure per square and timestamp. The resulting dataset was aggregated to a square-level time series and saved to `outputs/internet_traffic_dataset.csv`.

The data volume is large, so memory usage was controlled by parsing line-by-line and using integer-backed timestamps with compact numeric values. After aggregation, the top-square totals were inspected to identify the strongest forecasting targets. The highest-traffic square was `5161`, while `4159` and `4556` were also tracked as additional reference areas.

## Exploratory Analysis
The total traffic by square shows a highly skewed distribution, with a small number of cells carrying much of the network load. The highest totals are concentrated in a few squares, including 5161, 5059, and 5259. The first EDA plot summarizes this concentration, while the daily series plot shows strong temporal variation and recurring peaks that motivate lag-based forecasting.

| Rank | Square ID | Total Internet Traffic |
|---|---:|---:|
| 1 | 5161 | 14,614,418 |
| 2 | 5059 | 13,680,137 |
| 3 | 5259 | 12,259,581 |
| 4 | 5061 | 11,228,350 |
| 5 | 6064 | 10,658,021 |

The selected reference areas also show substantial traffic levels: square `4159` registered 2,755,431 and square `4556` registered 5,310,022 total traffic units.

## Methodology
A daily traffic series was constructed for the highest-traffic square (`5161`) and several lag features were added (1, 2, 3, 7, 14, and 30 days) along with calendar indicators such as day-of-week and month. A time-ordered train/test split was used to preserve the temporal structure of the forecasting task. The experiment followed an iterative design: baseline linear regression was evaluated first, then tree-based models were tuned to improve nonlinear behavior.

The final methodology compares three models:
- Linear Regression as the baseline benchmark.
- Random Forest Regressor as a nonlinear ensemble model capturing local patterns.
- HistGradientBoostingRegressor as a more flexible boosting approach.

## Results and Discussion
The experiment results for square 5161 are summarized below.

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Random Forest (n=300, max_depth=8) | 58,206 | 68,691 | 50.34 |
| Random Forest (n=200, max_depth=None) | 58,869 | 69,790 | 51.10 |
| Linear Regression | 61,880 | 74,288 | 56.44 |
| HistGradientBoostingRegressor | 111,965 | 123,747 | 112.08 |

The best-performing configuration was the random forest model, which achieved the lowest MAE and RMSE. This suggests the series contains nonlinear structure that a simple linear model cannot fully capture. However, the MAPE values remain relatively high, indicating that daily traffic fluctuations are difficult to forecast precisely in a long-horizon setting. The boosting model underperformed in this setup, likely due to the limited number of temporal samples and the need for stronger tuning or a more structured time-series feature set.

## Conclusion and Future Work
The project demonstrates that the Milan telecom traffic series is dominated by a few highly active squares and that a nonlinear ensemble model outperforms the baseline linear benchmark for the selected forecasting target. The main limitation is that the model still struggles with variance and relative error, suggesting that stronger temporal features, longer lookback windows, or time-series-specific models may improve performance.

Future work should extend the analysis to more than one square, add richer lag and seasonal features, and compare against more specialized time-series methods such as SARIMA or Prophet. Additional model tuning with cross-validation and a larger feature set would likely improve generalization and produce a more robust forecasting pipeline.

## References
1. Kaggle: Milan Telecom Traffic 2013 dataset, accessed via the public Kaggle listing.
2. Hyndman, R. and Athanasopoulos, G. Forecasting: Principles and Practice.
3. Scikit-learn documentation for linear regression, random forest regression, and gradient boosting models.
