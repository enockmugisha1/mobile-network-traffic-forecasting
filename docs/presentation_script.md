# 7-10 Minute Presentation Script

## 1. Introduction (60–80 seconds)
- Introduce the Milan telecom dataset and the problem of forecasting mobile-Internet demand in urban cells.
- Explain why this matters: it supports capacity planning, service reliability, and anomaly detection.
- State the research question: which forecasting model best captures traffic dynamics for the busiest spatial cells?

## 2. Data handling and preprocessing (90 seconds)
- Describe the raw dataset format: multiple daily sparse text files with activity codes and numerical features.
- Explain that internet traffic was identified as activity code 39 and aggregated by square and timestamp.
- Mention the preprocessing decisions: line-by-line parsing, filtering, timestamp conversion, and square-level aggregation to a compact series.

## 3. Exploratory analysis (60–90 seconds)
- Show that traffic is concentrated in a few squares; the busiest areas dominate the network.
- Mention the most active squares, including 5161, 5059, and 5259, and note the importance of reference areas 4159 and 4556.
- Explain that strong temporal variability and repeated peaks justify forecasting with lag features and calendar variables.

## 4. Methodology and models (2 minutes)
- Describe the daily aggregation and feature creation process.
- Explain the three selected models: Linear Regression, Random Forest, and HistGradientBoostingRegressor.
- Note the iterative experimentation workflow: start with a baseline, then tune the tree-based model and compare the results.

## 5. Results and interpretation (2 minutes)
- Share the model comparison table and highlight the best-performing Random Forest model.
- Explain that the random forest outperformed the linear baseline because the traffic signal contains nonlinear structure.
- Discuss the practical limitation: MAPEs remain moderate, which suggests the task is harder than a simple univariate forecast and would benefit from richer temporal features.

## 6. Important technical decision and limitation (60 seconds)
- Technical decision: using a daily square-level series with lag features rather than a raw minute-by-minute signal reduces noise and makes the modeling pipeline manageable.
- Limitation: the model still struggles with high variance and could be improved with more specialized time-series algorithms or richer training features.

## 7. Conclusion (30–45 seconds)
- Summarize the key message: the project demonstrates that a focused forecasting pipeline can reveal meaningful structure in the telecom traffic data, with the random forest delivering the strongest performance in the tested setup.
- Mention next steps: multi-square forecasting, stronger seasonal features, and alternative time-series models.
