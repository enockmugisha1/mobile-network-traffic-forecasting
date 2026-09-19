# Milan Mobile Network Traffic Forecasting

This project builds a reproducible forecasting pipeline for the Milan Telecom Traffic 2013 dataset. The workflow is designed for the assignment requirements: data ingestion from sparse text files, exploratory analysis of traffic patterns, iterative model experimentation, and concise reporting.

## Dataset
The raw data is stored in the `mobile-network/` folder and contains daily `.txt` files with sparse rows for multiple activity types. The project focuses on internet traffic rows, identified by activity code `39`.

## Repository structure
- `src/data_pipeline.py` — parses raw TXT files and aggregates internet traffic into a time-series dataset.
- `src/eda.py` — produces traffic summaries and plots for exploratory analysis.
- `src/forecast_experiment.py` — trains and evaluates candidate forecasting models with experiment logging.
- `outputs/` — generated dataset and experiment CSV files plus figures.

## Setup
```bash
cd /home/enock/mobile-network-traffic-forecasting.worktrees/how-on-this-ai-agent-on-this-vscode
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the pipeline
```bash
. .venv/bin/activate
python src/data_pipeline.py
python src/eda.py
python src/forecast_experiment.py
```

## Notes
- The pipeline targets the highest-traffic square by total Internet traffic and builds daily lag features for forecasting.
- The experiment script logs multiple model configurations and saves the results to `outputs/forecast_experiments.csv`.
- The report should be based on the outputs in `outputs/` and the observed validation performance.
