# WRDS Forecast Dashboard

Local `Streamlit` dashboard for uploading WRDS-style firm time-series data, validating the pipeline, and comparing `Naive`, `ARIMA`, and `Chronos` forecasts with transparent diagnostics, audit controls, and saved history.

## Features

- Upload the provided WRDS CSV or other compatible CSV files.
- Auto-detect common WRDS columns and allow manual schema mapping.
- Inspect entity counts, date coverage, numeric targets, and continuity gaps.
- Verify `ARIMA` differencing and model selection directly from the UI.
- Run single-entity diagnostics with naive baseline, `ARIMA`, and optional `Chronos`.
- Run residual-based `NLI` diagnostics with structural-break handling, Ljung-Box, `BDS`, and a `Tsay`-style auxiliary test.
- Switch between practical fallback mode and strict deterministic mode.
- Compare `Actual vs Prediction` for `Naive`, `ARIMA`, and `Chronos`.
- Render live interactive charts with `Plotly`.
- Review audit information including eligibility, target provenance, and winsorization.
- Run cross-firm tournament summaries, quartile views, and uncertainty charts.
- View live execution logs while analysis is running.
- Save and reload dashboard history for analysis, audit, validation, NLI distribution, tournament, and uncertainty views.
- Export forecast tables and diagnostic summaries for Gretl comparison.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## One Command Start

```bash
./run.sh
```

This creates `.venv` if needed, installs dependencies from `requirements.txt`, and starts the dashboard on `127.0.0.1:8502`.

Environment overrides:

- `PORT=8503 ./run.sh`
- `HOST=0.0.0.0 ./run.sh`
- `PYTHON_BIN=python3.11 ./run.sh`

## Run

```bash
source .venv/bin/activate
streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

## Expected Dataset Shape

The dashboard works best when the upload contains:

- a firm identifier such as `gvkey`
- a date column such as `datadate`
- one or more numeric target metrics such as `ROA` or `ROE`

Optional accounting fields such as `atq`, `seqq`, `niq`, `naics`, and `dlrsn` improve diagnostics and summaries but are not mandatory for basic forecasting.

## Saved History

The dashboard stores saved snapshots locally so you can reopen past work without recomputing it.

- Storage path: `.dashboard_history/history.sqlite3`
- Saved content includes analysis results, diagnostics, audit data, validation output, NLI distributions, tournament results, uncertainty summaries, and execution logs.
- Use the sidebar `History` section to load or delete previous snapshots.

## Notes

- `Chronos` is optional. If the dependency is unavailable locally, the dashboard still runs the rest of the pipeline and reports the missing dependency in the sidebar.
- Strict break-detection mode requires local `R` plus the `strucchange` package.
- `ARIMA` results may differ from Gretl because of search ranges, convergence behavior, and missing-value handling. Those differences are intentional and documented in the UI exports.
