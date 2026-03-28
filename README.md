# WRDS Forecast Dashboard

Professional `Streamlit` dashboard for auditing, forecasting, and comparing WRDS-style firm-level time-series data with transparent `Naive`, `ARIMA`, and optional `Chronos` workflows.

The application is designed for exploratory analysis, reproducible forecasting comparisons, and diagnostic transparency. It supports both single-entity deep dives and cross-firm analysis on the cleaned dataset, with saved history, live execution logs, forecast interval charts, and exportable outputs for downstream review in tools such as Gretl, Excel, or Python/R notebooks.

## Overview

The dashboard focuses on four goals:

1. Make the preprocessing and forecasting pipeline visible instead of opaque.
2. Compare simple and advanced forecasting methods on the same holdout window.
3. Diagnose non-linearity and structural breaks before interpreting results.
4. Preserve prior runs so analysis can be reloaded without recomputation.

In practice, that means the app lets you upload a WRDS-style dataset, map columns, inspect continuity and exclusions, run single-firm diagnostics, run full cleaned-dataset summaries, compute cross-firm forecast tournaments, and revisit saved historical snapshots later.

## Core Capabilities

### Data ingestion and preparation

- Upload the provided WRDS CSV or any compatible CSV with entity, date, and numeric target columns.
- Auto-detect common WRDS schema patterns and allow manual column remapping.
- Profile row count, entity count, date span, and available numeric metrics.
- Inspect continuity gaps and entity-level eligibility before forecasting.
- Apply optional winsorization to the selected target metric.
- Track target provenance so users can see where the modeled signal came from.

### Single-entity diagnostics

- Run a full diagnostic pipeline for one firm/entity at a time.
- Compare `Naive`, `ARIMA`, and optional `Chronos` forecasts on the same holdout.
- Review `ARIMA` order selection, differencing, residual diagnostics, and candidate search output.
- Compute non-linearity diagnostics using structural breaks, Ljung-Box, `BDS`, and a `Tsay`-style auxiliary test.
- View `Actual vs Prediction`, cumulative error, error-gap, residual, and autocorrelation charts.
- Render Chronos interval views for `50%`, `80%`, `90%`, and `95%` forecast bands when available.
- Display model-comparison callouts that identify whether `ARIMA` or `Chronos` performed better on the holdout.

### Full cleaned-dataset analysis

- Summarize the cleaned panel across all eligible firms.
- Compute full-dataset `NLI` distributions and quartile views.
- Run multi-firm forecast tournaments across a configurable number of firms.
- Compare `Naive`, `ARIMA`, and optional `Chronos` across firms using aggregate metrics and model win rates.
- Inspect full-dataset forecast views, including panel-level actual-vs-predicted aggregates, error distributions, calibration-style comparisons, and per-firm drilldowns from the saved multi-firm run.

### Transparency and reproducibility

- Expose live execution logs and timestamped step status updates in the UI.
- Support `Strict deterministic` and `Practical fallback` evaluation modes.
- Preserve validation outputs, tournament results, and diagnostic artifacts in saved history snapshots.
- Allow loading history snapshots without re-uploading the original dataset.

### History and exports

- Save analysis snapshots to local SQLite history.
- Reload or delete snapshots directly from the sidebar.
- Export forecast tables, diagnostics, entity summaries, and tournament outputs as CSV.
- Reuse saved tournament results and saved distribution results even when no file is currently uploaded.

## Forecasting And Diagnostics Workflow

The dashboard follows a transparent workflow rather than a single black-box forecast call:

1. Upload a dataset and map required fields.
2. Select a target metric such as `ROA` or `ROE`.
3. Review continuity, eligibility, provenance, and optional winsorization.
4. Choose a holdout strategy and evaluation mode.
5. Run either:
   - single-entity diagnostics, or
   - full cleaned-dataset summary and optional tournament views.
6. Review model performance, residual diagnostics, `NLI`, structural breaks, and interval charts.
7. Save the result to history for later inspection.

## Forecasting Modes

### `Strict deterministic`

This mode is designed for audit-oriented runs where deterministic behavior matters more than always returning an answer.

- Uses strict Bai-Perron-style break detection when `R`/`strucchange` is available.
- Enforces the Ljung-Box residual whitening gate.
- Avoids practical fallback behavior when the selected model fails.

### `Practical fallback`

This mode is designed for exploratory work where continuing the workflow is more important than enforcing every gate.

- Uses the more forgiving pipeline behavior.
- Allows fallback logic when `ARIMA` windows fail.
- Is generally a better default when you want results across more firms.

## Chronos Support

`Chronos` is optional but fully integrated when available.

- The app supports optional Chronos forecasting for both single-entity analysis and multi-firm tournament runs.
- Chronos models load from Hugging Face by default if not already cached locally.
- Set `CHRONOS_OFFLINE=1` to force cache-only loading.
- The dashboard displays Chronos point forecasts plus interval bands when the model output supports them.
- Deterministic Chronos runs may produce collapsed intervals; the app still displays the interval charts and explains when the bands lie directly on top of the point forecast.

## Installation

### Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### One-command start

```bash
./run.sh
```

`run.sh` will:

- create `.venv` if it does not exist
- upgrade `pip`
- install Python dependencies from `requirements.txt`
- start Streamlit on `127.0.0.1:8502` by default

Supported environment overrides:

- `PORT=8503 ./run.sh`
- `HOST=0.0.0.0 ./run.sh`
- `PYTHON_BIN=python3.11 ./run.sh`

### Manual launch

```bash
source .venv/bin/activate
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

## System Requirements

### Python dependencies

The application uses the packages listed in `requirements.txt`, including:

- `streamlit`
- `pandas`
- `numpy`
- `plotly`
- `statsmodels`
- `scipy`
- `ruptures`
- `chronos-forecasting`
- `torch`

### Optional R support for strict break detection

Strict Bai-Perron-style break detection requires:

- `R`
- the `strucchange` package

For hosted deployments, the required system packages are listed in `packages.txt`:

- `r-base`
- `r-cran-strucchange`

## Expected Dataset Shape

The dashboard works best when the uploaded file includes:

- an entity identifier such as `gvkey`
- a date column such as `datadate`
- one or more numeric targets such as `ROA`, `ROE`, or similar firm-level metrics

Optional accounting and metadata columns improve diagnostics and summaries:

- `atq`
- `seqq`
- `niq`
- `naics`
- `dlrsn`

These optional fields are not required for basic forecasting, but they improve audit context, exclusions, and full-panel summaries.

## Saved History

The dashboard stores saved snapshots locally so prior work can be reopened without rerunning the entire pipeline.

### Storage

- History database path: `.dashboard_history/history.sqlite3`

### What gets saved

- analysis results
- diagnostics
- audit information
- validation outputs
- `NLI` distribution results
- tournament results
- forecast interval outputs
- execution logs and step statuses

### How history works

- Use the sidebar `History` section to load or delete snapshots.
- History snapshots can be loaded without uploading a file first.
- On a deployed shared app instance, the history list is shared across sessions using the same app storage.
- If the hosting platform wipes local disk during rebuilds or restarts, history may be lost unless you move storage to an external persistent database.

## Deployment Notes

### Streamlit deployment

When deploying, make sure the platform installs:

- Python dependencies from `requirements.txt`
- system packages from `packages.txt` for strict `R` support

### Chronos deployment considerations

- The app attempts to download Chronos models automatically if they are not cached.
- Environments with no outbound network access should either pre-cache models or run with `CHRONOS_OFFLINE=1`.

### History persistence

The current history backend uses a local SQLite file. This is suitable for local use and some simple hosted environments, but not all platforms guarantee persistent local storage between restarts.

## Device And Layout Behavior

The UI now uses responsive stacked layout behavior by default.

- Wide metric rows wrap automatically.
- Side-by-side charts stack more cleanly on smaller screens.
- The main view selector is rendered vertically for better usability on phones and tablets.

The app is designed to remain usable across desktop, tablet, and mobile browsers, with the best experience on desktop or larger tablet screens when reviewing dense tables and multi-chart comparisons.

## Exports

Depending on the current view and saved state, the dashboard can export:

- entity summaries
- naive forecast tables
- `ARIMA` forecast tables
- `Chronos` forecast tables
- `ARIMA` candidate diagnostics
- full-dataset `NLI` distributions
- tournament firm metrics
- tournament firm forecast summaries
- tournament forecast panels

## Project Structure

- `app.py`: main Streamlit application
- `src/arima_pipeline.py`: ARIMA fitting and expanding-window forecasting
- `src/chronos_pipeline.py`: Chronos loading, prediction, and interval extraction
- `src/nli_pipeline.py`: non-linearity and residual-based diagnostics
- `src/tournament_pipeline.py`: multi-firm forecasting tournament logic
- `src/visuals.py`: Plotly figure builders
- `src/history_store.py`: SQLite-backed history persistence
- `run.sh`: local startup helper
- `packages.txt`: deployment-time system packages for strict `R` support

## Troubleshooting

### `Chronos could not run`

Typical causes:

- model download failed
- no network access to Hugging Face
- local environment missing Chronos dependencies

What to check:

- verify internet access or pre-cached models
- unset `CHRONOS_OFFLINE` if you want downloads
- verify `torch` and `chronos-forecasting` installed correctly

### `Strict Bai-Perron mode unavailable`

Typical cause:

- `Rscript` or `strucchange` is not installed

What to check:

- install `R`
- install `strucchange`
- ensure deployment includes `packages.txt`

### History loads but old snapshots behave differently

Older snapshots may not contain newer forecast interval fields or newer tournament outputs. The app attempts to load these snapshots defensively, but some newly added visualizations may only appear for snapshots created after the latest feature updates.

### `ARIMA` results differ from Gretl

This can happen because of:

- different model search ranges
- convergence differences
- practical fallback behavior
- missing-value handling
- different segment selection after break detection

These differences are expected and should be interpreted as implementation differences rather than silent errors.

## Recommended Usage

For the most informative workflow:

1. Start with a single-entity diagnostic run to understand the target and the holdout behavior.
2. Review `NLI`, residual checks, and `ARIMA` candidate search before over-interpreting forecast accuracy.
3. Enable Chronos when you want model comparison and interval views.
4. Use `Tournament Summary` for cross-firm comparison and full-dataset forecast views.
5. Save important runs to history so you can reload them without recomputation.
