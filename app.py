from __future__ import annotations

from datetime import datetime
from typing import Dict
import warnings

import pandas as pd
import streamlit as st

from src.arima_pipeline import (
    expanding_window_forecast as run_arima_forecast,
    naive_expanding_window_forecast as run_naive_forecast,
)
from src.break_detection import exact_break_detection_status
from src.chronos_pipeline import chronos_import_status, expanding_window_forecast as run_chronos_forecast
from src.data_pipeline import (
    apply_target_winsorization,
    build_dataset_profile,
    build_entity_eligibility_summary,
    build_entity_options,
    build_target_series,
    compute_continuity_summary,
    describe_target_provenance,
    detect_metric_columns,
    load_csv,
    prepare_dataset,
    resolve_entity_id,
    summarize_entities,
)
from src.history_store import (
    delete_history_snapshot,
    list_history_entries,
    load_history_snapshot,
    save_history_snapshot,
)
from src.nli_pipeline import compute_nli, compute_nli_distribution
from src.schema_mapper import CANONICAL_FIELDS, build_mapping_options, describe_mapping, infer_schema_mapping, validate_mapping
from src.tournament_pipeline import compute_forecasting_tournament
from src.visuals import (
    build_combined_forecast_figure,
    build_cumulative_error_figure,
    build_error_gap_figure,
    build_eligibility_figure,
    build_forecast_figure,
    build_nli_distribution_figure,
    build_nli_quartile_figure,
    build_residual_autocorrelation_figure,
    build_residual_figure,
    build_tournament_gap_scatter,
    build_tournament_leaderboard_figure,
    build_tournament_quartile_win_figure,
    build_uncertainty_width_nli_figure,
    build_volatility_gap_figure,
    build_coverage_quartile_figure,
    build_winner_distribution_figure,
)
from src.validation import arima_reference_summary, compare_prediction_frames

try:
    from urllib3.exceptions import NotOpenSSLWarning

    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

_EXECUTION_STEP_PLACEHOLDER = None
_EXECUTION_LOG_PLACEHOLDER = None


st.set_page_config(page_title="WRDS Forecast Dashboard", layout="wide")
st.title("WRDS Forecast Dashboard")
st.caption(
    "Upload the WRDS CSV or another compatible time-series dataset, map columns, "
    "and compare transparent ARIMA diagnostics against Chronos forecasts."
)


@st.cache_data(show_spinner=False)
def cached_load_csv(upload_bytes: bytes) -> pd.DataFrame:
    return load_csv(upload_bytes)


@st.cache_data(show_spinner=False)
def cached_prepare_dataset(upload_bytes: bytes, mapping_items: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    frame = load_csv(upload_bytes)
    mapping = dict(mapping_items)
    return prepare_dataset(frame, mapping)


if "nli_distribution_result" not in st.session_state:
    st.session_state["nli_distribution_result"] = None
if "nli_distribution_meta" not in st.session_state:
    st.session_state["nli_distribution_meta"] = None
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None
if "analysis_meta" not in st.session_state:
    st.session_state["analysis_meta"] = None
if "dashboard_view" not in st.session_state:
    st.session_state["dashboard_view"] = "Forecasts"
if "execution_log" not in st.session_state:
    st.session_state["execution_log"] = []
if "step_statuses" not in st.session_state:
    st.session_state["step_statuses"] = []
if "validation_result" not in st.session_state:
    st.session_state["validation_result"] = None
if "validation_meta" not in st.session_state:
    st.session_state["validation_meta"] = None
if "tournament_result" not in st.session_state:
    st.session_state["tournament_result"] = None
if "tournament_meta" not in st.session_state:
    st.session_state["tournament_meta"] = None
if "history_loaded" not in st.session_state:
    st.session_state["history_loaded"] = False
if "history_loaded_label" not in st.session_state:
    st.session_state["history_loaded_label"] = ""


def render_profile(profile) -> None:
    left, middle, right, extra = st.columns(4)
    left.metric("Rows", f"{profile.row_count:,}")
    middle.metric("Entities", f"{profile.entity_count:,}")
    right.metric("Columns", f"{profile.column_count:,}")
    extra.metric(
        "Date Range",
        f"{profile.date_min.date() if profile.date_min is not None else 'N/A'} -> "
        f"{profile.date_max.date() if profile.date_max is not None else 'N/A'}",
    )

    if profile.warnings:
        for warning in profile.warnings:
            st.warning(warning)


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def reset_execution_console() -> None:
    st.session_state["execution_log"] = []
    st.session_state["step_statuses"] = []
    _refresh_execution_console()


def append_execution_log(message: str) -> None:
    st.session_state["execution_log"].append(f"[{_timestamp()}] {message}")
    _refresh_execution_console()


def update_step_status(step_name: str, status: str, detail: str = "") -> None:
    steps = [step for step in st.session_state["step_statuses"] if step["step"] != step_name]
    steps.append(
        {
            "step": step_name,
            "status": status,
            "updated_at": _timestamp(),
            "detail": detail,
        }
    )
    st.session_state["step_statuses"] = steps
    _refresh_execution_console()


def _refresh_execution_console() -> None:
    if _EXECUTION_STEP_PLACEHOLDER is not None:
        if st.session_state["step_statuses"]:
            _EXECUTION_STEP_PLACEHOLDER.dataframe(
                pd.DataFrame(st.session_state["step_statuses"]),
                width="stretch",
                hide_index=True,
            )
        else:
            _EXECUTION_STEP_PLACEHOLDER.caption("No steps yet.")
    if _EXECUTION_LOG_PLACEHOLDER is not None:
        if st.session_state["execution_log"]:
            _EXECUTION_LOG_PLACEHOLDER.code("\n".join(st.session_state["execution_log"][-200:]), language="text")
        else:
            _EXECUTION_LOG_PLACEHOLDER.caption("No execution log yet. Run an analysis to populate this console.")


def render_execution_console() -> None:
    global _EXECUTION_STEP_PLACEHOLDER, _EXECUTION_LOG_PLACEHOLDER
    with st.expander("Execution Console", expanded=True):
        _EXECUTION_STEP_PLACEHOLDER = st.empty()
        _EXECUTION_LOG_PLACEHOLDER = st.empty()
        _refresh_execution_console()


def render_mapping_editor(columns: list[str], inferred_mapping: Dict[str, str]) -> Dict[str, str]:
    st.subheader("Schema Mapping")
    st.caption("Auto-detection is prefilled. Adjust anything that does not match your file.")
    mapping: Dict[str, str] = {}
    options = build_mapping_options(columns)

    for field in CANONICAL_FIELDS:
        default_value = inferred_mapping.get(field.name, "")
        default_index = options.index(default_value) if default_value in options else 0
        mapping[field.name] = st.selectbox(
            f"{field.name} {'*' if field.required else ''}",
            options,
            index=default_index,
            help=field.description,
            key=f"mapping_{field.name}",
        )

    return mapping


def build_model_gap(arima_predictions: pd.DataFrame, chronos_predictions: pd.DataFrame) -> pd.DataFrame:
    if arima_predictions.empty or chronos_predictions.empty:
        return pd.DataFrame()
    merged = arima_predictions[["date", "actual", "error"]].rename(columns={"error": "arima_error"}).merge(
        chronos_predictions[["date", "error"]].rename(columns={"error": "chronos_error"}),
        on="date",
        how="inner",
    )
    merged["error_gap"] = merged["arima_error"].abs() - merged["chronos_error"].abs()
    return merged


def infer_document_holdout_size(series_frame: pd.DataFrame) -> int | None:
    working = series_frame.sort_values("date").reset_index(drop=True)
    if working.empty:
        return None
    mask = (working["date"] >= pd.Timestamp("2023-01-01")) & (working["date"] <= pd.Timestamp("2024-12-31"))
    if mask.sum() < 2:
        return None
    first_idx = int(mask.idxmax())
    if first_idx <= 0:
        return None
    if first_idx != len(working) - int(mask.sum()):
        return None
    return int(mask.sum())


def render_quality_badges(labels: list[str]) -> None:
    if not labels:
        return
    st.caption(" | ".join(f"`{label}`" for label in labels))


def build_history_snapshot_payload() -> dict:
    return {
        "analysis_result": st.session_state.get("analysis_result"),
        "analysis_meta": st.session_state.get("analysis_meta"),
        "nli_distribution_result": st.session_state.get("nli_distribution_result"),
        "nli_distribution_meta": st.session_state.get("nli_distribution_meta"),
        "validation_result": st.session_state.get("validation_result"),
        "validation_meta": st.session_state.get("validation_meta"),
        "tournament_result": st.session_state.get("tournament_result"),
        "tournament_meta": st.session_state.get("tournament_meta"),
        "dashboard_view": st.session_state.get("dashboard_view"),
        "execution_log": st.session_state.get("execution_log"),
        "step_statuses": st.session_state.get("step_statuses"),
    }


def save_current_snapshot(snapshot_type: str, label: str) -> None:
    save_history_snapshot(
        build_history_snapshot_payload(),
        snapshot_type=snapshot_type,
        label=label,
        analysis_meta=st.session_state.get("analysis_meta"),
    )
    st.session_state["history_loaded"] = False
    st.session_state["history_loaded_label"] = ""


with st.sidebar:
    st.header("Upload")
    uploaded_file = st.file_uploader("CSV dataset", type=["csv"])
    chronos_available, chronos_message = chronos_import_status()
    if chronos_available:
        st.success("Chronos dependency detected.")
    else:
        st.info(f"Chronos unavailable locally: {chronos_message}")
    strict_breaks_available, strict_breaks_message = exact_break_detection_status()
    if strict_breaks_available:
        st.success("Strict Bai-Perron mode available via R `strucchange`.")
    else:
        st.warning(f"Strict Bai-Perron mode unavailable: {strict_breaks_message}")
    with st.expander("History", expanded=False):
        history_entries = list_history_entries()
        if history_entries.empty:
            st.caption("No saved history yet.")
        else:
            history_entries = history_entries.copy()
            history_entries["display_label"] = (
                history_entries["created_at"]
                + " | "
                + history_entries["snapshot_type"]
                + " | "
                + history_entries["label"]
            )
            selected_label = st.selectbox("Saved snapshots", history_entries["display_label"].tolist(), key="history_select")
            selected_row = history_entries.loc[history_entries["display_label"] == selected_label].iloc[0]
            load_history = st.button("Load Snapshot")
            delete_history = st.button("Delete Snapshot")
            if load_history:
                snapshot = load_history_snapshot(selected_row["id"])
                st.session_state["analysis_result"] = snapshot.get("analysis_result")
                st.session_state["analysis_meta"] = snapshot.get("analysis_meta")
                st.session_state["nli_distribution_result"] = snapshot.get("nli_distribution_result")
                st.session_state["nli_distribution_meta"] = snapshot.get("nli_distribution_meta")
                st.session_state["validation_result"] = snapshot.get("validation_result")
                st.session_state["validation_meta"] = snapshot.get("validation_meta")
                st.session_state["tournament_result"] = snapshot.get("tournament_result")
                st.session_state["tournament_meta"] = snapshot.get("tournament_meta")
                st.session_state["dashboard_view"] = snapshot.get("dashboard_view", "Forecasts")
                st.session_state["execution_log"] = snapshot.get("execution_log", [])
                st.session_state["step_statuses"] = snapshot.get("step_statuses", [])
                st.session_state["history_loaded"] = True
                st.session_state["history_loaded_label"] = selected_row["label"]
                st.rerun()
            if delete_history:
                delete_history_snapshot(selected_row["id"])
                if st.session_state.get("history_loaded_label") == selected_row["label"]:
                    st.session_state["history_loaded"] = False
                    st.session_state["history_loaded_label"] = ""
                st.rerun()

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

upload_bytes = uploaded_file.getvalue()
raw_frame = cached_load_csv(upload_bytes)
schema_inference = infer_schema_mapping(raw_frame.columns)

mapping = render_mapping_editor(list(raw_frame.columns), schema_inference.mapping)
mapping_errors = validate_mapping(mapping, raw_frame.columns)
if mapping_errors:
    for error in mapping_errors:
        st.error(error)
    st.stop()

prepared = cached_prepare_dataset(upload_bytes, tuple(sorted(mapping.items())))
profile = build_dataset_profile(prepared, warnings=schema_inference.warnings)
render_profile(profile)

with st.expander("Detected Mapping", expanded=False):
    st.dataframe(describe_mapping(mapping), width="stretch")

metric_columns = detect_metric_columns(prepared)
if not metric_columns:
    st.error("No numeric target columns were detected after applying the schema mapping.")
    st.stop()

overview_left, overview_right = st.columns([2, 1])
with overview_left:
    analysis_scope = st.selectbox(
        "Analysis scope",
        ["Single entity diagnostics", "Full cleaned dataset summary"],
    )
    target_column = st.selectbox(
        "Target metric",
        metric_columns,
        index=metric_columns.index("ROA") if "ROA" in metric_columns else 0,
    )
    entity_options = build_entity_options(prepared)
    entity_label = st.selectbox("Firm / entity", entity_options)

entity_id = resolve_entity_id(prepared, entity_label)
document_holdout_size = infer_document_holdout_size(
    build_target_series(prepared, entity_id, target_column)
)

with overview_right:
    evaluation_mode_label = st.selectbox(
        "Evaluation mode",
        ["Strict deterministic", "Practical fallback"],
    )
    evaluation_mode = "strict" if evaluation_mode_label == "Strict deterministic" else "practical"
    if evaluation_mode == "strict" and not strict_breaks_available:
        st.caption("Strict mode requires R `strucchange`, which is currently unavailable.")
    holdout_policy = st.selectbox(
        "Holdout policy",
        ["Document default (2023-2024 when available)", "Manual holdout quarters"],
    )
    if holdout_policy == "Document default (2023-2024 when available)" and document_holdout_size is not None:
        holdout_size = int(document_holdout_size)
        st.caption(f"Using document-aligned holdout of `{holdout_size}` quarter(s) covering the final 2023-2024 window.")
    else:
        if holdout_policy == "Document default (2023-2024 when available)" and document_holdout_size is None:
            st.caption("Document-aligned 2023-2024 holdout is unavailable for this firm, so manual holdout is being used.")
        holdout_size = st.slider("Holdout quarters", min_value=2, max_value=12, value=8)
    max_p = st.slider("Max AR order (p)", min_value=0, max_value=4, value=2)
    max_d = st.slider("Max differencing order (d)", min_value=0, max_value=2, value=2)
    max_q = st.slider("Max MA order (q)", min_value=0, max_value=4, value=2)
    run_chronos = st.checkbox("Run Chronos", value=False)
    chronos_deterministic = st.checkbox("Chronos deterministic inference", value=True, disabled=not run_chronos)
    chronos_seed = int(
        st.number_input("Chronos seed", min_value=0, max_value=999999, value=17, step=1, disabled=not run_chronos)
    )
    chronos_samples = int(
        st.slider("Chronos samples", min_value=5, max_value=100, value=20, step=5, disabled=not run_chronos or chronos_deterministic)
    )

with st.expander("Audit Controls", expanded=False):
    enable_winsorization = st.checkbox("Enable winsorization for target metric", value=False)
    winsor_lower = float(st.slider("Winsor lower percentile", min_value=0.0, max_value=0.1, value=0.01, step=0.01))
    winsor_upper = float(st.slider("Winsor upper percentile", min_value=0.9, max_value=1.0, value=0.99, step=0.01))
    minimum_history = int(st.slider("Minimum usable history for audit checks", min_value=8, max_value=40, value=10))

analysis_frame, winsor_report = apply_target_winsorization(
    prepared,
    target_column,
    enabled=enable_winsorization,
    lower_quantile=winsor_lower,
    upper_quantile=winsor_upper,
)
target_provenance = describe_target_provenance(analysis_frame, target_column)
eligibility_summary = build_entity_eligibility_summary(analysis_frame, target_column, min_history=minimum_history)
entity_series = build_target_series(analysis_frame, entity_id, target_column)

with st.expander("Audit Summary", expanded=False):
    audit_left, audit_middle, audit_right = st.columns(3)
    audit_left.metric("Eligible firms", str(int(eligibility_summary["eligible"].sum())))
    audit_middle.metric("Excluded firms", str(int((~eligibility_summary["eligible"]).sum())))
    audit_right.metric("Winsorized rows", str(winsor_report.changed_rows))
    st.write(
        {
            "target_provenance": target_provenance.source_type,
            "provenance_detail": target_provenance.detail,
            "winsorization_detail": winsor_report.detail,
            "minimum_history_rule": minimum_history,
        }
    )
    st.dataframe(eligibility_summary.head(50), width="stretch")

if not st.session_state.get("history_loaded") and analysis_scope == "Single entity diagnostics" and len(entity_series) < holdout_size + 10:
    st.error("The selected series is too short for the requested holdout and ARIMA diagnostics.")
    st.stop()

continuity = compute_continuity_summary(entity_series)
with st.expander("Series Continuity", expanded=False):
    st.dataframe(continuity, width="stretch")

render_execution_console()

analysis_meta = {
    "analysis_scope": analysis_scope,
    "entity_id": entity_id,
    "entity_label": entity_label,
    "target_column": target_column,
    "holdout_size": int(holdout_size),
    "holdout_policy": holdout_policy,
    "evaluation_mode": evaluation_mode,
    "max_p": int(max_p),
    "max_d": int(max_d),
    "max_q": int(max_q),
    "run_chronos": bool(run_chronos),
    "chronos_deterministic": bool(chronos_deterministic),
    "chronos_seed": int(chronos_seed),
    "chronos_samples": int(chronos_samples),
    "winsorization_enabled": bool(enable_winsorization),
    "winsor_lower": float(winsor_lower),
    "winsor_upper": float(winsor_upper),
    "minimum_history": int(minimum_history),
    "target_provenance": target_provenance.source_type,
    "series_length": int(len(entity_series)),
    "series_end": entity_series["date"].max().isoformat() if not entity_series.empty else "",
    "strict_breaks_available": bool(strict_breaks_available),
}

run_analysis = st.button("Run Analysis", type="primary")
cached_analysis_meta = st.session_state.get("analysis_meta")
history_loaded = bool(st.session_state.get("history_loaded"))
has_matching_analysis = (
    (cached_analysis_meta == analysis_meta and st.session_state.get("analysis_result") is not None)
    or (history_loaded and st.session_state.get("analysis_result") is not None)
)

if run_analysis:
    try:
        reset_execution_console()
        append_execution_log(f"Analysis requested for scope `{analysis_scope}` on target `{target_column}`.")
        status = st.status("Running analysis...", expanded=True)
        status.write(f"Scope: `{analysis_scope}`")
        status.write(f"Target metric: `{target_column}`")
        if analysis_scope == "Single entity diagnostics":
            if evaluation_mode == "strict" and not strict_breaks_available:
                raise RuntimeError(f"Strict mode requires R `strucchange`: {strict_breaks_message}")
            update_step_status("Select entity context", "completed", entity_label)
            status.write(f"Selected entity: `{entity_label}`")
            append_execution_log(f"Selected entity `{entity_label}` with {len(entity_series)} observations.")
            update_step_status("Compute NLI", "running", "Detect breaks, fit ARIMA, run residual tests.")
            status.write("Step 1/5: Detect structural breaks and compute residual-based NLI.")
            nli_result = compute_nli(
                entity_series,
                max_p=max_p,
                max_d=max_d,
                max_q=max_q,
                break_method=evaluation_mode,
                strict_whitening=evaluation_mode == "strict",
            )
            update_step_status("Compute NLI", "completed", f"NLI={nli_result.nli_score:.3f}")
            append_execution_log(
                f"Completed NLI computation. ARIMA order={nli_result.arima_result.order}, "
                f"d={nli_result.arima_result.order[1]}, BDS={nli_result.nli_score:.3f}."
            )
            update_step_status("Run naive baseline", "running", f"Holdout={holdout_size}")
            status.write("Step 2/5: Generate naive carry-forward baseline forecasts.")
            naive_forecast = run_naive_forecast(entity_series, holdout_size=holdout_size)
            update_step_status("Run naive baseline", "completed", "Naive carry-forward baseline finished.")
            append_execution_log("Completed naive carry-forward baseline forecasts.")
            update_step_status("Run ARIMA forecast", "running", f"Holdout={holdout_size}")
            status.write("Step 3/5: Generate expanding-window ARIMA forecasts.")
            arima_forecast = run_arima_forecast(
                entity_series,
                order=nli_result.arima_result.order,
                holdout_size=holdout_size,
                evaluation_mode=evaluation_mode,
            )
            update_step_status(
                "Run ARIMA forecast",
                "completed",
                (
                    f"Failed={int(arima_forecast.metrics['failed_windows'].iloc[0])}, "
                    f"Fallbacks={int(arima_forecast.metrics['fallback_windows'].iloc[0])}"
                ),
            )
            append_execution_log(f"Completed expanding-window ARIMA forecasts in `{evaluation_mode}` mode.")
            chronos_forecast = None
            if run_chronos:
                update_step_status("Run Chronos forecast", "running", "Load local model cache and predict.")
                status.write("Step 4/5: Run Chronos forecast for the selected entity.")
                chronos_forecast = run_chronos_forecast(
                    entity_series,
                    holdout_size=holdout_size,
                    deterministic=chronos_deterministic,
                    seed=chronos_seed,
                    num_samples=chronos_samples,
                )
                chronos_status = "completed" if chronos_forecast.available else "warning"
                chronos_detail = chronos_forecast.message if chronos_forecast.message else "Chronos finished."
                update_step_status("Run Chronos forecast", chronos_status, chronos_detail)
                append_execution_log(f"Chronos step finished: {chronos_detail}")
            else:
                update_step_status("Run Chronos forecast", "skipped", "Chronos checkbox disabled.")
                status.write("Step 4/5: Skip Chronos because it is disabled.")
                append_execution_log("Skipped Chronos forecast because it was disabled.")
            update_step_status("Save analysis result", "running", "Persist single-entity diagnostics in session.")
            status.write("Step 5/5: Save diagnostics and chart data for the dashboard.")
            st.session_state["analysis_result"] = {
                "analysis_scope": analysis_scope,
                "entity_series": entity_series,
                "nli_result": nli_result,
                "naive_forecast": naive_forecast,
                "arima_forecast": arima_forecast,
                "chronos_forecast": chronos_forecast,
                "target_provenance": target_provenance,
                "winsor_report": winsor_report,
                "eligibility_summary": eligibility_summary,
            }
            update_step_status("Save analysis result", "completed", "Single-entity analysis saved.")
            append_execution_log("Saved single-entity analysis results to session state.")
            st.session_state["dashboard_view"] = "Forecasts"
        else:
            append_execution_log("Starting full cleaned dataset analysis.")
            update_step_status("Summarize cleaned panel", "running", "Compute firm coverage and continuity.")
            status.write("Step 1/3: Summarize the cleaned panel by firm coverage and continuity.")
            entity_summary = summarize_entities(analysis_frame, target_column)
            update_step_status("Summarize cleaned panel", "completed", f"Rows={len(entity_summary)}")
            append_execution_log(f"Completed cleaned panel summary for {len(entity_summary)} firms.")
            update_step_status("Compute full-dataset NLI", "running", "Process all cleaned firms.")
            status.write("Step 2/3: Compute full-dataset NLI distribution across all cleaned firms.")
            progress_bar = st.progress(0, text="Preparing full-dataset NLI run...")
            progress_state = {"last_logged_processed": 0}

            def update_progress(processed: int, total: int, label: str) -> None:
                denominator = max(total, 1)
                progress_bar.progress(
                    min(processed / denominator, 1.0),
                    text=f"Processing {processed}/{denominator}: {label}",
                )
                if processed == 1 or processed == denominator or processed - progress_state["last_logged_processed"] >= 25:
                    append_execution_log(f"NLI distribution progress {processed}/{denominator}. Current firm: {label}")
                    progress_state["last_logged_processed"] = processed

            full_distribution_result = compute_nli_distribution(
                analysis_frame,
                target_column,
                max_entities=None,
                progress_callback=update_progress,
                break_method=evaluation_mode,
                strict_whitening=evaluation_mode == "strict",
            )
            progress_bar.progress(1.0, text="Full-dataset NLI run complete.")
            update_step_status(
                "Compute full-dataset NLI",
                "completed",
                f"Successful={full_distribution_result.successful_entities}, Failed={full_distribution_result.failed_entities}",
            )
            append_execution_log(
                "Completed full-dataset NLI distribution. "
                f"Processed={full_distribution_result.processed_entities}, "
                f"Successful={full_distribution_result.successful_entities}, "
                f"Failed={full_distribution_result.failed_entities}."
            )
            update_step_status("Save analysis result", "running", "Persist dataset summary in session.")
            status.write("Step 3/3: Save dataset-wide summaries and distribution outputs.")
            st.session_state["analysis_result"] = {
                "analysis_scope": analysis_scope,
                "entity_summary": entity_summary,
                "full_distribution_result": full_distribution_result,
                "eligibility_summary": eligibility_summary,
                "winsor_report": winsor_report,
                "target_provenance": target_provenance,
            }
            update_step_status("Save analysis result", "completed", "Full-dataset analysis saved.")
            append_execution_log("Saved full-dataset analysis results to session state.")
            st.session_state["dashboard_view"] = "Dataset Summary"

        st.session_state["analysis_meta"] = analysis_meta
        st.session_state["nli_distribution_result"] = None
        st.session_state["nli_distribution_meta"] = None
        st.session_state["validation_result"] = None
        st.session_state["validation_meta"] = None
        st.session_state["tournament_result"] = None
        st.session_state["tournament_meta"] = None
        has_matching_analysis = True
        append_execution_log("Analysis completed successfully.")
        save_current_snapshot(
            "analysis",
            f"{analysis_scope} | {target_column} | {entity_label if analysis_scope == 'Single entity diagnostics' else 'dataset'}",
        )
        status.update(label="Analysis complete.", state="complete")
    except Exception as exc:
        append_execution_log(f"Analysis failed: {exc}")
        update_step_status("Analysis", "error", str(exc))
        st.error(f"Analysis failed: {exc}")
        st.stop()

if not has_matching_analysis:
    if st.session_state.get("analysis_result") is not None:
        st.info("Selections changed. Click `Run Analysis` again to refresh the dashboard.")
    st.stop()

analysis_result = st.session_state["analysis_result"]
analysis_scope = analysis_result["analysis_scope"]
active_analysis_meta = st.session_state.get("analysis_meta") or analysis_meta
active_holdout_size = int(active_analysis_meta.get("holdout_size", holdout_size))
active_evaluation_mode = str(active_analysis_meta.get("evaluation_mode", evaluation_mode))
active_evaluation_label = "Strict deterministic" if active_evaluation_mode == "strict" else "Practical fallback"
active_entity_label = str(active_analysis_meta.get("entity_label", entity_label))
if history_loaded:
    st.info(f"Showing loaded history snapshot: `{st.session_state.get('history_loaded_label', 'saved run')}`")
if analysis_scope == "Single entity diagnostics":
    nli_result = analysis_result["nli_result"]
    result_entity_series = analysis_result.get("entity_series", entity_series)
    naive_forecast = analysis_result["naive_forecast"]
    arima_forecast = analysis_result["arima_forecast"]
    chronos_forecast = analysis_result["chronos_forecast"]
    result_target_provenance = analysis_result.get("target_provenance", target_provenance)
    result_winsor_report = analysis_result.get("winsor_report", winsor_report)
    result_eligibility_summary = analysis_result.get("eligibility_summary", eligibility_summary)
    quality_labels = [active_evaluation_label, f"breaks:{nli_result.break_result.method}"]
    quality_labels.append(f"target:{result_target_provenance.source_type}")
    if result_winsor_report.enabled:
        quality_labels.append("winsorized")
    if int(arima_forecast.metrics["failed_windows"].iloc[0]) > 0:
        quality_labels.append("arima-failed-windows")
    elif int(arima_forecast.metrics["fallback_windows"].iloc[0]) > 0:
        quality_labels.append("arima-fallbacks")
    else:
        quality_labels.append("arima-clean")
    if chronos_forecast is not None and chronos_forecast.available:
        quality_labels.append(
            "chronos-deterministic"
            if chronos_forecast.metrics["inference_mode"].iloc[0] == "deterministic"
            else "chronos-sampled"
        )
    render_quality_badges(quality_labels)

    st.subheader("Transparency Diagnostics")
    diag_left, diag_middle, diag_right = st.columns(3)
    diag_left.metric("Selected ARIMA order", str(nli_result.arima_result.order))
    diag_middle.metric("Differencing order (d)", str(nli_result.arima_result.order[1]))
    diag_right.metric("NLI score (BDS z-stat)", f"{nli_result.nli_score:.3f}")

    info_left, info_middle, info_right = st.columns(3)
    info_left.metric("BDS p-value", f"{nli_result.bds_pvalue:.4f}")
    info_middle.metric(
        "Tsay p-value",
        "N/A" if nli_result.tsay_pvalue is None else f"{nli_result.tsay_pvalue:.4f}",
    )
    info_right.metric("Break count", str(len(nli_result.break_result.break_indices)))

    meta_left, meta_middle, meta_right = st.columns(3)
    meta_left.metric("Holdout quarters", str(active_holdout_size))
    meta_middle.metric("Evaluation mode", active_evaluation_label)
    meta_right.metric("ARIMA failed windows", str(int(arima_forecast.metrics["failed_windows"].iloc[0])))

    with st.expander("Data provenance and audit", expanded=False):
        st.write(
            {
                "target_provenance": result_target_provenance.source_type,
                "provenance_detail": result_target_provenance.detail,
                "winsorization_detail": result_winsor_report.detail,
                "eligible_firms": int(result_eligibility_summary["eligible"].sum()),
                "excluded_firms": int((~result_eligibility_summary["eligible"]).sum()),
            }
        )

    with st.expander("Validation Checks", expanded=False):
        if history_loaded:
            st.caption("Loaded history snapshots show saved validation results. Run the analysis again to recompute validation checks.")
            run_validation_checks = False
        else:
            run_validation_checks = st.button("Run Validation Checks")
        if run_validation_checks:
            append_execution_log("Running repeat-run and reference validation checks.")
            arima_repeat = run_arima_forecast(
                entity_series,
                order=nli_result.arima_result.order,
                holdout_size=holdout_size,
                evaluation_mode=evaluation_mode,
            )
            validation_payload = {
                "naive_repeat": compare_prediction_frames(naive_forecast.predictions, naive_forecast.predictions),
                "arima_repeat": compare_prediction_frames(arima_forecast.predictions, arima_repeat.predictions),
                "arima_reference": arima_reference_summary(
                    nli_result.break_result.segment["value"],
                    nli_result.arima_result.order,
                ),
            }
            if chronos_forecast is not None and chronos_forecast.available:
                if chronos_forecast.metrics["inference_mode"].iloc[0] == "deterministic":
                    chronos_repeat = run_chronos_forecast(
                        entity_series,
                        holdout_size=holdout_size,
                        deterministic=chronos_deterministic,
                        seed=chronos_seed,
                        num_samples=chronos_samples,
                    )
                    validation_payload["chronos_repeat"] = compare_prediction_frames(
                        chronos_forecast.predictions,
                        chronos_repeat.predictions,
                    )
                else:
                    validation_payload["chronos_repeat"] = {
                        "status": "skipped",
                        "message": "Chronos repeat-run check is skipped in sampled mode.",
                    }
            st.session_state["validation_result"] = validation_payload
            st.session_state["validation_meta"] = analysis_meta
            append_execution_log("Validation checks completed.")
            save_current_snapshot("validation", f"Validation | {target_column} | {entity_label}")

        validation_result = st.session_state.get("validation_result")
        validation_meta = st.session_state.get("validation_meta")
        if validation_result is not None and (history_loaded or validation_meta == analysis_meta):
            naive_repeat = validation_result["naive_repeat"]
            arima_repeat = validation_result["arima_repeat"]
            arima_reference = validation_result["arima_reference"]
            checks = [
                {
                    "check": "Naive repeat-run",
                    "status": naive_repeat.status,
                    "detail": naive_repeat.message,
                    "max_abs_diff": naive_repeat.max_abs_diff,
                },
                {
                    "check": "ARIMA repeat-run",
                    "status": arima_repeat.status,
                    "detail": arima_repeat.message,
                    "max_abs_diff": arima_repeat.max_abs_diff,
                },
                {
                    "check": "ARIMA vs statsmodels",
                    "status": arima_reference.status,
                    "detail": arima_reference.message,
                    "max_abs_diff": arima_reference.abs_diff,
                },
            ]
            chronos_repeat = validation_result.get("chronos_repeat")
            if chronos_repeat is not None:
                if hasattr(chronos_repeat, "status"):
                    checks.append(
                        {
                            "check": "Chronos repeat-run",
                            "status": chronos_repeat.status,
                            "detail": chronos_repeat.message,
                            "max_abs_diff": chronos_repeat.max_abs_diff,
                        }
                    )
                else:
                    checks.append(
                        {
                            "check": "Chronos repeat-run",
                            "status": chronos_repeat["status"],
                            "detail": chronos_repeat["message"],
                            "max_abs_diff": None,
                        }
                    )
            st.dataframe(pd.DataFrame(checks), width="stretch", hide_index=True)

    with st.expander("ARIMA Candidate Search", expanded=False):
        st.dataframe(nli_result.arima_result.candidate_table, width="stretch")

    with st.expander("Residual Diagnostics", expanded=True):
        left, right = st.columns(2)
        left.dataframe(nli_result.ljung_box, width="stretch")
        right.write(
            {
                "break_detection_method": nli_result.break_result.method,
                "break_indices": nli_result.break_result.break_indices,
                "break_dates": nli_result.break_result.break_dates,
                "segment_start": nli_result.break_result.segment_start,
                "segment_end": nli_result.break_result.segment_end,
                "message": nli_result.break_result.message,
            }
        )
        if arima_forecast.warnings:
            st.warning("Some ARIMA forecast windows failed or needed fallback logic.")
            st.dataframe(pd.DataFrame({"warning": arima_forecast.warnings}), width="stretch")

view_options = (
    ["Forecasts", "Residuals", "NLI Distribution", "Exports"]
    if analysis_scope == "Single entity diagnostics"
    else ["Dataset Summary", "Tournament Summary", "NLI Distribution", "Exports"]
)
view = st.radio("View", view_options, horizontal=True, key="dashboard_view")

if view == "Forecasts":
    metric_frames = [naive_forecast.metrics, arima_forecast.metrics]
    st.plotly_chart(
        build_combined_forecast_figure(
            result_entity_series,
            [naive_forecast.predictions, arima_forecast.predictions, chronos_forecast.predictions if chronos_forecast is not None and chronos_forecast.available else pd.DataFrame()],
            f"Combined Forecast Comparison: {active_entity_label}",
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        build_forecast_figure(result_entity_series, naive_forecast.predictions, f"Naive Carry-Forward: {active_entity_label}"),
        use_container_width=True,
    )
    st.plotly_chart(
        build_forecast_figure(result_entity_series, arima_forecast.predictions, f"ARIMA Actual vs Prediction: {active_entity_label}"),
        use_container_width=True,
    )

    if chronos_forecast is None:
        st.info("Chronos was not run. Enable the checkbox to compare it directly.")
    elif not chronos_forecast.available:
        st.warning(f"Chronos could not run: {chronos_forecast.message}")
    else:
        if chronos_forecast.message:
            st.info(chronos_forecast.message)
        if not bool(chronos_forecast.metrics["quantile_order_valid"].iloc[0]):
            st.warning("Chronos quantile order check failed.")
        if not bool(chronos_forecast.metrics["interval_width_nonnegative"].iloc[0]):
            st.warning("Chronos interval width check failed.")
        st.plotly_chart(
            build_forecast_figure(
                result_entity_series,
                chronos_forecast.predictions,
                f"Chronos Actual vs Prediction: {active_entity_label}",
            ),
            use_container_width=True,
        )
        metric_frames.append(chronos_forecast.metrics)
        comparison = build_model_gap(arima_forecast.predictions, chronos_forecast.predictions)
        if not comparison.empty:
            st.plotly_chart(build_error_gap_figure(comparison), use_container_width=True)
    st.plotly_chart(
        build_cumulative_error_figure(
            [
                naive_forecast.predictions,
                arima_forecast.predictions,
                chronos_forecast.predictions if chronos_forecast is not None and chronos_forecast.available else pd.DataFrame(),
            ]
        ),
        use_container_width=True,
    )
    st.dataframe(pd.concat(metric_frames, ignore_index=True), width="stretch")

elif view == "Residuals":
    residual_left, residual_right = st.columns(2)
    residual_left.plotly_chart(
        build_residual_figure(nli_result.arima_result.residuals),
        use_container_width=True,
    )
    residual_right.plotly_chart(
        build_residual_autocorrelation_figure(nli_result.arima_result.residuals),
        use_container_width=True,
    )

elif view == "Dataset Summary":
    entity_summary = analysis_result["entity_summary"]
    full_distribution_result = analysis_result["full_distribution_result"]
    audit_summary = analysis_result["eligibility_summary"]
    summary_left, summary_middle, summary_right = st.columns(3)
    summary_left.metric("Processed firms", str(full_distribution_result.processed_entities))
    summary_middle.metric("Successful NLI scores", str(full_distribution_result.successful_entities))
    summary_right.metric("Failed firms", str(full_distribution_result.failed_entities))
    st.plotly_chart(
        build_nli_distribution_figure(full_distribution_result.distribution, selected_entity=entity_id),
        use_container_width=True,
    )
    summary_chart_left, summary_chart_right = st.columns(2)
    summary_chart_left.plotly_chart(build_eligibility_figure(audit_summary), use_container_width=True)
    summary_chart_right.plotly_chart(
        build_nli_quartile_figure(full_distribution_result.distribution),
        use_container_width=True,
    )
    st.dataframe(entity_summary.head(50), width="stretch")
    with st.expander("Eligibility and exclusions", expanded=False):
        st.dataframe(audit_summary.head(100), width="stretch")
    if full_distribution_result.skipped_short_series:
        st.info(f"Skipped {full_distribution_result.skipped_short_series} firms with fewer than 10 usable observations.")
    if not full_distribution_result.failure_examples.empty:
        with st.expander("Sample full-dataset NLI failures", expanded=False):
            st.dataframe(full_distribution_result.failure_examples, width="stretch")

elif view == "Tournament Summary":
    st.caption("Run a cross-firm forecasting tournament to compare Naive, ARIMA, and Chronos across firms.")
    tournament_limit = st.number_input(
        "Max entities for tournament",
        min_value=10,
        max_value=max(10, int(analysis_frame["entity_id"].nunique())),
        value=min(50, int(analysis_frame["entity_id"].nunique())),
        step=10,
        key="tournament_limit",
    )
    tournament_run_chronos = st.checkbox("Include Chronos in tournament", value=run_chronos, key="tournament_run_chronos")
    tournament_meta = {
        "target_column": target_column,
        "tournament_limit": int(tournament_limit),
        "holdout_size": int(holdout_size),
        "run_chronos": bool(tournament_run_chronos),
        "evaluation_mode": evaluation_mode,
        "winsorization_enabled": bool(enable_winsorization),
    }
    tournament_left, tournament_right = st.columns([1, 1])
    compute_tournament = tournament_left.button("Compute Tournament Summary")
    clear_tournament = tournament_right.button("Clear Tournament Result")

    if clear_tournament:
        st.session_state["tournament_result"] = None
        st.session_state["tournament_meta"] = None

    if compute_tournament:
        reset_execution_console()
        append_execution_log(
            f"Tournament requested for `{target_column}` across up to {int(tournament_limit)} firms."
        )
        update_step_status("Compute tournament", "running", f"Limit={int(tournament_limit)}")
        progress_bar = st.progress(0, text="Preparing tournament run...")
        progress_state = {"last_logged_processed": 0}

        def update_tournament_progress(processed: int, total: int, label: str) -> None:
            denominator = max(total, 1)
            progress_bar.progress(min(processed / denominator, 1.0), text=f"Tournament {processed}/{denominator}: {label}")
            if processed == 1 or processed == denominator or processed - progress_state["last_logged_processed"] >= 10:
                append_execution_log(f"Tournament progress {processed}/{denominator}. Current firm: {label}")
                progress_state["last_logged_processed"] = processed

        st.session_state["tournament_result"] = compute_forecasting_tournament(
            analysis_frame,
            target_column,
            holdout_size=holdout_size,
            max_entities=int(tournament_limit),
            run_chronos=bool(tournament_run_chronos),
            evaluation_mode=evaluation_mode,
            max_p=max_p,
            max_d=max_d,
            max_q=max_q,
            chronos_deterministic=chronos_deterministic,
            chronos_seed=chronos_seed,
            chronos_samples=chronos_samples,
            progress_callback=update_tournament_progress,
        )
        st.session_state["tournament_meta"] = tournament_meta
        tournament_result = st.session_state["tournament_result"]
        progress_bar.progress(1.0, text="Tournament complete.")
        update_step_status(
            "Compute tournament",
            "completed",
            f"Successful={tournament_result.successful_entities}, Failed={tournament_result.failed_entities}",
        )
        append_execution_log(
            "Tournament completed. "
            f"Processed={tournament_result.processed_entities}, "
            f"Successful={tournament_result.successful_entities}, "
            f"Failed={tournament_result.failed_entities}."
        )
        save_current_snapshot("tournament", f"Tournament | {target_column} | top {int(tournament_limit)}")

    tournament_result = st.session_state.get("tournament_result")
    stored_tournament_meta = st.session_state.get("tournament_meta")
    if tournament_result is not None and (history_loaded or stored_tournament_meta == tournament_meta):
        if history_loaded and stored_tournament_meta is not None:
            st.caption(
                f"Showing saved tournament for `{stored_tournament_meta['target_column']}` "
                f"across up to {stored_tournament_meta['tournament_limit']} firms."
            )
        metrics = tournament_result.firm_metrics
        card_left, card_middle, card_right = st.columns(3)
        card_left.metric("Processed firms", str(tournament_result.processed_entities))
        card_middle.metric("Successful tournaments", str(tournament_result.successful_entities))
        card_right.metric("Failed firms", str(tournament_result.failed_entities))
        if metrics.empty:
            st.warning("No firms produced a valid tournament result with the current settings.")
        else:
            available_chronos = metrics.loc[metrics["chronos_available"]]
            summary_left, summary_middle, summary_right = st.columns(3)
            summary_left.metric("Chronos win rate", f"{metrics['chronos_beats_arima'].mean():.1%}")
            summary_middle.metric(
                "Mean delta MASE",
                f"{metrics['delta_mase'].dropna().mean():.3f}" if metrics["delta_mase"].notna().any() else "N/A",
            )
            summary_right.metric(
                "Mean Chronos coverage",
                f"{available_chronos['chronos_coverage_rate'].mean():.1%}" if not available_chronos.empty else "N/A",
            )
            chart_left, chart_right = st.columns(2)
            chart_left.plotly_chart(build_tournament_gap_scatter(metrics), use_container_width=True)
            chart_right.plotly_chart(build_volatility_gap_figure(metrics), use_container_width=True)
            chart_left, chart_right = st.columns(2)
            chart_left.plotly_chart(build_tournament_leaderboard_figure(metrics), use_container_width=True)
            chart_right.plotly_chart(build_tournament_quartile_win_figure(metrics), use_container_width=True)
            chart_left, chart_right = st.columns(2)
            chart_left.plotly_chart(build_uncertainty_width_nli_figure(metrics), use_container_width=True)
            chart_right.plotly_chart(build_coverage_quartile_figure(metrics), use_container_width=True)
            st.plotly_chart(build_winner_distribution_figure(metrics), use_container_width=True)
            st.dataframe(metrics.head(100), width="stretch")
        if not tournament_result.failure_examples.empty:
            with st.expander("Tournament failures", expanded=False):
                st.dataframe(tournament_result.failure_examples, width="stretch")

elif view == "NLI Distribution":
    st.caption(
        "Computing NLI across firms is expensive because each firm runs the full break-detection, "
        "ARIMA, and residual-testing pipeline. Start with a smaller sample."
    )
    default_limit = int(analysis_frame["entity_id"].nunique()) if analysis_scope == "Full cleaned dataset summary" else min(50, int(analysis_frame["entity_id"].nunique()))
    distribution_limit = st.number_input(
        "Max entities for NLI distribution",
        min_value=25,
        max_value=max(25, int(analysis_frame["entity_id"].nunique())),
        value=default_limit,
        step=25,
        key="distribution_limit",
    )
    distribution_meta = {
        "target_column": target_column,
        "distribution_limit": int(distribution_limit),
        "row_count": int(len(analysis_frame)),
        "entity_count": int(analysis_frame["entity_id"].nunique()),
        "winsorization_enabled": bool(enable_winsorization),
        "target_provenance": target_provenance.source_type,
    }

    action_left, action_right = st.columns([1, 1])
    compute_distribution = action_left.button("Compute NLI Distribution")
    clear_distribution = action_right.button("Clear Distribution Result")

    if clear_distribution:
        st.session_state["nli_distribution_result"] = None
        st.session_state["nli_distribution_meta"] = None

    if compute_distribution:
        reset_execution_console()
        append_execution_log(
            f"NLI distribution requested for target `{target_column}` across up to {int(distribution_limit)} firms."
        )
        update_step_status("Compute NLI distribution", "running", f"Limit={int(distribution_limit)}")
        progress_bar = st.progress(0, text="Preparing NLI distribution run...")
        status_placeholder = st.empty()
        progress_state = {"last_logged_processed": 0}

        def update_progress(processed: int, total: int, label: str) -> None:
            denominator = max(total, 1)
            progress_bar.progress(
                min(processed / denominator, 1.0),
                text=f"Processing {processed}/{denominator}: {label}",
            )
            status_placeholder.caption(f"Current firm: `{label}`")
            if processed == 1 or processed == denominator or processed - progress_state["last_logged_processed"] >= 25:
                append_execution_log(f"Distribution progress {processed}/{denominator}. Current firm: {label}")
                progress_state["last_logged_processed"] = processed

        st.session_state["nli_distribution_result"] = compute_nli_distribution(
            analysis_frame,
            target_column,
            max_entities=int(distribution_limit),
            progress_callback=update_progress,
            break_method=evaluation_mode,
            strict_whitening=evaluation_mode == "strict",
        )
        st.session_state["nli_distribution_meta"] = distribution_meta
        progress_bar.progress(1.0, text="NLI distribution complete.")
        status_placeholder.caption("Distribution run finished.")
        distribution_result = st.session_state["nli_distribution_result"]
        update_step_status(
            "Compute NLI distribution",
            "completed",
            f"Successful={distribution_result.successful_entities}, Failed={distribution_result.failed_entities}",
        )
        append_execution_log(
            "NLI distribution completed. "
            f"Processed={distribution_result.processed_entities}, "
            f"Successful={distribution_result.successful_entities}, "
            f"Failed={distribution_result.failed_entities}."
        )
        save_current_snapshot("distribution", f"NLI Distribution | {target_column} | top {int(distribution_limit)}")

    distribution_result = st.session_state.get("nli_distribution_result")
    stored_meta = st.session_state.get("nli_distribution_meta")
    if distribution_result is not None and stored_meta is not None:
        distribution = distribution_result.distribution
        st.caption(
            f"Showing saved distribution for `{stored_meta['target_column']}` across "
            f"{stored_meta['distribution_limit']} firms."
        )
        summary_left, summary_middle, summary_right = st.columns(3)
        summary_left.metric("Processed firms", str(distribution_result.processed_entities))
        summary_middle.metric("Successful NLI scores", str(distribution_result.successful_entities))
        summary_right.metric("Failed firms", str(distribution_result.failed_entities))
        if distribution.empty:
            st.warning("No firms produced a valid NLI score with the current target metric.")
        else:
            st.plotly_chart(
                build_nli_distribution_figure(distribution, selected_entity=entity_id),
                use_container_width=True,
            )
            st.dataframe(distribution.head(50), width="stretch")
        if distribution_result.skipped_short_series:
            st.info(f"Skipped {distribution_result.skipped_short_series} firms with fewer than 10 usable observations.")
        if not distribution_result.failure_examples.empty:
            with st.expander("Sample NLI distribution failures", expanded=False):
                st.dataframe(distribution_result.failure_examples, width="stretch")

else:
    entity_summary_csv = summarize_entities(analysis_frame, target_column).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download entity summary",
        entity_summary_csv,
        file_name="entity_summary.csv",
        mime="text/csv",
    )
    if analysis_scope == "Single entity diagnostics":
        st.download_button(
            "Download naive baseline forecasts",
            naive_forecast.predictions.to_csv(index=False).encode("utf-8"),
            file_name="naive_forecasts.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download ARIMA forecasts",
            arima_forecast.predictions.to_csv(index=False).encode("utf-8"),
            file_name="arima_forecasts.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download diagnostics",
            nli_result.arima_result.candidate_table.to_csv(index=False).encode("utf-8"),
            file_name="arima_candidates.csv",
            mime="text/csv",
        )
        if chronos_forecast is not None and chronos_forecast.available:
            st.download_button(
                "Download Chronos forecasts",
                chronos_forecast.predictions.to_csv(index=False).encode("utf-8"),
                file_name="chronos_forecasts.csv",
                mime="text/csv",
            )
    else:
        full_distribution_result = analysis_result["full_distribution_result"]
        st.download_button(
            "Download full-dataset NLI distribution",
            full_distribution_result.distribution.to_csv(index=False).encode("utf-8"),
            file_name="full_dataset_nli_distribution.csv",
            mime="text/csv",
        )
