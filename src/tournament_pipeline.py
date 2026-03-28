from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.arima_pipeline import expanding_window_forecast as run_arima_forecast
from src.arima_pipeline import naive_expanding_window_forecast as run_naive_forecast
from src.chronos_pipeline import expanding_window_forecast as run_chronos_forecast
from src.nli_pipeline import compute_nli


@dataclass
class ForecastTournamentResult:
    firm_metrics: pd.DataFrame
    forecast_panel: pd.DataFrame
    firm_forecast_summary: pd.DataFrame
    requested_entities: int
    processed_entities: int
    successful_entities: int
    skipped_entities: int
    failed_entities: int
    failure_examples: pd.DataFrame


@dataclass
class ForecastTournamentChunkResult:
    firm_metrics: pd.DataFrame
    forecast_panel: pd.DataFrame
    firm_forecast_summary: pd.DataFrame
    failures: pd.DataFrame
    processed_entities: int
    skipped_entities: int
    next_index: int
    total_entities: int
    completed: bool


FORECAST_PANEL_COLUMNS = [
    "entity_id",
    "entity_label",
    "date",
    "model",
    "actual",
    "forecast",
    "error",
    "absolute_error",
    "holdout_step",
    "q025",
    "q05",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "q95",
    "q975",
]


def _build_forecast_panel_row(entity_id: str, entity_label: str, forecast_frame: pd.DataFrame) -> pd.DataFrame:
    if forecast_frame.empty:
        return pd.DataFrame(columns=FORECAST_PANEL_COLUMNS)
    working = forecast_frame.copy().sort_values("date").reset_index(drop=True)
    working["entity_id"] = entity_id
    working["entity_label"] = entity_label
    working["absolute_error"] = working["error"].abs() if "error" in working.columns else np.nan
    working["holdout_step"] = np.arange(1, len(working) + 1, dtype=int)
    for column in ("q025", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q975"):
        if column not in working.columns:
            working[column] = np.nan
    return working[FORECAST_PANEL_COLUMNS].copy()


def compute_forecasting_tournament(
    frame: pd.DataFrame,
    target_column: str,
    holdout_size: int,
    max_entities: int | None = None,
    run_chronos: bool = True,
    evaluation_mode: str = "strict",
    max_p: int = 2,
    max_d: int = 2,
    max_q: int = 2,
    chronos_deterministic: bool = True,
    chronos_seed: int = 17,
    chronos_samples: int = 20,
    minimum_history: int = 10,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> ForecastTournamentResult:
    rows: list[dict] = []
    forecast_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    failures: list[dict] = []
    processed_entities = 0
    skipped_entities = 0
    total_entities = frame["entity_id"].nunique()
    if max_entities is not None:
        total_entities = min(total_entities, max_entities)

    start_index = 0
    while start_index < total_entities:
        chunk = compute_forecasting_tournament_chunk(
            frame,
            target_column,
            holdout_size=holdout_size,
            start_index=start_index,
            batch_size=max(1, min(100, total_entities - start_index)),
            max_entities=max_entities,
            run_chronos=run_chronos,
            evaluation_mode=evaluation_mode,
            max_p=max_p,
            max_d=max_d,
            max_q=max_q,
            chronos_deterministic=chronos_deterministic,
            chronos_seed=chronos_seed,
            chronos_samples=chronos_samples,
            minimum_history=minimum_history,
            progress_callback=progress_callback,
        )
        rows.extend(chunk.firm_metrics.to_dict("records"))
        if not chunk.forecast_panel.empty:
            forecast_frames.append(chunk.forecast_panel)
        summary_rows.extend(chunk.firm_forecast_summary.to_dict("records"))
        failures.extend(chunk.failures.to_dict("records"))
        processed_entities += chunk.processed_entities
        skipped_entities += chunk.skipped_entities
        start_index = chunk.next_index
        if chunk.completed:
            break

    return build_forecasting_tournament_result(
        firm_metrics_rows=rows,
        forecast_panel_rows=pd.concat(forecast_frames, ignore_index=True) if forecast_frames else pd.DataFrame(columns=FORECAST_PANEL_COLUMNS),
        firm_forecast_summary_rows=summary_rows,
        failures=failures,
        processed_entities=processed_entities,
        skipped_entities=skipped_entities,
        requested_entities=total_entities,
    )


def compute_forecasting_tournament_chunk(
    frame: pd.DataFrame,
    target_column: str,
    holdout_size: int,
    start_index: int,
    batch_size: int,
    max_entities: int | None = None,
    run_chronos: bool = True,
    evaluation_mode: str = "strict",
    max_p: int = 2,
    max_d: int = 2,
    max_q: int = 2,
    chronos_deterministic: bool = True,
    chronos_seed: int = 17,
    chronos_samples: int = 20,
    minimum_history: int = 10,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> ForecastTournamentChunkResult:
    rows: list[dict] = []
    forecast_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    failures: list[dict] = []
    processed_entities = 0
    skipped_entities = 0
    grouped = frame.groupby("entity_id", sort=True)
    total_entities = frame["entity_id"].nunique()
    if max_entities is not None:
        total_entities = min(total_entities, max_entities)
    start_index = max(0, int(start_index))
    end_index = min(total_entities, start_index + max(1, int(batch_size)))

    for index, (entity_id, subset) in enumerate(grouped):
        if index < start_index:
            continue
        if index >= end_index or (max_entities is not None and index >= max_entities):
            break
        processed_entities += 1
        entity_label = str(subset["entity_label"].iloc[0])
        if progress_callback is not None:
            progress_callback(start_index + processed_entities, total_entities, entity_label)

        series_frame = subset[["date", target_column]].rename(columns={target_column: "value"}).dropna()
        series_frame = series_frame.sort_values("date").reset_index(drop=True)
        required_history = max(int(minimum_history), holdout_size + 10)
        if len(series_frame) < required_history:
            skipped_entities += 1
            failures.append(
                {
                    "entity_id": entity_id,
                    "entity_label": entity_label,
                    "error": f"Insufficient history for requested holdout and diagnostics (need >= {required_history} rows).",
                }
            )
            continue

        try:
            nli_result = compute_nli(
                series_frame,
                max_p=max_p,
                max_d=max_d,
                max_q=max_q,
                break_method=evaluation_mode,
                strict_whitening=evaluation_mode == "strict",
            )
            naive_forecast = run_naive_forecast(series_frame, holdout_size=holdout_size)
            arima_forecast = run_arima_forecast(
                series_frame,
                order=nli_result.arima_result.order,
                holdout_size=holdout_size,
                evaluation_mode=evaluation_mode,
            )
            chronos_forecast = None
            if run_chronos:
                chronos_forecast = run_chronos_forecast(
                    series_frame,
                    holdout_size=holdout_size,
                    deterministic=chronos_deterministic,
                    seed=chronos_seed,
                    num_samples=chronos_samples,
                )
        except Exception as exc:
            failures.append(
                {
                    "entity_id": entity_id,
                    "entity_label": entity_label,
                    "error": str(exc),
                }
            )
            continue

        chronos_available = bool(chronos_forecast is not None and chronos_forecast.available)
        chronos_mase = float(chronos_forecast.metrics["MASE"].iloc[0]) if chronos_available else float("nan")
        arima_mase = float(arima_forecast.metrics["MASE"].iloc[0])
        naive_mase = float(naive_forecast.metrics["MASE"].iloc[0])
        delta_mase = arima_mase - chronos_mase if chronos_available else float("nan")
        volatility = float(series_frame["value"].std(ddof=0)) if len(series_frame) > 1 else 0.0
        size_proxy = float(subset["assets"].dropna().median()) if "assets" in subset.columns and subset["assets"].notna().any() else float("nan")

        candidates = {
            "Naive": naive_mase,
            "ARIMA": arima_mase,
        }
        if chronos_available:
            candidates["Chronos"] = chronos_mase
        winner = min(candidates.items(), key=lambda item: item[1])[0] if candidates else "N/A"

        entity_forecasts = [
            _build_forecast_panel_row(str(entity_id), entity_label, naive_forecast.predictions),
            _build_forecast_panel_row(str(entity_id), entity_label, arima_forecast.predictions),
        ]
        if chronos_available and chronos_forecast is not None:
            entity_forecasts.append(_build_forecast_panel_row(str(entity_id), entity_label, chronos_forecast.predictions))
        combined_entity_forecasts = pd.concat(entity_forecasts, ignore_index=True)
        forecast_frames.append(combined_entity_forecasts)

        def _model_mae(model_name: str) -> float:
            model_rows = combined_entity_forecasts.loc[combined_entity_forecasts["model"] == model_name]
            if model_rows.empty:
                return float("nan")
            return float(model_rows["absolute_error"].mean())

        rows.append(
            {
                "entity_id": entity_id,
                "entity_label": entity_label,
                "observations": int(len(series_frame)),
                "nli_score": float(nli_result.nli_score),
                "break_count": int(len(nli_result.break_result.break_indices)),
                "break_method": nli_result.break_result.method,
                "volatility": volatility,
                "size_proxy": size_proxy,
                "naive_mase": naive_mase,
                "arima_mase": arima_mase,
                "chronos_mase": chronos_mase,
                "delta_mase": delta_mase,
                "arima_beats_naive": bool(arima_mase < naive_mase),
                "chronos_beats_naive": bool(chronos_available and chronos_mase < naive_mase),
                "chronos_beats_arima": bool(chronos_available and chronos_mase < arima_mase),
                "winner": winner,
                "chronos_available": chronos_available,
                "chronos_coverage_rate": float(chronos_forecast.metrics["interval_coverage_rate"].iloc[0]) if chronos_available else float("nan"),
                "chronos_interval_width": float(chronos_forecast.metrics["median_interval_width"].iloc[0]) if chronos_available else float("nan"),
            }
        )
        summary_rows.append(
            {
                "entity_id": entity_id,
                "entity_label": entity_label,
                "holdout_points": int(combined_entity_forecasts["date"].nunique()),
                "holdout_start": combined_entity_forecasts["date"].min(),
                "holdout_end": combined_entity_forecasts["date"].max(),
                "holdout_actual_mean": float(combined_entity_forecasts["actual"].mean()),
                "holdout_actual_std": float(combined_entity_forecasts["actual"].std(ddof=0)),
                "naive_mae": _model_mae("Naive"),
                "arima_mae": _model_mae("ARIMA"),
                "chronos_mae": _model_mae("Chronos"),
                "winner": winner,
                "chronos_available": chronos_available,
                "nli_score": float(nli_result.nli_score),
                "break_count": int(len(nli_result.break_result.break_indices)),
                "volatility": volatility,
            }
        )

    next_index = end_index
    return ForecastTournamentChunkResult(
        firm_metrics=pd.DataFrame(rows),
        forecast_panel=pd.concat(forecast_frames, ignore_index=True) if forecast_frames else pd.DataFrame(columns=FORECAST_PANEL_COLUMNS),
        firm_forecast_summary=pd.DataFrame(summary_rows),
        failures=pd.DataFrame(failures, columns=["entity_id", "entity_label", "error"]),
        processed_entities=processed_entities,
        skipped_entities=skipped_entities,
        next_index=next_index,
        total_entities=total_entities,
        completed=next_index >= total_entities,
    )


def build_forecasting_tournament_result(
    firm_metrics_rows: list[dict] | pd.DataFrame,
    forecast_panel_rows: list[dict] | pd.DataFrame,
    firm_forecast_summary_rows: list[dict] | pd.DataFrame,
    failures: list[dict] | pd.DataFrame,
    processed_entities: int,
    skipped_entities: int,
    requested_entities: int,
) -> ForecastTournamentResult:
    firm_metrics = pd.DataFrame(firm_metrics_rows)
    if not firm_metrics.empty:
        firm_metrics = firm_metrics.sort_values(["delta_mase", "entity_label"], ascending=[False, True]).reset_index(drop=True)
    forecast_panel = pd.DataFrame(forecast_panel_rows)
    if forecast_panel.empty:
        forecast_panel = pd.DataFrame(columns=FORECAST_PANEL_COLUMNS)
    else:
        forecast_panel = forecast_panel.sort_values(["entity_label", "model", "date"]).reset_index(drop=True)
    firm_forecast_summary = pd.DataFrame(firm_forecast_summary_rows)
    if not firm_forecast_summary.empty:
        firm_forecast_summary = firm_forecast_summary.sort_values(["winner", "entity_label"]).reset_index(drop=True)
    failure_frame = pd.DataFrame(failures)
    if failure_frame.empty:
        failure_frame = pd.DataFrame(columns=["entity_id", "entity_label", "error"])
    failure_examples = failure_frame.head(20).reset_index(drop=True)
    return ForecastTournamentResult(
        firm_metrics=firm_metrics,
        forecast_panel=forecast_panel,
        firm_forecast_summary=firm_forecast_summary,
        requested_entities=int(requested_entities),
        processed_entities=int(processed_entities),
        successful_entities=int(len(firm_metrics)),
        skipped_entities=int(skipped_entities),
        failed_entities=int(len(failure_frame)),
        failure_examples=failure_examples,
    )
