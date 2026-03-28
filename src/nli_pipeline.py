from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import bds

from src.arima_pipeline import ARIMAFitResult, fit_best_arima
from src.break_detection import BreakDetectionResult, detect_structural_breaks


@dataclass
class NLIResult:
    break_result: BreakDetectionResult
    arima_result: ARIMAFitResult
    ljung_box: pd.DataFrame
    bds_statistic: float
    bds_pvalue: float
    tsay_f_stat: float | None
    tsay_pvalue: float | None
    nli_score: float


@dataclass
class NLIDistributionResult:
    distribution: pd.DataFrame
    requested_entities: int
    processed_entities: int
    successful_entities: int
    skipped_short_series: int
    failed_entities: int
    failure_examples: pd.DataFrame


@dataclass
class NLIDistributionChunkResult:
    distribution: pd.DataFrame
    failures: pd.DataFrame
    processed_entities: int
    skipped_short_series: int
    next_index: int
    total_entities: int
    completed: bool


def _safe_scalar(value: Iterable[float] | float) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        return float("nan")
    return float(array[0])


def run_tsay_test(residuals: pd.Series, lag: int = 2) -> tuple[float | None, float | None]:
    series = pd.Series(residuals, dtype=float).dropna().reset_index(drop=True)
    if len(series) <= lag + 5:
        return None, None

    lagged = pd.concat(
        [series.shift(step).rename(f"lag_{step}") for step in range(1, lag + 1)],
        axis=1,
    ).dropna()
    target = series.loc[lagged.index]
    quadratic_terms = {}

    for i, left in enumerate(lagged.columns):
        for right in lagged.columns[i:]:
            quadratic_terms[f"{left}*{right}"] = lagged[left] * lagged[right]

    if not quadratic_terms:
        return None, None

    design = sm.add_constant(pd.DataFrame(quadratic_terms))
    model = sm.OLS(target, design).fit()
    restriction = " = 0, ".join(quadratic_terms.keys()) + " = 0"
    test = model.f_test(restriction)
    return float(test.fvalue), float(test.pvalue)


def compute_nli(
    series_frame: pd.DataFrame,
    max_p: int = 2,
    max_d: int = 2,
    max_q: int = 2,
    break_method: str = "practical",
    strict_whitening: bool = False,
) -> NLIResult:
    break_result = detect_structural_breaks(series_frame, method=break_method)
    segment = break_result.segment
    if len(segment) < 10:
        raise ValueError("Selected segment is too short for NLI analysis.")

    arima_result = fit_best_arima(segment["value"], max_p=max_p, max_d=max_d, max_q=max_q)
    residuals = arima_result.residuals.dropna()
    lb = acorr_ljungbox(
        residuals,
        lags=[min(4, max(len(residuals) // 2, 1))],
        model_df=arima_result.order[0] + arima_result.order[2],
        return_df=True,
    )
    lb_value = float(lb["lb_pvalue"].iloc[0]) if not lb.empty else float("nan")
    if strict_whitening and not pd.isna(lb_value) and lb_value < 0.05:
        raise ValueError(f"Residual whitening failed Ljung-Box gate (p={lb_value:.4f}).")
    bds_statistic, bds_pvalue = bds(residuals, max_dim=2)
    tsay_f_stat, tsay_pvalue = run_tsay_test(residuals)

    return NLIResult(
        break_result=break_result,
        arima_result=arima_result,
        ljung_box=lb.reset_index().rename(columns={"index": "lag"}),
        bds_statistic=_safe_scalar(bds_statistic),
        bds_pvalue=_safe_scalar(bds_pvalue),
        tsay_f_stat=tsay_f_stat,
        tsay_pvalue=tsay_pvalue,
        nli_score=_safe_scalar(bds_statistic),
    )


def compute_nli_distribution(
    frame: pd.DataFrame,
    target_column: str,
    max_entities: int | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    break_method: str = "practical",
    strict_whitening: bool = False,
    max_p: int = 2,
    max_d: int = 2,
    max_q: int = 2,
    min_history: int = 10,
) -> NLIDistributionResult:
    rows: list[dict] = []
    failures: list[dict] = []
    processed_entities = 0
    skipped_short_series = 0
    start_index = 0
    total_entities = frame["entity_id"].nunique()
    if max_entities is not None:
        total_entities = min(total_entities, max_entities)

    while start_index < total_entities:
        chunk = compute_nli_distribution_chunk(
            frame,
            target_column,
            start_index=start_index,
            batch_size=max(1, min(100, total_entities - start_index)),
            max_entities=max_entities,
            progress_callback=progress_callback,
            break_method=break_method,
            strict_whitening=strict_whitening,
            max_p=max_p,
            max_d=max_d,
            max_q=max_q,
            min_history=min_history,
        )
        rows.extend(chunk.distribution.to_dict("records"))
        failures.extend(chunk.failures.to_dict("records"))
        processed_entities += chunk.processed_entities
        skipped_short_series += chunk.skipped_short_series
        start_index = chunk.next_index
        if chunk.completed:
            break

    return build_nli_distribution_result(
        rows,
        failures,
        processed_entities=processed_entities,
        skipped_short_series=skipped_short_series,
        requested_entities=total_entities,
    )


def compute_nli_distribution_chunk(
    frame: pd.DataFrame,
    target_column: str,
    start_index: int,
    batch_size: int,
    max_entities: int | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    break_method: str = "practical",
    strict_whitening: bool = False,
    max_p: int = 2,
    max_d: int = 2,
    max_q: int = 2,
    min_history: int = 10,
) -> NLIDistributionChunkResult:
    rows = []
    failures = []
    processed_entities = 0
    skipped_short_series = 0
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
        if progress_callback is not None:
            progress_callback(start_index + processed_entities, total_entities, subset["entity_label"].iloc[0])
        series_frame = subset[["date", target_column]].rename(columns={target_column: "value"}).dropna()
        required_history = max(int(min_history), 10)
        if len(series_frame) < required_history:
            skipped_short_series += 1
            continue
        try:
            result = compute_nli(
                series_frame,
                max_p=max_p,
                max_d=max_d,
                max_q=max_q,
                break_method=break_method,
                strict_whitening=strict_whitening,
            )
            rows.append(
                {
                    "entity_id": entity_id,
                    "entity_label": subset["entity_label"].iloc[0],
                    "nli_score": result.nli_score,
                    "arima_order": str(result.arima_result.order),
                    "break_count": len(result.break_result.break_indices),
                    "break_method": result.break_result.method,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "entity_id": entity_id,
                    "entity_label": subset["entity_label"].iloc[0],
                    "error": str(exc),
                }
            )
            continue

    next_index = end_index
    return NLIDistributionChunkResult(
        distribution=pd.DataFrame(
            rows,
            columns=["entity_id", "entity_label", "nli_score", "arima_order", "break_count", "break_method"],
        ),
        failures=pd.DataFrame(failures, columns=["entity_id", "entity_label", "error"]),
        processed_entities=processed_entities,
        skipped_short_series=skipped_short_series,
        next_index=next_index,
        total_entities=total_entities,
        completed=next_index >= total_entities,
    )


def build_nli_distribution_result(
    rows: list[dict] | pd.DataFrame,
    failures: list[dict] | pd.DataFrame,
    processed_entities: int,
    skipped_short_series: int,
    requested_entities: int,
) -> NLIDistributionResult:
    distribution = pd.DataFrame(rows)
    if not distribution.empty:
        distribution = distribution.sort_values("nli_score", ascending=False).reset_index(drop=True)
    else:
        distribution = pd.DataFrame(columns=["entity_id", "entity_label", "nli_score", "arima_order", "break_count", "break_method"])

    failure_frame = pd.DataFrame(failures)
    if failure_frame.empty:
        failure_frame = pd.DataFrame(columns=["entity_id", "entity_label", "error"])
    failure_examples = failure_frame.head(10).reset_index(drop=True)
    return NLIDistributionResult(
        distribution=distribution,
        requested_entities=int(requested_entities),
        processed_entities=int(processed_entities),
        successful_entities=int(len(distribution)),
        skipped_short_series=int(skipped_short_series),
        failed_entities=int(len(failure_frame)),
        failure_examples=failure_examples,
    )
