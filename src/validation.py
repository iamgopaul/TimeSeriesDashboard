from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.arima_pipeline import _build_fit_result, _one_step_forecast


@dataclass
class RepeatRunCheckResult:
    status: str
    max_abs_diff: float
    compared_points: int
    message: str


@dataclass
class ARIMAReferenceResult:
    available: bool
    status: str
    custom_forecast: float | None
    reference_forecast: float | None
    abs_diff: float | None
    message: str


def compare_prediction_frames(
    first: pd.DataFrame,
    second: pd.DataFrame,
    tolerance: float = 1e-9,
) -> RepeatRunCheckResult:
    if first.empty or second.empty:
        return RepeatRunCheckResult(
            status="skipped",
            max_abs_diff=float("nan"),
            compared_points=0,
            message="One or both prediction frames are empty.",
        )

    left = first[["date", "forecast"]].rename(columns={"forecast": "forecast_left"})
    right = second[["date", "forecast"]].rename(columns={"forecast": "forecast_right"})
    merged = left.merge(right, on="date", how="inner")
    if merged.empty:
        return RepeatRunCheckResult(
            status="skipped",
            max_abs_diff=float("nan"),
            compared_points=0,
            message="No overlapping forecast dates were available for comparison.",
        )

    diffs = (merged["forecast_left"] - merged["forecast_right"]).abs()
    max_abs_diff = float(diffs.max())
    if max_abs_diff == 0.0:
        status = "exact_match"
        message = "Repeat run matched exactly."
    elif max_abs_diff <= tolerance:
        status = "within_tolerance"
        message = f"Repeat run stayed within tolerance ({tolerance:g})."
    else:
        status = "material_drift"
        message = f"Repeat run drift exceeded tolerance ({tolerance:g})."

    return RepeatRunCheckResult(
        status=status,
        max_abs_diff=max_abs_diff,
        compared_points=int(len(merged)),
        message=message,
    )


def arima_reference_summary(values: pd.Series, order: tuple[int, int, int], tolerance: float = 1e-6) -> ARIMAReferenceResult:
    clean = pd.Series(values, dtype=float).dropna().reset_index(drop=True)
    if len(clean) < 12:
        return ARIMAReferenceResult(
            available=False,
            status="skipped",
            custom_forecast=None,
            reference_forecast=None,
            abs_diff=None,
            message="Series is too short for a meaningful ARIMA reference check.",
        )

    try:
        fit_result = _build_fit_result(clean, order)
        custom_forecast = float(_one_step_forecast(clean, fit_result))
    except Exception as exc:
        return ARIMAReferenceResult(
            available=False,
            status="custom_failed",
            custom_forecast=None,
            reference_forecast=None,
            abs_diff=None,
            message=f"Custom ARIMA failed during reference check: {exc}",
        )

    try:
        from statsmodels.tsa.arima.model import ARIMA

        reference_model = ARIMA(clean.to_numpy(dtype=float), order=order, trend="c").fit()
        reference_forecast = float(np.asarray(reference_model.forecast(steps=1)).reshape(-1)[0])
    except Exception as exc:
        return ARIMAReferenceResult(
            available=False,
            status="reference_failed",
            custom_forecast=custom_forecast,
            reference_forecast=None,
            abs_diff=None,
            message=f"statsmodels ARIMA reference check failed: {exc}",
        )

    abs_diff = abs(custom_forecast - reference_forecast)
    if abs_diff <= tolerance:
        status = "within_tolerance"
        message = f"Custom ARIMA matched statsmodels within tolerance ({tolerance:g})."
    else:
        status = "diverged"
        message = f"Custom ARIMA differed from statsmodels by {abs_diff:.6g}."

    return ARIMAReferenceResult(
        available=True,
        status=status,
        custom_forecast=custom_forecast,
        reference_forecast=reference_forecast,
        abs_diff=float(abs_diff),
        message=message,
    )
