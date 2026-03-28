from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.metrics import mase, mean_error, smape


@dataclass
class ARIMAFitResult:
    order: tuple[int, int, int]
    aicc: float
    residuals: pd.Series
    fitted_values: pd.Series
    candidate_table: pd.DataFrame
    intercept: float
    ar_params: np.ndarray
    ma_params: np.ndarray


@dataclass
class ForecastRunResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    holdout_size: int
    warnings: list[str]
    evaluation_mode: str


def _aicc(aic: float, nobs: int, k_params: int) -> float:
    if nobs <= k_params + 1:
        return float("inf")
    return float(aic + ((2 * k_params * (k_params + 1)) / (nobs - k_params - 1)))


def _difference(values: np.ndarray, d: int) -> np.ndarray:
    differenced = np.asarray(values, dtype=float)
    for _ in range(d):
        differenced = np.diff(differenced)
    return differenced


def _compute_arma_components(
    series: np.ndarray,
    intercept: float,
    ar_params: np.ndarray,
    ma_params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p = len(ar_params)
    q = len(ma_params)
    start = max(p, q)
    fitted = np.full(len(series), np.nan, dtype=float)
    residuals = np.zeros(len(series), dtype=float)

    for idx in range(start, len(series)):
        ar_term = 0.0
        ma_term = 0.0
        if p:
            ar_term = float(np.dot(ar_params, series[idx - np.arange(1, p + 1)]))
        if q:
            ma_term = float(np.dot(ma_params, residuals[idx - np.arange(1, q + 1)]))
        fitted[idx] = intercept + ar_term + ma_term
        residuals[idx] = series[idx] - fitted[idx]

    return fitted, residuals


def _fit_arima_model(values: pd.Series, order: tuple[int, int, int]) -> dict:
    p, d, q = order
    original = np.asarray(values, dtype=float)
    differenced = _difference(original, d)
    start = max(p, q)
    if len(differenced) <= max(start + 2, 6):
        raise ValueError("Insufficient data after differencing.")

    def objective(params: np.ndarray) -> float:
        intercept = params[0]
        ar_params = params[1 : 1 + p]
        ma_params = params[1 + p :]
        _, residuals = _compute_arma_components(differenced, intercept, ar_params, ma_params)
        usable = residuals[start:]
        return float(np.sum(np.square(usable)))

    initial = np.zeros(1 + p + q, dtype=float)
    initial[0] = float(np.mean(differenced[start:])) if len(differenced[start:]) else 0.0
    bounds = [(-10.0, 10.0)] * len(initial)
    result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        raise RuntimeError(result.message)

    intercept = float(result.x[0])
    ar_params = np.asarray(result.x[1 : 1 + p], dtype=float)
    ma_params = np.asarray(result.x[1 + p :], dtype=float)
    fitted_diff, residuals_diff = _compute_arma_components(differenced, intercept, ar_params, ma_params)
    usable_residuals = residuals_diff[start:]
    sigma2 = float(np.mean(np.square(usable_residuals)))
    if sigma2 <= 0.0:
        sigma2 = 1e-9
    k_params = 1 + p + q
    nobs = len(usable_residuals)
    aic = float(nobs * np.log(sigma2) + 2 * k_params)
    return {
        "intercept": intercept,
        "ar_params": ar_params,
        "ma_params": ma_params,
        "aicc": _aicc(aic, nobs, k_params),
        "residuals": residuals_diff,
        "fitted": fitted_diff,
        "diffed_values": differenced,
        "start": start,
    }


def _invert_one_step_forecast(values: np.ndarray, d: int, next_diff_value: float) -> float:
    if d == 0:
        return float(next_diff_value)

    histories = [np.asarray(values, dtype=float)]
    for _ in range(d):
        histories.append(np.diff(histories[-1]))

    next_value = float(next_diff_value)
    for level in range(d - 1, -1, -1):
        next_value = float(histories[level][-1] + next_value)
    return next_value


def _one_step_forecast(values: pd.Series, fit_result: ARIMAFitResult) -> float:
    p, d, q = fit_result.order
    original = np.asarray(values, dtype=float)
    differenced = _difference(original, d)
    _, residuals = _compute_arma_components(
        differenced,
        fit_result.intercept,
        fit_result.ar_params,
        fit_result.ma_params,
    )
    ar_term = 0.0
    ma_term = 0.0
    if p:
        ar_term = float(np.dot(fit_result.ar_params, differenced[-np.arange(1, p + 1)]))
    if q:
        ma_term = float(np.dot(fit_result.ma_params, residuals[-np.arange(1, q + 1)]))
    next_diff_value = fit_result.intercept + ar_term + ma_term
    return _invert_one_step_forecast(original, d, next_diff_value)


def _naive_forecast(values: pd.Series, d: int) -> float:
    original = np.asarray(values, dtype=float)
    if len(original) == 0:
        raise ValueError("Cannot forecast an empty series.")
    if d <= 0:
        return float(original[-1])
    return _invert_one_step_forecast(original, d, 0.0)


def _build_metric_frame(
    model_name: str,
    prediction_frame: pd.DataFrame,
    insample: np.ndarray,
    extra_fields: dict | None = None,
) -> pd.DataFrame:
    valid = prediction_frame.dropna(subset=["actual", "forecast"])
    row = {
        "model": model_name,
        "MASE": float("nan"),
        "sMAPE": float("nan"),
        "mean_error": float("nan"),
        "successful_windows": int(len(valid)),
        "failed_windows": int(len(prediction_frame) - len(valid)),
    }
    if not valid.empty:
        row["MASE"] = mase(
            valid["actual"].to_numpy(),
            valid["forecast"].to_numpy(),
            insample,
        )
        row["sMAPE"] = smape(
            valid["actual"].to_numpy(),
            valid["forecast"].to_numpy(),
        )
        row["mean_error"] = mean_error(
            valid["actual"].to_numpy(),
            valid["forecast"].to_numpy(),
        )
    if extra_fields:
        row.update(extra_fields)
    return pd.DataFrame([row])


def _build_fit_result(values: pd.Series, order: tuple[int, int, int]) -> ARIMAFitResult:
    fitted = _fit_arima_model(values, order)
    start = int(fitted["start"])
    residuals = pd.Series(fitted["residuals"], name="residual")
    fitted_values = pd.Series(fitted["fitted"], name="fitted")
    residuals.iloc[:start] = np.nan
    fitted_values.iloc[:start] = np.nan
    candidate_table = pd.DataFrame([{"order": str(order), "aicc": fitted["aicc"], "status": "ok"}])
    return ARIMAFitResult(
        order=order,
        aicc=float(fitted["aicc"]),
        residuals=residuals,
        fitted_values=fitted_values,
        candidate_table=candidate_table,
        intercept=float(fitted["intercept"]),
        ar_params=np.asarray(fitted["ar_params"], dtype=float),
        ma_params=np.asarray(fitted["ma_params"], dtype=float),
    )


def fit_best_arima(
    values: Sequence[float],
    max_p: int = 2,
    max_d: int = 2,
    max_q: int = 2,
) -> ARIMAFitResult:
    values = pd.Series(values, dtype=float).dropna().reset_index(drop=True)
    if len(values) < 10:
        raise ValueError("At least 10 observations are required for ARIMA selection.")

    attempts: List[dict] = []
    best_model = None
    best_order = None
    best_aicc = float("inf")

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                order = (p, d, q)
                try:
                    fitted = _fit_arima_model(values, order)
                    aicc = float(fitted["aicc"])
                    attempts.append({"order": str(order), "aicc": aicc, "status": "ok"})
                    if aicc < best_aicc:
                        best_aicc = aicc
                        best_model = fitted
                        best_order = order
                except Exception as exc:  # pragma: no cover - exact message varies by platform
                    attempts.append({"order": str(order), "aicc": float("inf"), "status": str(exc)})

    if best_model is None or best_order is None:
        raise RuntimeError("No ARIMA model converged for the selected series.")

    start = int(best_model["start"])
    residuals = pd.Series(best_model["residuals"], name="residual")
    fitted_values = pd.Series(best_model["fitted"], name="fitted")
    residuals.iloc[:start] = np.nan
    fitted_values.iloc[:start] = np.nan
    candidate_table = pd.DataFrame(attempts).sort_values(["aicc", "order"]).reset_index(drop=True)

    return ARIMAFitResult(
        order=best_order,
        aicc=float(best_aicc),
        residuals=residuals,
        fitted_values=fitted_values,
        candidate_table=candidate_table,
        intercept=float(best_model["intercept"]),
        ar_params=np.asarray(best_model["ar_params"], dtype=float),
        ma_params=np.asarray(best_model["ma_params"], dtype=float),
    )


def expanding_window_forecast(
    series_frame: pd.DataFrame,
    order: tuple[int, int, int],
    holdout_size: int,
    evaluation_mode: str = "practical",
) -> ForecastRunResult:
    if evaluation_mode not in {"strict", "practical"}:
        raise ValueError("evaluation_mode must be `strict` or `practical`.")

    working = series_frame.sort_values("date").reset_index(drop=True)
    if holdout_size <= 0 or holdout_size >= len(working):
        raise ValueError("Holdout size must be positive and smaller than the series length.")

    predictions = []
    warnings: list[str] = []
    split_index = len(working) - holdout_size

    for idx in range(split_index, len(working)):
        train_values = working.loc[: idx - 1, "value"].astype(float)
        actual = float(working.loc[idx, "value"])
        date = working.loc[idx, "date"]
        fit_strategy = "selected_order"
        fit_order = order

        try:
            fitted = _build_fit_result(train_values, order)
            forecast_value = _one_step_forecast(train_values, fitted)
        except Exception as primary_exc:
            if evaluation_mode == "strict":
                fitted = None
                forecast_value = np.nan
                fit_strategy = "failed"
                warnings.append(
                    f"{date.date()}: selected ARIMA order {order} failed in strict mode ({primary_exc})."
                )
            else:
                try:
                    fitted = fit_best_arima(
                        train_values,
                        max_p=order[0],
                        max_d=order[1],
                        max_q=order[2],
                    )
                    forecast_value = _one_step_forecast(train_values, fitted)
                    fit_strategy = "window_reselected"
                    fit_order = fitted.order
                    warnings.append(
                        f"{date.date()}: selected ARIMA order {order} failed; "
                        f"reselected {fit_order} for this window ({primary_exc})."
                    )
                except Exception as fallback_exc:
                    fitted = None
                    forecast_value = _naive_forecast(train_values, order[1])
                    fit_strategy = "naive_fallback"
                    warnings.append(
                        f"{date.date()}: ARIMA optimization failed for order {order}; "
                        f"used naive fallback ({primary_exc}; {fallback_exc})."
                    )

        predictions.append(
            {
                "date": date,
                "actual": actual,
                "forecast": forecast_value,
                "error": forecast_value - actual,
                "model": "ARIMA",
                "fit_strategy": fit_strategy,
                "fit_order": str(fit_order),
            }
        )

    prediction_frame = pd.DataFrame(predictions)
    insample = working.loc[: split_index - 1, "value"].to_numpy(dtype=float)
    fallback_count = int((prediction_frame["fit_strategy"] != "selected_order").sum())
    metric_frame = _build_metric_frame(
        "ARIMA",
        prediction_frame,
        insample,
        extra_fields={
            "fallback_windows": fallback_count,
            "evaluation_mode": evaluation_mode,
        },
    )

    return ForecastRunResult(
        predictions=prediction_frame,
        metrics=metric_frame,
        holdout_size=holdout_size,
        warnings=warnings,
        evaluation_mode=evaluation_mode,
    )


def naive_expanding_window_forecast(series_frame: pd.DataFrame, holdout_size: int) -> ForecastRunResult:
    working = series_frame.sort_values("date").reset_index(drop=True)
    if holdout_size <= 0 or holdout_size >= len(working):
        raise ValueError("Holdout size must be positive and smaller than the series length.")

    split_index = len(working) - holdout_size
    predictions = []
    for idx in range(split_index, len(working)):
        train_values = working.loc[: idx - 1, "value"].astype(float)
        actual = float(working.loc[idx, "value"])
        date = working.loc[idx, "date"]
        forecast_value = _naive_forecast(train_values, d=0)
        predictions.append(
            {
                "date": date,
                "actual": actual,
                "forecast": forecast_value,
                "error": forecast_value - actual,
                "model": "Naive",
                "fit_strategy": "carry_forward",
                "fit_order": "N/A",
            }
        )

    prediction_frame = pd.DataFrame(predictions)
    insample = working.loc[: split_index - 1, "value"].to_numpy(dtype=float)
    metric_frame = _build_metric_frame(
        "Naive",
        prediction_frame,
        insample,
        extra_fields={
            "fallback_windows": 0,
            "evaluation_mode": "strict",
        },
    )

    return ForecastRunResult(
        predictions=prediction_frame,
        metrics=metric_frame,
        holdout_size=holdout_size,
        warnings=[],
        evaluation_mode="strict",
    )
