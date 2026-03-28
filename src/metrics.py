from __future__ import annotations

import numpy as np


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denominator = np.abs(actual) + np.abs(forecast)
    denominator = np.where(denominator == 0.0, 1.0, denominator)
    return float(np.mean(2.0 * np.abs(forecast - actual) / denominator))


def mase(actual: np.ndarray, forecast: np.ndarray, insample: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    insample = np.asarray(insample, dtype=float)

    if insample.size < 2:
        return float("nan")

    scale = np.mean(np.abs(np.diff(insample)))
    if scale == 0.0:
        return float("nan")
    return float(np.mean(np.abs(actual - forecast)) / scale)


def mean_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.mean(forecast - actual))
