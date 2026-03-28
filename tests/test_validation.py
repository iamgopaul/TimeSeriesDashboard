import numpy as np
import pandas as pd

from src.validation import arima_reference_summary, compare_prediction_frames


def test_compare_prediction_frames_detects_exact_match():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-03-31", periods=3, freq="QE-DEC"),
            "forecast": [1.0, 2.0, 3.0],
        }
    )

    result = compare_prediction_frames(frame, frame.copy())

    assert result.status == "exact_match"
    assert result.max_abs_diff == 0.0


def test_compare_prediction_frames_detects_material_drift():
    left = pd.DataFrame(
        {
            "date": pd.date_range("2020-03-31", periods=2, freq="QE-DEC"),
            "forecast": [1.0, 2.0],
        }
    )
    right = pd.DataFrame(
        {
            "date": pd.date_range("2020-03-31", periods=2, freq="QE-DEC"),
            "forecast": [1.0, 2.1],
        }
    )

    result = compare_prediction_frames(left, right, tolerance=1e-4)

    assert result.status == "material_drift"
    assert result.max_abs_diff > 0.0


def test_arima_reference_summary_returns_reference_result():
    values = pd.Series(np.linspace(1.0, 2.5, 20))

    result = arima_reference_summary(values, order=(1, 1, 0))

    assert result.status in {"within_tolerance", "diverged", "reference_failed"}
    assert result.custom_forecast is not None
