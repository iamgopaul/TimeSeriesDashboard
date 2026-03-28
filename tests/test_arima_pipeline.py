import numpy as np
import pandas as pd

from src.arima_pipeline import expanding_window_forecast, fit_best_arima, naive_expanding_window_forecast


def test_fit_best_arima_returns_valid_order():
    rng = np.random.default_rng(7)
    values = pd.Series(np.sin(np.linspace(0, 2, 24)) + rng.normal(0, 0.05, 24))

    result = fit_best_arima(values, max_p=1, max_d=1, max_q=1)

    assert len(result.order) == 3
    assert not result.candidate_table.empty


def test_expanding_window_forecast_returns_holdout_predictions():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-03-31", periods=16, freq="QE-DEC"),
            "value": np.linspace(1.0, 2.5, 16),
        }
    )

    result = expanding_window_forecast(frame, order=(1, 1, 0), holdout_size=4)

    assert len(result.predictions) == 4
    assert set(result.metrics.columns) >= {"model", "MASE", "sMAPE", "mean_error"}


def test_naive_expanding_window_forecast_carries_forward_last_value():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-03-31", periods=6, freq="QE-DEC"),
            "value": [1.0, 1.2, 1.4, 1.1, 1.3, 1.6],
        }
    )

    result = naive_expanding_window_forecast(frame, holdout_size=2)

    assert list(result.predictions["forecast"]) == [1.1, 1.3]
    assert result.metrics["model"].iloc[0] == "Naive"
    assert int(result.metrics["failed_windows"].iloc[0]) == 0


def test_strict_arima_mode_marks_failed_windows(monkeypatch):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-03-31", periods=12, freq="QE-DEC"),
            "value": np.linspace(1.0, 2.1, 12),
        }
    )

    def fail_build_fit_result(values, order):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.arima_pipeline._build_fit_result", fail_build_fit_result)
    result = expanding_window_forecast(frame, order=(1, 1, 0), holdout_size=3, evaluation_mode="strict")

    assert result.evaluation_mode == "strict"
    assert set(result.predictions["fit_strategy"]) == {"failed"}
    assert result.predictions["forecast"].isna().all()
    assert int(result.metrics["failed_windows"].iloc[0]) == 3
