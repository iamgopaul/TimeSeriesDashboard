import pandas as pd

from src.tournament_pipeline import compute_forecasting_tournament


def test_compute_forecasting_tournament_returns_firm_metrics(monkeypatch):
    frame = pd.DataFrame(
        {
            "entity_id": ["1"] * 12,
            "entity_label": ["A (1)"] * 12,
            "date": pd.date_range("2020-03-31", periods=12, freq="QE-DEC"),
            "ROA": [0.1 + (idx * 0.01) for idx in range(12)],
            "assets": [100.0] * 12,
        }
    )

    class DummyNLIResult:
        def __init__(self):
            self.nli_score = 1.5
            self.break_result = type("BreakResult", (), {"break_indices": [], "method": "r-strucchange"})()
            self.arima_result = type("ArimaResult", (), {"order": (1, 1, 0)})()

    class DummyForecast:
        def __init__(self, model, mase_value):
            self.available = True
            self.predictions = pd.DataFrame(
                {
                    "date": pd.date_range("2022-09-30", periods=2, freq="QE-DEC"),
                    "actual": [1.0, 1.1],
                    "forecast": [1.0, 1.0],
                    "error": [0.0, -0.1],
                    "model": [model, model],
                }
            )
            self.metrics = pd.DataFrame(
                [
                    {
                        "model": model,
                        "MASE": mase_value,
                        "interval_coverage_rate": 1.0,
                        "median_interval_width": 0.0,
                    }
                ]
            )

    monkeypatch.setattr("src.tournament_pipeline.compute_nli", lambda *args, **kwargs: DummyNLIResult())
    monkeypatch.setattr("src.tournament_pipeline.run_naive_forecast", lambda *args, **kwargs: DummyForecast("Naive", 1.0))
    monkeypatch.setattr("src.tournament_pipeline.run_arima_forecast", lambda *args, **kwargs: DummyForecast("ARIMA", 0.8))
    monkeypatch.setattr("src.tournament_pipeline.run_chronos_forecast", lambda *args, **kwargs: DummyForecast("Chronos", 0.6))

    result = compute_forecasting_tournament(frame, "ROA", holdout_size=2, max_entities=1)

    assert result.successful_entities == 1
    assert not result.firm_metrics.empty
    assert bool(result.firm_metrics["chronos_beats_arima"].iloc[0])
    assert not result.forecast_panel.empty
    assert set(result.forecast_panel["model"]) == {"Naive", "ARIMA", "Chronos"}
    assert {"entity_id", "entity_label", "absolute_error", "holdout_step", "q025", "q975"}.issubset(
        result.forecast_panel.columns
    )
    assert not result.firm_forecast_summary.empty
    assert result.firm_forecast_summary.loc[0, "winner"] == "Chronos"
    assert result.firm_forecast_summary.loc[0, "holdout_points"] == 2
