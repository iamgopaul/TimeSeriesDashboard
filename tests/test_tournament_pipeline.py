import pandas as pd

from src.tournament_pipeline import (
    build_forecasting_tournament_result,
    compute_forecasting_tournament,
    compute_forecasting_tournament_chunk,
)


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


def test_compute_forecasting_tournament_respects_minimum_history(monkeypatch):
    frame = pd.DataFrame(
        {
            "entity_id": ["1"] * 12,
            "entity_label": ["A (1)"] * 12,
            "date": pd.date_range("2020-03-31", periods=12, freq="QE-DEC"),
            "ROA": [0.1 + (idx * 0.01) for idx in range(12)],
            "assets": [100.0] * 12,
        }
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Forecast routines should not run for insufficient history.")

    monkeypatch.setattr("src.tournament_pipeline.compute_nli", fail_if_called)
    monkeypatch.setattr("src.tournament_pipeline.run_naive_forecast", fail_if_called)
    monkeypatch.setattr("src.tournament_pipeline.run_arima_forecast", fail_if_called)
    monkeypatch.setattr("src.tournament_pipeline.run_chronos_forecast", fail_if_called)

    result = compute_forecasting_tournament(
        frame,
        "ROA",
        holdout_size=2,
        max_entities=1,
        minimum_history=20,
    )

    assert result.successful_entities == 0
    assert result.failed_entities == 1
    assert "need >= 20 rows" in result.failure_examples.loc[0, "error"]


def test_compute_forecasting_tournament_chunk_resume_matches_final_result(monkeypatch):
    frame = pd.DataFrame(
        {
            "entity_id": ["1"] * 12 + ["2"] * 12,
            "entity_label": ["A (1)"] * 12 + ["B (2)"] * 12,
            "date": list(pd.date_range("2020-03-31", periods=12, freq="QE-DEC")) * 2,
            "ROA": [0.1 + (idx * 0.01) for idx in range(12)] + [0.2 + (idx * 0.01) for idx in range(12)],
            "assets": [100.0] * 24,
        }
    )

    class DummyNLIResult:
        def __init__(self, entity_id):
            self.nli_score = 1.0 if entity_id == "1" else 2.0
            self.break_result = type("BreakResult", (), {"break_indices": [], "method": "strict"})()
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
                [{"model": model, "MASE": mase_value, "interval_coverage_rate": 1.0, "median_interval_width": 0.0}]
            )

    def fake_compute_nli(series_frame, **kwargs):
        first_value = float(series_frame["value"].iloc[0])
        return DummyNLIResult("1" if first_value < 0.2 else "2")

    monkeypatch.setattr("src.tournament_pipeline.compute_nli", fake_compute_nli)
    monkeypatch.setattr("src.tournament_pipeline.run_naive_forecast", lambda *args, **kwargs: DummyForecast("Naive", 1.0))
    monkeypatch.setattr("src.tournament_pipeline.run_arima_forecast", lambda *args, **kwargs: DummyForecast("ARIMA", 0.8))
    monkeypatch.setattr("src.tournament_pipeline.run_chronos_forecast", lambda *args, **kwargs: DummyForecast("Chronos", 0.6))

    first_chunk = compute_forecasting_tournament_chunk(frame, "ROA", holdout_size=2, start_index=0, batch_size=1)
    second_chunk = compute_forecasting_tournament_chunk(
        frame,
        "ROA",
        holdout_size=2,
        start_index=first_chunk.next_index,
        batch_size=1,
    )
    combined = build_forecasting_tournament_result(
        firm_metrics_rows=pd.concat([first_chunk.firm_metrics, second_chunk.firm_metrics], ignore_index=True),
        forecast_panel_rows=pd.concat([first_chunk.forecast_panel, second_chunk.forecast_panel], ignore_index=True),
        firm_forecast_summary_rows=pd.concat(
            [first_chunk.firm_forecast_summary, second_chunk.firm_forecast_summary], ignore_index=True
        ),
        failures=pd.concat([first_chunk.failures, second_chunk.failures], ignore_index=True),
        processed_entities=first_chunk.processed_entities + second_chunk.processed_entities,
        skipped_entities=first_chunk.skipped_entities + second_chunk.skipped_entities,
        requested_entities=2,
    )

    assert combined.processed_entities == 2
    assert combined.successful_entities == 2
    assert set(combined.firm_metrics["entity_id"].astype(str)) == {"1", "2"}
    assert set(combined.forecast_panel["entity_id"].astype(str)) == {"1", "2"}
