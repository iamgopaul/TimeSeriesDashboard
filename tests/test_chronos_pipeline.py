import numpy as np
import pandas as pd

from src.chronos_pipeline import expanding_window_forecast, load_chronos_pipeline


def test_load_chronos_pipeline_falls_back(monkeypatch):
    calls = []

    def fake_get_pipeline(model_name):
        calls.append(model_name)
        if model_name == "amazon/chronos-bolt-base":
            raise TypeError("input_patch_size")
        return object()

    monkeypatch.setattr("src.chronos_pipeline.get_chronos_pipeline", fake_get_pipeline)

    pipeline, resolved_model, message = load_chronos_pipeline("amazon/chronos-bolt-base")

    assert pipeline is not None
    assert resolved_model == "amazon/chronos-t5-base"
    assert "using `amazon/chronos-t5-base` instead" in message
    assert calls == ["amazon/chronos-bolt-base", "amazon/chronos-t5-base"]


def test_chronos_deterministic_mode_collapses_quantiles(monkeypatch):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-03-31", periods=6, freq="QE-DEC"),
            "value": np.linspace(1.0, 1.5, 6),
        }
    )

    class FakePipeline:
        def predict(self, context, prediction_length, num_samples):
            assert prediction_length == 1
            assert num_samples == 1
            return np.asarray([[2.5]])

    monkeypatch.setattr("src.chronos_pipeline.chronos_import_status", lambda: (True, "ok"))
    monkeypatch.setattr("src.chronos_pipeline.load_chronos_pipeline", lambda model_name: (FakePipeline(), model_name, "loaded"))

    result = expanding_window_forecast(frame, holdout_size=2, deterministic=True, seed=99)

    assert result.available
    assert (result.predictions["q025"] == result.predictions["q50"]).all()
    assert (result.predictions["q10"] == result.predictions["q50"]).all()
    assert (result.predictions["q25"] == result.predictions["q50"]).all()
    assert (result.predictions["q50"] == result.predictions["q90"]).all()
    assert (result.predictions["q75"] == result.predictions["q50"]).all()
    assert (result.predictions["q975"] == result.predictions["q50"]).all()
    assert (result.predictions["sample_count"] == 1).all()
    assert bool(result.metrics["quantile_order_valid"].iloc[0])
    assert bool(result.metrics["interval_width_nonnegative"].iloc[0])
    assert bool(result.metrics["intervals_collapsed"].iloc[0])


def test_chronos_sampled_mode_emits_nested_intervals(monkeypatch):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-03-31", periods=6, freq="QE-DEC"),
            "value": np.linspace(1.0, 1.5, 6),
        }
    )

    class FakePipeline:
        def predict(self, context, prediction_length, num_samples):
            assert prediction_length == 1
            assert num_samples == 5
            return np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0]])

    monkeypatch.setattr("src.chronos_pipeline.chronos_import_status", lambda: (True, "ok"))
    monkeypatch.setattr("src.chronos_pipeline.load_chronos_pipeline", lambda model_name: (FakePipeline(), model_name, "loaded"))

    result = expanding_window_forecast(frame, holdout_size=2, deterministic=False, seed=99, num_samples=5)

    assert result.available
    assert {"q025", "q10", "q25", "q50", "q75", "q90", "q975"}.issubset(result.predictions.columns)
    assert (result.predictions["q025"] <= result.predictions["q10"]).all()
    assert (result.predictions["q10"] <= result.predictions["q25"]).all()
    assert (result.predictions["q25"] <= result.predictions["q50"]).all()
    assert (result.predictions["q50"] <= result.predictions["q75"]).all()
    assert (result.predictions["q75"] <= result.predictions["q90"]).all()
    assert (result.predictions["q90"] <= result.predictions["q975"]).all()
    assert (result.predictions["interval_width_50"] <= result.predictions["interval_width"]).all()
    assert (result.predictions["interval_width"] <= result.predictions["interval_width_95"]).all()
    assert not bool(result.metrics["intervals_collapsed"].iloc[0])
