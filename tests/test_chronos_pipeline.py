import numpy as np
import pandas as pd

from src.chronos_pipeline import chronos_offline_mode_enabled, expanding_window_forecast, load_chronos_pipeline


def test_load_chronos_pipeline_falls_back(monkeypatch):
    calls = []

    def fake_get_pipeline(model_name, local_files_only=False):
        calls.append((model_name, local_files_only))
        if model_name == "amazon/chronos-bolt-base":
            raise TypeError("input_patch_size")
        return object()

    monkeypatch.setattr("src.chronos_pipeline.get_chronos_pipeline", fake_get_pipeline)
    monkeypatch.delenv("CHRONOS_OFFLINE", raising=False)

    pipeline, resolved_model, message = load_chronos_pipeline("amazon/chronos-bolt-base")

    assert pipeline is not None
    assert resolved_model == "amazon/chronos-t5-base"
    assert "using `amazon/chronos-t5-base` instead" in message
    assert calls == [("amazon/chronos-bolt-base", False), ("amazon/chronos-t5-base", False)]


def test_chronos_offline_mode_env_flag(monkeypatch):
    monkeypatch.setenv("CHRONOS_OFFLINE", "1")
    assert chronos_offline_mode_enabled()
    monkeypatch.setenv("CHRONOS_OFFLINE", "false")
    assert not chronos_offline_mode_enabled()


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
    assert (result.predictions["q05"] == result.predictions["q50"]).all()
    assert (result.predictions["q10"] == result.predictions["q50"]).all()
    assert (result.predictions["q25"] == result.predictions["q50"]).all()
    assert (result.predictions["q50"] == result.predictions["q90"]).all()
    assert (result.predictions["q95"] == result.predictions["q50"]).all()
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
    assert {"q025", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q975"}.issubset(result.predictions.columns)
    assert (result.predictions["q025"] <= result.predictions["q05"]).all()
    assert (result.predictions["q05"] <= result.predictions["q10"]).all()
    assert (result.predictions["q10"] <= result.predictions["q25"]).all()
    assert (result.predictions["q25"] <= result.predictions["q50"]).all()
    assert (result.predictions["q50"] <= result.predictions["q75"]).all()
    assert (result.predictions["q75"] <= result.predictions["q90"]).all()
    assert (result.predictions["q90"] <= result.predictions["q95"]).all()
    assert (result.predictions["q95"] <= result.predictions["q975"]).all()
    assert (result.predictions["interval_width_50"] <= result.predictions["interval_width"]).all()
    assert (result.predictions["interval_width"] <= result.predictions["interval_width_90"]).all()
    assert (result.predictions["interval_width_90"] <= result.predictions["interval_width_95"]).all()
    assert not bool(result.metrics["intervals_collapsed"].iloc[0])


def test_chronos_handles_positional_predict_and_direct_quantiles(monkeypatch):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-03-31", periods=6, freq="QE-DEC"),
            "value": np.linspace(1.0, 1.5, 6),
        }
    )

    class FakeBoltPipeline:
        def predict(self, *args, **kwargs):
            if "context" in kwargs:
                raise TypeError("ChronosPipeline.predict() got an unexpected keyword argument 'context'")
            if "num_samples" in kwargs:
                raise TypeError("ChronosBoltPipeline.predict() got an unexpected keyword argument 'num_samples'")
            assert len(args) == 1
            return np.asarray([[[1.0], [1.2], [1.4], [1.6], [1.8], [2.0], [2.2], [2.4], [2.6]]])

    monkeypatch.setattr("src.chronos_pipeline.chronos_import_status", lambda: (True, "ok"))
    monkeypatch.setattr(
        "src.chronos_pipeline.load_chronos_pipeline",
        lambda model_name: (FakeBoltPipeline(), "amazon/chronos-bolt-base", "loaded"),
    )

    result = expanding_window_forecast(frame, holdout_size=2, deterministic=False, seed=99, num_samples=5)

    assert result.available
    assert (result.predictions["quantile_source"] == "direct_quantiles").all()
    assert (result.predictions["q05"] <= result.predictions["q10"]).all()
    assert (result.predictions["q10"] <= result.predictions["q50"]).all()
    assert (result.predictions["q50"] <= result.predictions["q90"]).all()
    assert (result.predictions["q90"] <= result.predictions["q95"]).all()
