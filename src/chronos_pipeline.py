from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os

import numpy as np
import pandas as pd

from src.metrics import mase, mean_error, smape


@dataclass
class ChronosForecastResult:
    available: bool
    message: str
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    model_used: str | None = None


def chronos_import_status() -> tuple[bool, str]:
    try:  # pragma: no cover - availability depends on local install
        import torch  # noqa: F401
        from chronos import ChronosPipeline  # noqa: F401
        return True, "Chronos is available."
    except Exception as exc:  # pragma: no cover - availability depends on local install
        return False, str(exc)


@lru_cache(maxsize=1)
def get_chronos_pipeline(model_name: str):
    import torch  # pragma: no cover - expensive external dependency
    from chronos import ChronosPipeline  # pragma: no cover - expensive external dependency

    # Force offline loading so proxy-blocked environments fail fast using only
    # locally cached model files.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    dtype = torch.float32
    return ChronosPipeline.from_pretrained(
        model_name,
        device_map="cpu",
        dtype=dtype,
        local_files_only=True,
    )


def load_chronos_pipeline(
    preferred_model: str = "amazon/chronos-bolt-base",
) -> tuple[object | None, str | None, str]:
    candidates = [preferred_model]
    if preferred_model != "amazon/chronos-t5-base":
        candidates.append("amazon/chronos-t5-base")

    errors: list[str] = []
    for candidate in candidates:
        try:
            pipeline = get_chronos_pipeline(candidate)
            message = (
                "Chronos forecast completed."
                if candidate == preferred_model
                else f"Primary model `{preferred_model}` was unavailable; using `{candidate}` instead."
            )
            return pipeline, candidate, message
        except Exception as exc:  # pragma: no cover - depends on local environment
            errors.append(f"{candidate}: {exc}")

    return None, None, "Chronos models failed to load. " + " | ".join(errors)


def expanding_window_forecast(
    series_frame: pd.DataFrame,
    holdout_size: int,
    model_name: str = "amazon/chronos-bolt-base",
    deterministic: bool = True,
    seed: int = 17,
    num_samples: int = 20,
) -> ChronosForecastResult:
    available, message = chronos_import_status()
    if not available:
        return ChronosForecastResult(
            available=False,
            message=message,
            predictions=pd.DataFrame(),
            metrics=pd.DataFrame(),
            model_used=None,
        )

    import torch  # pragma: no cover - external dependency

    working = series_frame.sort_values("date").reset_index(drop=True)
    if holdout_size <= 0 or holdout_size >= len(working):
        raise ValueError("Holdout size must be positive and smaller than the series length.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    pipeline, resolved_model_name, load_message = load_chronos_pipeline(model_name)
    if pipeline is None or resolved_model_name is None:
        return ChronosForecastResult(
            available=False,
            message=load_message,
            predictions=pd.DataFrame(),
            metrics=pd.DataFrame(),
            model_used=None,
        )

    split_index = len(working) - holdout_size
    predictions = []

    for idx in range(split_index, len(working)):
        context = working.loc[: idx - 1, "value"].to_numpy(dtype=np.float32)
        actual = float(working.loc[idx, "value"])
        date = working.loc[idx, "date"]
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        forecast_samples = pipeline.predict(
            context=torch.tensor(context),
            prediction_length=1,
            num_samples=1 if deterministic else num_samples,
        )
        samples = np.asarray(forecast_samples[0], dtype=float).reshape(-1)
        if deterministic:
            quantiles = np.asarray([samples[0], samples[0], samples[0]], dtype=float)
        else:
            quantiles = np.quantile(samples, [0.1, 0.5, 0.9], axis=0).reshape(-1)
        predictions.append(
            {
                "date": date,
                "actual": actual,
                "forecast": float(quantiles[1]),
                "q10": float(quantiles[0]),
                "q50": float(quantiles[1]),
                "q90": float(quantiles[2]),
                "error": float(quantiles[1] - actual),
                "model": "Chronos",
                "model_used": resolved_model_name,
                "inference_mode": "deterministic" if deterministic else "sampled",
                "seed": int(seed),
                "sample_count": 1 if deterministic else int(num_samples),
            }
        )

    prediction_frame = pd.DataFrame(predictions)
    prediction_frame["inside_interval"] = (
        (prediction_frame["actual"] >= prediction_frame["q10"])
        & (prediction_frame["actual"] <= prediction_frame["q90"])
    )
    prediction_frame["interval_width"] = prediction_frame["q90"] - prediction_frame["q10"]
    quantile_order_valid = bool(
        ((prediction_frame["q10"] <= prediction_frame["q50"]) & (prediction_frame["q50"] <= prediction_frame["q90"])).all()
    )
    nonnegative_interval_width = bool((prediction_frame["interval_width"] >= 0).all())
    insample = working.loc[: split_index - 1, "value"].to_numpy(dtype=float)
    metric_frame = pd.DataFrame(
        [
            {
                "model": "Chronos",
                "MASE": mase(
                    prediction_frame["actual"].to_numpy(),
                    prediction_frame["forecast"].to_numpy(),
                    insample,
                ),
                "sMAPE": smape(
                    prediction_frame["actual"].to_numpy(),
                    prediction_frame["forecast"].to_numpy(),
                ),
                "mean_error": mean_error(
                    prediction_frame["actual"].to_numpy(),
                    prediction_frame["forecast"].to_numpy(),
                ),
                "median_interval_width": float(
                    np.median(prediction_frame["q90"] - prediction_frame["q10"])
                ),
                "interval_coverage_rate": float(prediction_frame["inside_interval"].mean()),
                "quantile_order_valid": quantile_order_valid,
                "interval_width_nonnegative": nonnegative_interval_width,
                "model_used": resolved_model_name,
                "inference_mode": "deterministic" if deterministic else "sampled",
                "seed": int(seed),
                "sample_count": 1 if deterministic else int(num_samples),
            }
        ]
    )

    return ChronosForecastResult(
        available=True,
        message=(
            f"{load_message} Inference mode: "
            f"{'deterministic' if deterministic else f'sampled ({int(num_samples)} samples)'}."
        ),
        predictions=prediction_frame,
        metrics=metric_frame,
        model_used=resolved_model_name,
    )
