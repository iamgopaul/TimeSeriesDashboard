import numpy as np

from src.metrics import mase, mean_error, smape


def test_metrics_return_finite_values():
    actual = np.array([1.0, 2.0, 3.0])
    forecast = np.array([1.1, 1.9, 3.2])
    insample = np.array([0.8, 1.2, 1.8, 2.4])

    assert smape(actual, forecast) > 0
    assert np.isfinite(mase(actual, forecast, insample))
    assert np.isfinite(mean_error(actual, forecast))
