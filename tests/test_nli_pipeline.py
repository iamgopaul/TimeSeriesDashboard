import pandas as pd

from src.nli_pipeline import compute_nli_distribution


def test_compute_nli_distribution_returns_summary_for_short_series():
    frame = pd.DataFrame(
        {
            "entity_id": ["1", "1", "2", "2"],
            "entity_label": ["A (1)", "A (1)", "B (2)", "B (2)"],
            "date": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-03-31", "2020-06-30"]),
            "ROA": [0.1, 0.2, 0.3, 0.4],
        }
    )

    result = compute_nli_distribution(frame, "ROA", max_entities=2)

    assert result.distribution.empty
    assert result.processed_entities == 2
    assert result.skipped_short_series == 2
    assert result.failed_entities == 0


def test_compute_nli_distribution_reports_progress_for_processed_entities():
    frame = pd.DataFrame(
        {
            "entity_id": ["1"] * 2 + ["2"] * 2,
            "entity_label": ["A (1)"] * 2 + ["B (2)"] * 2,
            "date": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-03-31", "2020-06-30"]),
            "ROA": [0.1, 0.2, 0.3, 0.4],
        }
    )
    calls = []

    def progress(processed, total, label):
        calls.append((processed, total, label))

    compute_nli_distribution(frame, "ROA", max_entities=2, progress_callback=progress)

    assert calls == [(1, 2, "A (1)"), (2, 2, "B (2)")]


def test_compute_nli_distribution_uses_configured_history_and_order_limits(monkeypatch):
    frame = pd.DataFrame(
        {
            "entity_id": ["1"] * 12,
            "entity_label": ["A (1)"] * 12,
            "date": pd.date_range("2020-03-31", periods=12, freq="QE-DEC"),
            "ROA": [0.1 + (idx * 0.01) for idx in range(12)],
        }
    )
    calls = []

    class DummyNLIResult:
        def __init__(self):
            self.nli_score = 1.2
            self.arima_result = type("ArimaResult", (), {"order": (1, 0, 1)})()
            self.break_result = type("BreakResult", (), {"break_indices": [], "method": "practical"})()

    def fake_compute_nli(series_frame, max_p, max_d, max_q, break_method, strict_whitening):
        calls.append(
            {
                "rows": len(series_frame),
                "max_p": max_p,
                "max_d": max_d,
                "max_q": max_q,
                "break_method": break_method,
                "strict_whitening": strict_whitening,
            }
        )
        return DummyNLIResult()

    monkeypatch.setattr("src.nli_pipeline.compute_nli", fake_compute_nli)

    result = compute_nli_distribution(
        frame,
        "ROA",
        max_entities=1,
        break_method="strict",
        strict_whitening=True,
        max_p=4,
        max_d=1,
        max_q=3,
        min_history=12,
    )

    assert result.successful_entities == 1
    assert calls == [
        {
            "rows": 12,
            "max_p": 4,
            "max_d": 1,
            "max_q": 3,
            "break_method": "strict",
            "strict_whitening": True,
        }
    ]
