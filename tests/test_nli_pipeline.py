import pandas as pd

from src.nli_pipeline import build_nli_distribution_result, compute_nli_distribution, compute_nli_distribution_chunk


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


def test_compute_nli_distribution_chunk_resume_matches_final_result(monkeypatch):
    frame = pd.DataFrame(
        {
            "entity_id": ["1"] * 12 + ["2"] * 12,
            "entity_label": ["A (1)"] * 12 + ["B (2)"] * 12,
            "date": list(pd.date_range("2020-03-31", periods=12, freq="QE-DEC")) * 2,
            "ROA": [0.1 + (idx * 0.01) for idx in range(12)] + [0.2 + (idx * 0.01) for idx in range(12)],
        }
    )

    class DummyNLIResult:
        def __init__(self, entity_id):
            self.nli_score = 1.0 if entity_id == "1" else 2.0
            self.arima_result = type("ArimaResult", (), {"order": (1, 0, 1)})()
            self.break_result = type("BreakResult", (), {"break_indices": [1], "method": "strict"})()

    def fake_compute_nli(series_frame, **kwargs):
        first_value = float(series_frame["value"].iloc[0])
        return DummyNLIResult("1" if first_value < 0.2 else "2")

    monkeypatch.setattr("src.nli_pipeline.compute_nli", fake_compute_nli)

    first_chunk = compute_nli_distribution_chunk(frame, "ROA", start_index=0, batch_size=1)
    second_chunk = compute_nli_distribution_chunk(frame, "ROA", start_index=first_chunk.next_index, batch_size=1)
    combined = build_nli_distribution_result(
        pd.concat([first_chunk.distribution, second_chunk.distribution], ignore_index=True),
        pd.concat([first_chunk.failures, second_chunk.failures], ignore_index=True),
        processed_entities=first_chunk.processed_entities + second_chunk.processed_entities,
        skipped_short_series=first_chunk.skipped_short_series + second_chunk.skipped_short_series,
        requested_entities=2,
    )

    assert combined.processed_entities == 2
    assert combined.successful_entities == 2
    assert combined.distribution["entity_id"].tolist() == ["2", "1"]
