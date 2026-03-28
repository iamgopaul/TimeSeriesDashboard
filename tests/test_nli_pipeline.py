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
