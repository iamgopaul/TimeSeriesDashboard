from __future__ import annotations

import pandas as pd

from src.history_store import delete_history_snapshot, list_history_entries, load_history_snapshot, save_history_snapshot
from src.validation import RepeatRunCheckResult


def test_history_snapshot_roundtrip(tmp_path) -> None:
    db_path = tmp_path / "history.sqlite3"
    payload = {
        "analysis_result": {
            "analysis_scope": "Single entity diagnostics",
            "entity_series": pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-03-31", "2024-06-30"]),
                    "value": [0.1, 0.2],
                }
            ),
            "chronos_predictions": pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-03-31", "2024-06-30"]),
                    "forecast": [0.15, 0.25],
                    "q025": [0.05, 0.1],
                    "q10": [0.08, 0.12],
                    "q25": [0.1, 0.15],
                    "q50": [0.15, 0.25],
                    "q75": [0.2, 0.3],
                    "q90": [0.22, 0.32],
                    "q975": [0.25, 0.35],
                }
            ),
        },
        "validation_result": {
            "naive_repeat": RepeatRunCheckResult(
                status="exact_match",
                max_abs_diff=0.0,
                compared_points=2,
                message="Repeat run matched exactly.",
            )
        },
        "step_statuses": [{"step": "Compute NLI", "status": "completed"}],
        "tuple_value": (1, 2, 3),
    }
    meta = {
        "analysis_scope": "Single entity diagnostics",
        "target_column": "ROA",
        "entity_label": "Firm A",
    }

    snapshot_id = save_history_snapshot(
        payload,
        snapshot_type="analysis",
        label="Single entity diagnostics | ROA | Firm A",
        analysis_meta=meta,
        db_path=db_path,
    )

    entries = list_history_entries(db_path)
    assert len(entries) == 1
    assert entries.loc[0, "id"] == snapshot_id
    assert entries.loc[0, "target_column"] == "ROA"

    loaded = load_history_snapshot(snapshot_id, db_path)
    assert list(loaded["analysis_result"]["entity_series"]["value"]) == [0.1, 0.2]
    assert "q025" in loaded["analysis_result"]["chronos_predictions"].columns
    assert "q975" in loaded["analysis_result"]["chronos_predictions"].columns
    assert loaded["validation_result"]["naive_repeat"].status == "exact_match"
    assert loaded["tuple_value"] == (1, 2, 3)


def test_history_snapshot_delete(tmp_path) -> None:
    db_path = tmp_path / "history.sqlite3"
    snapshot_id = save_history_snapshot(
        {"analysis_result": None},
        snapshot_type="analysis",
        label="Empty snapshot",
        analysis_meta=None,
        db_path=db_path,
    )

    assert len(list_history_entries(db_path)) == 1
    delete_history_snapshot(snapshot_id, db_path)
    assert list_history_entries(db_path).empty
