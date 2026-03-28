import pandas as pd

from src.checkpoint_store import (
    delete_run_checkpoint,
    list_run_checkpoints,
    load_run_checkpoint,
    save_run_checkpoint,
)


def test_checkpoint_store_roundtrip_and_delete(tmp_path):
    db_path = tmp_path / "checkpoints.sqlite3"
    payload = {
        "analysis_meta": {"analysis_scope": "Full Cleaned Dataset", "target_column": "ROA"},
        "distribution_rows": pd.DataFrame([{"entity_id": "1", "nli_score": 1.2}]),
        "processed_entities": 10,
        "total_entities": 100,
    }

    checkpoint_id = save_run_checkpoint(
        payload,
        job_type="full_dataset_analysis",
        status="running",
        label="ROA run",
        run_key="abc123",
        db_path=db_path,
    )

    entries = list_run_checkpoints(job_type="full_dataset_analysis", run_key="abc123", db_path=db_path)
    assert len(entries) == 1
    assert entries.loc[0, "status"] == "running"

    loaded = load_run_checkpoint(checkpoint_id, db_path=db_path)
    assert loaded["processed_entities"] == 10
    assert float(loaded["distribution_rows"].loc[0, "nli_score"]) == 1.2

    save_run_checkpoint(
        {**loaded, "processed_entities": 25},
        job_type="full_dataset_analysis",
        status="completed",
        label="ROA run",
        run_key="abc123",
        checkpoint_id=checkpoint_id,
        db_path=db_path,
    )
    updated_entries = list_run_checkpoints(job_type="full_dataset_analysis", run_key="abc123", db_path=db_path)
    assert updated_entries.loc[0, "status"] == "completed"

    delete_run_checkpoint(checkpoint_id, db_path=db_path)
    assert list_run_checkpoints(db_path=db_path).empty
