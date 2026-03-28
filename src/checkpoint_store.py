from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

import pandas as pd

from src.history_store import _deserialize_value, _serialize_value


DEFAULT_CHECKPOINT_DB_PATH = Path(__file__).resolve().parent.parent / ".dashboard_history" / "checkpoints.sqlite3"


def _ensure_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_checkpoints (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                label TEXT NOT NULL,
                run_key TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_run_checkpoint(
    payload: dict,
    job_type: str,
    status: str,
    label: str,
    run_key: str,
    checkpoint_id: str | None = None,
    db_path: Path = DEFAULT_CHECKPOINT_DB_PATH,
) -> str:
    _ensure_db(db_path)
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    serialized_payload = json.dumps(_serialize_value(payload))
    with sqlite3.connect(db_path) as conn:
        if checkpoint_id is None:
            checkpoint_id = str(uuid4())
            conn.execute(
                """
                INSERT INTO run_checkpoints (
                    id, created_at, updated_at, job_type, status, label, run_key, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (checkpoint_id, now, now, job_type, status, label, run_key, serialized_payload),
            )
        else:
            conn.execute(
                """
                UPDATE run_checkpoints
                SET updated_at = ?, job_type = ?, status = ?, label = ?, run_key = ?, payload_json = ?
                WHERE id = ?
                """,
                (now, job_type, status, label, run_key, serialized_payload, checkpoint_id),
            )
        conn.commit()
    return checkpoint_id


def list_run_checkpoints(
    job_type: str | None = None,
    status: str | None = None,
    run_key: str | None = None,
    db_path: Path = DEFAULT_CHECKPOINT_DB_PATH,
) -> pd.DataFrame:
    _ensure_db(db_path)
    query = """
        SELECT id, created_at, updated_at, job_type, status, label, run_key
        FROM run_checkpoints
    """
    clauses = []
    params: list[str] = []
    if job_type is not None:
        clauses.append("job_type = ?")
        params.append(job_type)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if run_key is not None:
        clauses.append("run_key = ?")
        params.append(run_key)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY updated_at DESC"
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn, params=params)


def load_run_checkpoint(checkpoint_id: str, db_path: Path = DEFAULT_CHECKPOINT_DB_PATH) -> dict:
    _ensure_db(db_path)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT payload_json FROM run_checkpoints WHERE id = ?",
            (checkpoint_id,),
        ).fetchone()
    if row is None:
        raise KeyError(f"Unknown checkpoint: {checkpoint_id}")
    payload = json.loads(row[0])
    return _deserialize_value(payload)


def delete_run_checkpoint(checkpoint_id: str, db_path: Path = DEFAULT_CHECKPOINT_DB_PATH) -> None:
    _ensure_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM run_checkpoints WHERE id = ?", (checkpoint_id,))
        conn.commit()
