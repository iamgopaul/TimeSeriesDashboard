from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime
from io import StringIO
import importlib
import json
from pathlib import Path
import sqlite3
from typing import Any, get_args, get_origin
from uuid import uuid4

import numpy as np
import pandas as pd


DEFAULT_HISTORY_DB_PATH = Path(__file__).resolve().parent.parent / ".dashboard_history" / "history.sqlite3"


def _ensure_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history_runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                snapshot_type TEXT NOT NULL,
                label TEXT NOT NULL,
                analysis_scope TEXT,
                target_column TEXT,
                entity_label TEXT,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _serialize_dataclass_value(value: Any) -> dict[str, Any]:
    return {
        field.name: _serialize_value(getattr(value, field.name))
        for field in fields(value)
    }


def _resolve_annotation(annotation: Any, module_globals: dict[str, Any]) -> Any:
    if not isinstance(annotation, str):
        return annotation
    cleaned = annotation.strip()
    if cleaned in module_globals:
        return module_globals[cleaned]
    if "|" in cleaned:
        candidates = [part.strip() for part in cleaned.split("|")]
        for candidate in candidates:
            if candidate == "None":
                continue
            if candidate in module_globals:
                return module_globals[candidate]
    return annotation


def _coerce_dataclass_field(field_type: Any, value: Any) -> Any:
    if value is None:
        return None
    origin = get_origin(field_type)
    if origin is None:
        if isinstance(field_type, type) and is_dataclass(field_type) and isinstance(value, dict):
            return _rebuild_dataclass(field_type, value)
        return value
    for candidate in get_args(field_type):
        if candidate is type(None):
            continue
        coerced = _coerce_dataclass_field(candidate, value)
        if coerced is not value or (isinstance(candidate, type) and is_dataclass(candidate) and isinstance(value, dict)):
            return coerced
    return value


def _rebuild_dataclass(cls: type, payload: dict[str, Any]) -> Any:
    module_globals = vars(importlib.import_module(cls.__module__))
    kwargs = {}
    for field in fields(cls):
        if field.name not in payload:
            continue
        annotation = cls.__annotations__.get(field.name, field.type)
        resolved_type = _resolve_annotation(annotation, module_globals)
        kwargs[field.name] = _coerce_dataclass_field(resolved_type, payload[field.name])
    return cls(**kwargs)


def _serialize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return {"__kind__": "path", "value": str(value)}
    if isinstance(value, pd.Timestamp):
        return {"__kind__": "timestamp", "value": value.isoformat()}
    if isinstance(value, pd.DataFrame):
        return {"__kind__": "dataframe", "value": value.to_json(orient="split", date_format="iso")}
    if isinstance(value, pd.Series):
        return {"__kind__": "series", "value": value.to_json(orient="split", date_format="iso")}
    if isinstance(value, np.ndarray):
        return {"__kind__": "ndarray", "value": value.tolist()}
    if isinstance(value, tuple):
        return {"__kind__": "tuple", "value": [_serialize_value(item) for item in value]}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {
            "__kind__": "dataclass",
            "module": value.__class__.__module__,
            "class": value.__class__.__name__,
            "value": _serialize_dataclass_value(value),
        }
    return {"__kind__": "repr", "value": repr(value)}


def _deserialize_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_deserialize_value(item) for item in value]
    if isinstance(value, dict):
        kind = value.get("__kind__")
        if kind == "path":
            return Path(value["value"])
        if kind == "timestamp":
            return pd.Timestamp(value["value"])
        if kind == "dataframe":
            return pd.read_json(StringIO(value["value"]), orient="split")
        if kind == "series":
            return pd.read_json(StringIO(value["value"]), orient="split", typ="series")
        if kind == "ndarray":
            return np.asarray(value["value"])
        if kind == "tuple":
            return tuple(_deserialize_value(item) for item in value["value"])
        if kind == "repr":
            return value["value"]
        if kind == "dataclass":
            module = importlib.import_module(value["module"])
            cls = getattr(module, value["class"])
            payload = _deserialize_value(value["value"])
            return _rebuild_dataclass(cls, payload)
        return {key: _deserialize_value(item) for key, item in value.items()}
    return value


def save_history_snapshot(
    payload: dict[str, Any],
    snapshot_type: str,
    label: str,
    analysis_meta: dict[str, Any] | None = None,
    db_path: Path = DEFAULT_HISTORY_DB_PATH,
) -> str:
    _ensure_db(db_path)
    snapshot_id = str(uuid4())
    created_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    meta = analysis_meta or {}
    serialized_payload = json.dumps(_serialize_value(payload))
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO history_runs (
                id, created_at, snapshot_type, label, analysis_scope, target_column, entity_label, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                created_at,
                snapshot_type,
                label,
                meta.get("analysis_scope"),
                meta.get("target_column"),
                meta.get("entity_label"),
                serialized_payload,
            ),
        )
        conn.commit()
    return snapshot_id


def list_history_entries(db_path: Path = DEFAULT_HISTORY_DB_PATH) -> pd.DataFrame:
    _ensure_db(db_path)
    with sqlite3.connect(db_path) as conn:
        frame = pd.read_sql_query(
            """
            SELECT id, created_at, snapshot_type, label, analysis_scope, target_column, entity_label
            FROM history_runs
            ORDER BY created_at DESC
            """,
            conn,
        )
    return frame


def load_history_snapshot(snapshot_id: str, db_path: Path = DEFAULT_HISTORY_DB_PATH) -> dict[str, Any]:
    _ensure_db(db_path)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT payload_json FROM history_runs WHERE id = ?",
            (snapshot_id,),
        ).fetchone()
    if row is None:
        raise KeyError(f"Unknown history snapshot: {snapshot_id}")
    payload = json.loads(row[0])
    return _deserialize_value(payload)


def delete_history_snapshot(snapshot_id: str, db_path: Path = DEFAULT_HISTORY_DB_PATH) -> None:
    _ensure_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM history_runs WHERE id = ?", (snapshot_id,))
        conn.commit()
