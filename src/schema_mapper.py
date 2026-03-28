from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd


@dataclass(frozen=True)
class CanonicalField:
    name: str
    required: bool
    aliases: tuple[str, ...]
    description: str


CANONICAL_FIELDS: tuple[CanonicalField, ...] = (
    CanonicalField(
        name="entity_id",
        required=True,
        aliases=("gvkey", "firm_id", "company_id", "id", "ticker", "tic"),
        description="Unique firm or entity identifier.",
    ),
    CanonicalField(
        name="date",
        required=True,
        aliases=("datadate", "date", "period", "quarter_end", "observation_date"),
        description="Observation date for each time-series point.",
    ),
    CanonicalField(
        name="entity_name",
        required=False,
        aliases=("conm", "company_name", "name", "firm_name"),
        description="Readable company or entity label.",
    ),
    CanonicalField(
        name="industry",
        required=False,
        aliases=("naics", "sic", "sector", "industry"),
        description="Industry or sector identifier.",
    ),
    CanonicalField(
        name="assets",
        required=False,
        aliases=("atq", "assets", "total_assets"),
        description="Assets or resource base column.",
    ),
    CanonicalField(
        name="equity",
        required=False,
        aliases=("seqq", "ceqq", "equity", "book_equity"),
        description="Equity or shareholder value column.",
    ),
    CanonicalField(
        name="net_income",
        required=False,
        aliases=("niq", "net_income", "income"),
        description="Net income column.",
    ),
    CanonicalField(
        name="deletion_reason",
        required=False,
        aliases=("dlrsn", "deletion_reason", "del_reason"),
        description="Deletion or terminal event code.",
    ),
)

WRDS_PRIORITY_METRICS = ("ROA", "ROE", "saleq", "niq")
GENERIC_METRIC_HINTS = (
    "value",
    "target",
    "metric",
    "return",
    "performance",
    "sales",
    "revenue",
    "forecast_target",
)


@dataclass
class SchemaInference:
    mapping: Dict[str, str]
    missing_required: List[str]
    warnings: List[str]
    suggested_metrics: List[str]


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def infer_schema_mapping(columns: Iterable[str]) -> SchemaInference:
    columns = list(columns)
    normalized = {_normalize(column): column for column in columns}
    mapping: Dict[str, str] = {}
    warnings: List[str] = []

    for field in CANONICAL_FIELDS:
        match = next(
            (normalized[alias] for alias in field.aliases if alias in normalized),
            None,
        )
        if match is not None:
            mapping[field.name] = match

    missing_required = [field.name for field in CANONICAL_FIELDS if field.required and field.name not in mapping]
    if missing_required:
        warnings.append(
            "Manual mapping is required for: " + ", ".join(sorted(missing_required))
        )

    return SchemaInference(
        mapping=mapping,
        missing_required=missing_required,
        warnings=warnings,
        suggested_metrics=suggest_metric_columns(columns),
    )


def build_mapping_options(columns: Iterable[str]) -> List[str]:
    return [""] + list(columns)


def validate_mapping(mapping: Dict[str, str], columns: Iterable[str]) -> List[str]:
    columns = set(columns)
    errors: List[str] = []

    for field in CANONICAL_FIELDS:
        source = mapping.get(field.name, "")
        if field.required and not source:
            errors.append(f"`{field.name}` is required.")
        if source and source not in columns:
            errors.append(f"`{source}` is not present in the uploaded file.")

    return errors


def apply_schema_mapping(frame: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    rename_map = {
        source: canonical
        for canonical, source in mapping.items()
        if source and source in frame.columns and source != canonical
    }
    renamed = frame.rename(columns=rename_map).copy()

    for field in CANONICAL_FIELDS:
        if field.name not in renamed.columns:
            renamed[field.name] = pd.NA

    return renamed


def suggest_metric_columns(columns: Iterable[str]) -> List[str]:
    columns = list(columns)
    normalized_to_original = {_normalize(column): column for column in columns}
    suggestions: List[str] = []

    for metric in WRDS_PRIORITY_METRICS:
        if metric in columns:
            suggestions.append(metric)

    for key, original in normalized_to_original.items():
        if original in suggestions:
            continue
        if any(hint in key for hint in GENERIC_METRIC_HINTS):
            suggestions.append(original)

    return suggestions


def describe_mapping(mapping: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for field in CANONICAL_FIELDS:
        rows.append(
            {
                "canonical_field": field.name,
                "source_column": mapping.get(field.name, ""),
                "required": field.required,
                "description": field.description,
            }
        )
    return pd.DataFrame(rows)
