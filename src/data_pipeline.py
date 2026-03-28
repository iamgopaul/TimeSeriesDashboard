from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Iterable, List

import pandas as pd

from src.schema_mapper import apply_schema_mapping


CANONICAL_NUMERIC_FIELDS = ("assets", "equity", "net_income")
CANONICAL_RESERVED_FIELDS = {
    "entity_id",
    "entity_name",
    "date",
    "industry",
    "deletion_reason",
}


@dataclass
class DatasetProfile:
    row_count: int
    entity_count: int
    column_count: int
    date_min: pd.Timestamp | None
    date_max: pd.Timestamp | None
    numeric_columns: List[str]
    suggested_metrics: List[str]
    warnings: List[str]


@dataclass
class TargetProvenance:
    target_column: str
    source_type: str
    can_reconstruct_from_raw: bool
    detail: str


@dataclass
class WinsorizationReport:
    enabled: bool
    target_column: str
    lower_quantile: float | None
    upper_quantile: float | None
    changed_rows: int
    detail: str


def load_csv(upload_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(upload_bytes), low_memory=False)


def prepare_dataset(frame: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    prepared = apply_schema_mapping(frame, mapping)

    prepared["entity_id"] = prepared["entity_id"].astype("string").str.strip()
    prepared["entity_name"] = prepared["entity_name"].astype("string").fillna("")
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared["industry"] = prepared["industry"].astype("string").fillna("")
    prepared["deletion_reason"] = prepared["deletion_reason"].astype("string").fillna("")

    for column in CANONICAL_NUMERIC_FIELDS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    numeric_candidates = [
        column
        for column in prepared.columns
        if column not in CANONICAL_RESERVED_FIELDS and column not in CANONICAL_NUMERIC_FIELDS
    ]
    for column in numeric_candidates:
        coerced = pd.to_numeric(prepared[column], errors="coerce")
        if coerced.notna().sum() > 0:
            prepared[column] = coerced

    prepared = prepared.dropna(subset=["entity_id", "date"]).copy()
    prepared["entity_label"] = prepared["entity_id"]
    has_name = prepared["entity_name"].str.len() > 0
    prepared.loc[has_name, "entity_label"] = (
        prepared.loc[has_name, "entity_name"] + " (" + prepared.loc[has_name, "entity_id"] + ")"
    )

    prepared = prepared.sort_values(["entity_id", "date"]).reset_index(drop=True)
    return prepared


def detect_metric_columns(frame: pd.DataFrame) -> List[str]:
    metric_columns: List[str] = []
    for column in frame.columns:
        if column in CANONICAL_RESERVED_FIELDS:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            metric_columns.append(column)
    preferred = [column for column in ("ROA", "ROE") if column in metric_columns]
    remainder = [column for column in metric_columns if column not in preferred]
    return preferred + remainder


def build_dataset_profile(frame: pd.DataFrame, warnings: Iterable[str] | None = None) -> DatasetProfile:
    numeric_columns = detect_metric_columns(frame)
    date_min = frame["date"].min() if "date" in frame.columns and not frame.empty else None
    date_max = frame["date"].max() if "date" in frame.columns and not frame.empty else None
    suggested_metrics = [column for column in ("ROA", "ROE") if column in frame.columns]
    if not suggested_metrics:
        suggested_metrics = numeric_columns[:5]
    return DatasetProfile(
        row_count=int(len(frame)),
        entity_count=int(frame["entity_id"].nunique()) if "entity_id" in frame.columns else 0,
        column_count=int(len(frame.columns)),
        date_min=date_min,
        date_max=date_max,
        numeric_columns=numeric_columns,
        suggested_metrics=suggested_metrics,
        warnings=list(warnings or []),
    )


def build_entity_options(frame: pd.DataFrame) -> List[str]:
    labels = frame[["entity_id", "entity_label"]].drop_duplicates().sort_values("entity_label")
    return labels["entity_label"].tolist()


def resolve_entity_id(frame: pd.DataFrame, entity_label: str) -> str:
    match = frame.loc[frame["entity_label"] == entity_label, "entity_id"]
    if match.empty:
        raise KeyError(f"Unknown entity label: {entity_label}")
    return str(match.iloc[0])


def build_target_series(frame: pd.DataFrame, entity_id: str, target_column: str) -> pd.DataFrame:
    subset = frame.loc[frame["entity_id"] == entity_id, ["date", target_column]].copy()
    subset = subset.rename(columns={target_column: "value"}).dropna(subset=["value"])
    subset = subset.sort_values("date").reset_index(drop=True)
    return subset


def describe_target_provenance(frame: pd.DataFrame, target_column: str) -> TargetProvenance:
    has_direct_metric = target_column in frame.columns and frame[target_column].notna().any()
    raw_requirements = {
        "ROA": ["net_income", "assets"],
        "ROE": ["net_income", "equity"],
    }
    reconstructable = False
    if target_column in raw_requirements:
        reconstructable = all(
            column in frame.columns and frame[column].notna().any()
            for column in raw_requirements[target_column]
        )

    if has_direct_metric and reconstructable:
        detail = f"`{target_column}` is present directly and can also be cross-checked from canonical raw fields."
        source_type = "direct_and_reconstructable"
    elif has_direct_metric:
        detail = f"`{target_column}` is being used as a directly uploaded metric."
        source_type = "direct_metric"
    elif reconstructable:
        detail = f"`{target_column}` is not present directly but appears reconstructable from canonical raw fields."
        source_type = "reconstructable_only"
    else:
        detail = f"`{target_column}` is being used as-is and cannot currently be reconstructed from canonical raw fields."
        source_type = "unverified_metric"

    return TargetProvenance(
        target_column=target_column,
        source_type=source_type,
        can_reconstruct_from_raw=bool(reconstructable),
        detail=detail,
    )


def apply_target_winsorization(
    frame: pd.DataFrame,
    target_column: str,
    enabled: bool = False,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> tuple[pd.DataFrame, WinsorizationReport]:
    if not enabled:
        return frame.copy(), WinsorizationReport(
            enabled=False,
            target_column=target_column,
            lower_quantile=None,
            upper_quantile=None,
            changed_rows=0,
            detail="Winsorization disabled.",
        )

    working = frame.copy()
    original = working[target_column].copy()
    valid = working[target_column].notna()
    if not valid.any():
        return working, WinsorizationReport(
            enabled=True,
            target_column=target_column,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            changed_rows=0,
            detail="Winsorization enabled, but no non-null target values were available.",
        )

    grouped = working.loc[valid].groupby("date")[target_column]
    lower_bounds = grouped.transform(lambda s: s.quantile(lower_quantile))
    upper_bounds = grouped.transform(lambda s: s.quantile(upper_quantile))
    clipped = working.loc[valid, target_column].clip(lower=lower_bounds, upper=upper_bounds)
    working.loc[valid, target_column] = clipped
    changed_rows = int((working[target_column].fillna(0.0) != original.fillna(0.0)).sum())
    detail = (
        f"Cross-sectional winsorization by date on `{target_column}` at "
        f"{lower_quantile:.0%}/{upper_quantile:.0%} changed {changed_rows} row(s)."
    )
    return working, WinsorizationReport(
        enabled=True,
        target_column=target_column,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        changed_rows=changed_rows,
        detail=detail,
    )


def compute_continuity_summary(series_frame: pd.DataFrame) -> pd.DataFrame:
    if series_frame.empty:
        return pd.DataFrame(
            [{"observations": 0, "missing_quarters": 0, "continuous": False}]
        )

    working = series_frame.sort_values("date").copy()
    working["period"] = working["date"].dt.to_period("Q")
    gaps = working["period"].diff().dropna()
    missing_quarters = int(sum(max(int(gap.n) - 1, 0) for gap in gaps))
    return pd.DataFrame(
        [
            {
                "observations": int(len(working)),
                "missing_quarters": missing_quarters,
                "continuous": missing_quarters == 0,
                "start_date": working["date"].min(),
                "end_date": working["date"].max(),
            }
        ]
    )


def summarize_entities(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    rows = []
    for entity_id, subset in frame.groupby("entity_id", sort=True):
        target = subset[["date", target_column]].dropna()
        if target.empty:
            continue
        continuity = compute_continuity_summary(target.rename(columns={target_column: "value"})).iloc[0]
        rows.append(
            {
                "entity_id": entity_id,
                "entity_label": subset["entity_label"].iloc[0],
                "observations": int(len(target)),
                "missing_quarters": int(continuity["missing_quarters"]),
                "start_date": target["date"].min(),
                "end_date": target["date"].max(),
            }
        )
    return pd.DataFrame(rows).sort_values(["observations", "entity_label"], ascending=[False, True])


def build_entity_eligibility_summary(frame: pd.DataFrame, target_column: str, min_history: int = 10) -> pd.DataFrame:
    rows = []
    for entity_id, subset in frame.groupby("entity_id", sort=True):
        target = subset[["date", target_column]].dropna()
        continuity = compute_continuity_summary(target.rename(columns={target_column: "value"})).iloc[0]
        reasons: list[str] = []
        if target.empty:
            reasons.append("missing_target")
        if int(len(target)) < int(min_history):
            reasons.append("short_history")
        if int(continuity["missing_quarters"]) > 0:
            reasons.append("gaps")
        deletion_codes = subset["deletion_reason"].dropna().astype(str)
        deletion_reason = ""
        if not deletion_codes.empty and deletion_codes.str.len().max() > 0:
            deletion_reason = deletion_codes.iloc[-1]
        rows.append(
            {
                "entity_id": entity_id,
                "entity_label": subset["entity_label"].iloc[0],
                "target_observations": int(len(target)),
                "missing_quarters": int(continuity["missing_quarters"]),
                "continuous": bool(continuity["continuous"]),
                "deletion_reason": deletion_reason,
                "eligible": len(reasons) == 0,
                "exclusion_reason": "eligible" if not reasons else ",".join(reasons),
            }
        )
    return pd.DataFrame(rows).sort_values(["eligible", "target_observations", "entity_label"], ascending=[False, False, True])
