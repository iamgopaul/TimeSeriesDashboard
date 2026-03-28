import pandas as pd

from src.data_pipeline import (
    apply_target_winsorization,
    build_dataset_profile,
    build_entity_eligibility_summary,
    build_target_series,
    compute_continuity_summary,
    describe_target_provenance,
    prepare_dataset,
)


def test_prepare_dataset_builds_labels_and_profile():
    frame = pd.DataFrame(
        {
            "gvkey": ["1001", "1001"],
            "datadate": ["2020-03-31", "2020-06-30"],
            "conm": ["Example Corp", "Example Corp"],
            "ROA": [0.1, 0.12],
        }
    )
    mapping = {"entity_id": "gvkey", "date": "datadate", "entity_name": "conm"}

    prepared = prepare_dataset(frame, mapping)
    profile = build_dataset_profile(prepared)

    assert profile.entity_count == 1
    assert "ROA" in profile.numeric_columns
    assert prepared["entity_label"].iloc[0] == "Example Corp (1001)"


def test_compute_continuity_summary_detects_quarter_gaps():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31", "2020-09-30", "2020-12-31"]),
            "value": [1.0, 2.0, 3.0],
        }
    )

    summary = compute_continuity_summary(frame)

    assert int(summary["missing_quarters"].iloc[0]) == 1
    assert not bool(summary["continuous"].iloc[0])


def test_build_target_series_filters_single_entity():
    frame = pd.DataFrame(
        {
            "entity_id": ["1", "1", "2"],
            "date": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-03-31"]),
            "ROA": [0.1, 0.2, 0.3],
        }
    )

    series = build_target_series(frame, "1", "ROA")

    assert len(series) == 2
    assert list(series["value"]) == [0.1, 0.2]


def test_describe_target_provenance_detects_reconstructable_roa():
    frame = pd.DataFrame(
        {
            "ROA": [0.1, 0.2],
            "net_income": [10.0, 11.0],
            "assets": [100.0, 120.0],
        }
    )

    provenance = describe_target_provenance(frame, "ROA")

    assert provenance.source_type == "direct_and_reconstructable"
    assert provenance.can_reconstruct_from_raw


def test_apply_target_winsorization_clips_cross_sectional_outlier():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31"] * 4),
            "ROA": [0.1, 0.11, 0.12, 9.0],
        }
    )

    winsorized, report = apply_target_winsorization(frame, "ROA", enabled=True, lower_quantile=0.0, upper_quantile=0.75)

    assert report.changed_rows == 1
    assert float(winsorized["ROA"].max()) < 9.0


def test_build_entity_eligibility_summary_marks_short_and_gappy_series():
    frame = pd.DataFrame(
        {
            "entity_id": ["1", "1", "2", "2"],
            "entity_label": ["A (1)", "A (1)", "B (2)", "B (2)"],
            "date": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-03-31", "2020-12-31"]),
            "ROA": [0.1, 0.2, 0.3, 0.4],
            "deletion_reason": ["", "", "01", "01"],
        }
    )

    summary = build_entity_eligibility_summary(frame, "ROA", min_history=3)

    assert not summary["eligible"].any()
    assert summary["exclusion_reason"].str.contains("short_history").all()
