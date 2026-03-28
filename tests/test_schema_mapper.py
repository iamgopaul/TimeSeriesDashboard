from src.schema_mapper import infer_schema_mapping, validate_mapping


def test_infer_schema_mapping_for_wrds_columns():
    inference = infer_schema_mapping(["gvkey", "datadate", "conm", "ROA", "atq"])

    assert inference.mapping["entity_id"] == "gvkey"
    assert inference.mapping["date"] == "datadate"
    assert inference.mapping["entity_name"] == "conm"
    assert "ROA" in inference.suggested_metrics


def test_validate_mapping_requires_entity_id_and_date():
    errors = validate_mapping({"entity_id": "", "date": ""}, ["firm", "period"])

    assert len(errors) == 2
