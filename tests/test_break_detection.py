from types import SimpleNamespace

import pandas as pd

from src.break_detection import detect_structural_breaks_exact, exact_break_detection_status


def test_exact_break_detection_status_handles_missing_rscript(monkeypatch):
    exact_break_detection_status.cache_clear()

    def missing_run(*args, **kwargs):
        raise FileNotFoundError("Rscript")

    monkeypatch.setattr("src.break_detection.subprocess.run", missing_run)
    available, message = exact_break_detection_status()

    assert not available
    assert "Rscript" in message


def test_detect_structural_breaks_exact_parses_r_output(monkeypatch):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-03-31", periods=16, freq="QE-DEC"),
            "value": [1.0] * 16,
        }
    )

    monkeypatch.setattr("src.break_detection.exact_break_detection_status", lambda: (True, "ok"))

    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout='{"break_indices":"3,6","break_count":2}', stderr="", returncode=0)

    monkeypatch.setattr("src.break_detection.subprocess.run", fake_run)
    result = detect_structural_breaks_exact(frame)

    assert result.method == "r-strucchange"
    assert result.break_indices == [3, 6]
    assert result.segment_start == 6
    assert result.segment_end == 16
