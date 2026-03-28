from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
import subprocess
import tempfile
from typing import List

import pandas as pd

try:  # pragma: no cover - dependency availability varies
    import ruptures as rpt
except Exception:  # pragma: no cover - dependency availability varies
    rpt = None


@dataclass
class BreakDetectionResult:
    break_indices: List[int]
    break_dates: List[str]
    segment_start: int
    segment_end: int
    segment: pd.DataFrame
    method: str
    message: str = ""


def _empty_result(working: pd.DataFrame, method: str, message: str = "") -> BreakDetectionResult:
    return BreakDetectionResult(
        break_indices=[],
        break_dates=[],
        segment_start=0,
        segment_end=len(working),
        segment=working,
        method=method,
        message=message,
    )


def _select_longest_segment(working: pd.DataFrame, break_indices: list[int], method: str, message: str = "") -> BreakDetectionResult:
    if not break_indices:
        return _empty_result(working, method=method, message=message)

    boundaries = [0] + break_indices + [len(working)]
    segments = [
        (boundaries[idx], boundaries[idx + 1])
        for idx in range(len(boundaries) - 1)
    ]
    start, end = max(segments, key=lambda pair: pair[1] - pair[0])
    segment = working.iloc[start:end].reset_index(drop=True)
    break_dates = [
        working.iloc[idx - 1]["date"].date().isoformat()
        for idx in break_indices
        if 0 < idx <= len(working)
    ]
    return BreakDetectionResult(
        break_indices=break_indices,
        break_dates=break_dates,
        segment_start=start,
        segment_end=end,
        segment=segment,
        method=method,
        message=message,
    )


@lru_cache(maxsize=1)
def exact_break_detection_status() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["Rscript", "-e", "suppressPackageStartupMessages(library(strucchange)); cat('ok\\n')"],
            check=True,
            capture_output=True,
            text=True,
        )
        return True, result.stdout.strip() or "R strucchange available."
    except FileNotFoundError:
        return False, "Rscript is not installed."
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        return False, message


def detect_structural_breaks_practical(series_frame: pd.DataFrame, penalty: float = 3.0) -> BreakDetectionResult:
    working = series_frame.sort_values("date").reset_index(drop=True)
    if len(working) < 12 or rpt is None:
        return _empty_result(working, method="none", message="Practical break detection unavailable for this series.")

    values = working["value"].to_numpy(dtype=float).reshape(-1, 1)
    algo = rpt.Pelt(model="l2").fit(values)
    break_indices = [idx for idx in algo.predict(pen=penalty) if idx < len(working)]
    return _select_longest_segment(working, break_indices, method="ruptures-pelt")


def detect_structural_breaks_exact(series_frame: pd.DataFrame, min_segment_fraction: float = 0.15) -> BreakDetectionResult:
    working = series_frame.sort_values("date").reset_index(drop=True)
    if len(working) < 12:
        return _empty_result(working, method="r-strucchange", message="Series too short for Bai-Perron-style break detection.")

    available, message = exact_break_detection_status()
    if not available:
        raise RuntimeError(f"Exact Bai-Perron-style break detection unavailable: {message}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as input_file:
        working[["date", "value"]].to_csv(input_file.name, index=False)
        input_path = input_file.name

    script = r"""
args <- commandArgs(trailingOnly = TRUE)
input_path <- args[1]
h_fraction <- as.numeric(args[2])
suppressPackageStartupMessages(library(strucchange))
df <- read.csv(input_path, stringsAsFactors = FALSE)
y <- as.numeric(df$value)
h_value <- max(2L, ceiling(length(y) * h_fraction))
bp <- breakpoints(y ~ 1, h = h_value)
bic_values <- BIC(bp)
break_count <- 0L
if (length(bic_values) > 0) {
  best_index <- which.min(as.numeric(bic_values))
  break_count <- max(best_index - 1L, 0L)
  bic_names <- names(bic_values)
  if (!is.null(bic_names) && length(bic_names) >= best_index && nzchar(bic_names[best_index])) {
    named_count <- suppressWarnings(as.integer(bic_names[best_index]))
    if (!is.na(named_count)) {
      break_count <- named_count
    }
  }
}
if (break_count > 0L) {
  bp_opt <- breakpoints(y ~ 1, breaks = break_count, h = h_value)
  breakpoints_out <- as.integer(bp_opt$breakpoints)
  breakpoints_out <- breakpoints_out[!is.na(breakpoints_out)]
} else {
  breakpoints_out <- integer(0)
}
break_string <- if (length(breakpoints_out) > 0L) paste(breakpoints_out, collapse = ",") else ""
cat(paste0("{\"break_indices\":\"", break_string, "\",\"break_count\":", length(breakpoints_out), "}"))
"""
    try:
        result = subprocess.run(
            ["Rscript", "-e", script, input_path, str(float(min_segment_fraction))],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"R strucchange failed: {message}") from exc
    finally:
        try:
            os.unlink(input_path)
        except OSError:
            pass

    try:
        payload = json.loads(result.stdout.strip() or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Unable to parse R strucchange output: {result.stdout}") from exc

    raw_indices = str(payload.get("break_indices", "")).strip()
    break_indices = [int(idx) for idx in raw_indices.split(",") if idx]
    return _select_longest_segment(
        working,
        break_indices,
        method="r-strucchange",
        message="Exact Bai-Perron-style break detection via R strucchange.",
    )


def detect_structural_breaks(series_frame: pd.DataFrame, method: str = "practical") -> BreakDetectionResult:
    if method == "strict":
        return detect_structural_breaks_exact(series_frame)
    if method == "practical":
        return detect_structural_breaks_practical(series_frame)
    raise ValueError("break detection method must be `strict` or `practical`.")
