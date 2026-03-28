from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


INTERVAL_COLUMN_MAP = {
    "50%": ("q25", "q75"),
    "80%": ("q10", "q90"),
    "95%": ("q025", "q975"),
}


def build_forecast_figure(
    actual_series: pd.DataFrame,
    forecast_frame: pd.DataFrame,
    title: str,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=actual_series["date"],
            y=actual_series["value"],
            mode="lines+markers",
            name="Actual",
        )
    )
    if not forecast_frame.empty:
        figure.add_trace(
            go.Scatter(
                x=forecast_frame["date"],
                y=forecast_frame["forecast"],
                mode="lines+markers",
                name=forecast_frame["model"].iloc[0],
            )
        )
        if {"q10", "q90"}.issubset(forecast_frame.columns):
            figure.add_trace(
                go.Scatter(
                    x=forecast_frame["date"],
                    y=forecast_frame["q90"],
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=forecast_frame["date"],
                    y=forecast_frame["q10"],
                    mode="lines",
                    fill="tonexty",
                    name="80% interval",
                    line={"width": 0},
                )
            )

    figure.update_layout(title=title, xaxis_title="Date", yaxis_title="Value")
    return figure


def build_interval_forecast_figure(
    actual_series: pd.DataFrame,
    forecast_frame: pd.DataFrame,
    title: str,
    interval_label: str,
) -> go.Figure:
    lower_column, upper_column = INTERVAL_COLUMN_MAP[interval_label]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=actual_series["date"],
            y=actual_series["value"],
            mode="lines+markers",
            name="Actual",
            line={"width": 3},
        )
    )
    if not forecast_frame.empty:
        model_name = str(forecast_frame["model"].iloc[0])
        figure.add_trace(
            go.Scatter(
                x=forecast_frame["date"],
                y=forecast_frame[upper_column],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=forecast_frame["date"],
                y=forecast_frame[lower_column],
                mode="lines",
                fill="tonexty",
                name=f"{interval_label} interval",
                line={"width": 0},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=forecast_frame["date"],
                y=forecast_frame["forecast"],
                mode="lines+markers",
                name=f"{model_name} median forecast",
            )
        )

    figure.update_layout(title=title, xaxis_title="Date", yaxis_title="Value")
    return figure


def build_combined_forecast_figure(
    actual_series: pd.DataFrame,
    forecast_frames: list[pd.DataFrame],
    title: str,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=actual_series["date"],
            y=actual_series["value"],
            mode="lines+markers",
            name="Actual",
            line={"width": 3},
        )
    )
    for forecast_frame in forecast_frames:
        if forecast_frame is None or forecast_frame.empty:
            continue
        model_name = str(forecast_frame["model"].iloc[0])
        figure.add_trace(
            go.Scatter(
                x=forecast_frame["date"],
                y=forecast_frame["forecast"],
                mode="lines+markers",
                name=model_name,
            )
        )
    figure.update_layout(title=title, xaxis_title="Date", yaxis_title="Value")
    return figure


def build_cumulative_error_figure(forecast_frames: list[pd.DataFrame], title: str = "Cumulative Absolute Error") -> go.Figure:
    figure = go.Figure()
    for forecast_frame in forecast_frames:
        if forecast_frame is None or forecast_frame.empty or "error" not in forecast_frame.columns:
            continue
        working = forecast_frame.sort_values("date").copy()
        working["cumulative_abs_error"] = working["error"].abs().cumsum()
        figure.add_trace(
            go.Scatter(
                x=working["date"],
                y=working["cumulative_abs_error"],
                mode="lines+markers",
                name=str(working["model"].iloc[0]),
            )
        )
    figure.update_layout(title=title, xaxis_title="Date", yaxis_title="Cumulative absolute error")
    return figure


def build_panel_aggregate_forecast_figure(forecast_panel: pd.DataFrame, title: str = "Panel Actual Vs Predicted") -> go.Figure:
    figure = go.Figure()
    if forecast_panel.empty:
        return figure
    working = forecast_panel.dropna(subset=["date", "actual", "forecast"]).copy()
    if working.empty:
        return figure
    actual_series = (
        working.groupby("date", as_index=False)["actual"]
        .mean()
        .sort_values("date")
    )
    figure.add_trace(
        go.Scatter(
            x=actual_series["date"],
            y=actual_series["actual"],
            mode="lines+markers",
            name="Panel actual mean",
            line={"width": 3},
        )
    )
    forecast_summary = (
        working.groupby(["date", "model"], as_index=False)["forecast"]
        .mean()
        .sort_values(["model", "date"])
    )
    for model_name, subset in forecast_summary.groupby("model", sort=False):
        figure.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["forecast"],
                mode="lines+markers",
                name=f"{model_name} mean forecast",
            )
        )
    figure.update_layout(title=title, xaxis_title="Holdout date", yaxis_title="Mean value across firms")
    return figure


def build_multifirm_error_distribution_figure(
    forecast_panel: pd.DataFrame,
    title: str = "Holdout Absolute Error Distribution",
) -> go.Figure:
    figure = go.Figure()
    if forecast_panel.empty:
        return figure
    working = forecast_panel.dropna(subset=["model", "absolute_error"]).copy()
    if working.empty:
        return figure
    for model_name, subset in working.groupby("model", sort=False):
        figure.add_trace(
            go.Box(
                y=subset["absolute_error"],
                name=str(model_name),
                boxmean=True,
            )
        )
    figure.update_layout(title=title, yaxis_title="Absolute error", xaxis_title="Model")
    return figure


def build_forecast_calibration_figure(
    forecast_panel: pd.DataFrame,
    title: str = "Actual Vs Predicted Across Holdout Points",
) -> go.Figure:
    figure = go.Figure()
    if forecast_panel.empty:
        return figure
    working = forecast_panel.dropna(subset=["model", "actual", "forecast"]).copy()
    if working.empty:
        return figure
    diagonal_min = float(min(working["actual"].min(), working["forecast"].min()))
    diagonal_max = float(max(working["actual"].max(), working["forecast"].max()))
    figure.add_trace(
        go.Scatter(
            x=[diagonal_min, diagonal_max],
            y=[diagonal_min, diagonal_max],
            mode="lines",
            name="Perfect fit",
            line={"dash": "dash"},
        )
    )
    for model_name, subset in working.groupby("model", sort=False):
        figure.add_trace(
            go.Scatter(
                x=subset["actual"],
                y=subset["forecast"],
                mode="markers",
                name=str(model_name),
                text=subset["entity_label"] if "entity_label" in subset.columns else None,
                hovertemplate=(
                    "Firm: %{text}<br>Actual: %{x:.3f}<br>Forecast: %{y:.3f}<extra></extra>"
                    if "entity_label" in subset.columns
                    else "Actual: %{x:.3f}<br>Forecast: %{y:.3f}<extra></extra>"
                ),
                marker={"size": 8, "opacity": 0.7},
            )
        )
    figure.update_layout(title=title, xaxis_title="Actual", yaxis_title="Forecast")
    return figure


def build_firm_error_leaderboard_figure(
    summary: pd.DataFrame,
    model_column: str,
    title: str,
    top_n: int = 10,
    ascending: bool = True,
) -> go.Figure:
    figure = go.Figure()
    if summary.empty or model_column not in summary.columns:
        return figure
    working = summary.dropna(subset=[model_column]).sort_values(model_column, ascending=ascending).head(top_n)
    if working.empty:
        return figure
    figure.add_trace(
        go.Bar(
            x=working[model_column],
            y=working["entity_label"],
            orientation="h",
            name=model_column,
        )
    )
    figure.update_layout(title=title, xaxis_title="Mean absolute error", yaxis_title="Firm")
    return figure


def build_residual_figure(residuals: pd.Series) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=np.arange(len(residuals)),
            y=residuals,
            mode="lines+markers",
            name="Residual",
        )
    )
    figure.update_layout(title="Residual Trace", xaxis_title="Index", yaxis_title="Residual")
    return figure


def build_residual_autocorrelation_figure(residuals: pd.Series, max_lag: int = 8) -> go.Figure:
    series = pd.Series(residuals, dtype=float).dropna()
    lags = list(range(1, min(max_lag, len(series) - 1) + 1))
    values = [float(series.autocorr(lag=lag)) for lag in lags]
    figure = go.Figure(
        data=[
            go.Bar(x=lags, y=values, name="ACF"),
        ]
    )
    figure.update_layout(title="Residual Autocorrelation", xaxis_title="Lag", yaxis_title="Correlation")
    return figure


def build_nli_distribution_figure(distribution: pd.DataFrame, selected_entity: str | None = None) -> go.Figure:
    working = distribution.copy()
    if working.empty:
        return go.Figure()

    working = working.sort_values("nli_score").reset_index(drop=True)
    working["rank"] = np.arange(1, len(working) + 1)
    median_score = float(working["nli_score"].median())
    q1_score = float(working["nli_score"].quantile(0.25))
    q3_score = float(working["nli_score"].quantile(0.75))

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.55, 0.45],
    )
    figure.add_trace(
        go.Histogram(
            x=working["nli_score"],
            nbinsx=min(30, max(10, len(working) // 3)),
            name="NLI distribution",
            opacity=0.8,
            marker={"color": "#4C78A8"},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=working["nli_score"],
            y=working["rank"],
            mode="markers",
            name="Firm ranks",
            marker={"size": 8, "color": "#9ECAE9"},
            text=working["entity_label"],
            hovertemplate="Firm: %{text}<br>NLI: %{x:.3f}<br>Rank: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    for value, label in ((q1_score, "Q1"), (median_score, "Median"), (q3_score, "Q3")):
        figure.add_vline(x=value, line_dash="dash", annotation_text=label)

    if selected_entity:
        selected_row = working.loc[working["entity_id"] == selected_entity]
        if not selected_row.empty:
            selected_score = float(selected_row["nli_score"].iloc[0])
            selected_rank = int(selected_row["rank"].iloc[0])
            selected_label = selected_row["entity_label"].iloc[0]
            figure.add_trace(
                go.Scatter(
                    x=[selected_score],
                    y=[selected_rank],
                    mode="markers",
                    name="Selected firm",
                    marker={"size": 12, "color": "#E45756", "symbol": "diamond"},
                    text=[selected_label],
                    hovertemplate="Selected: %{text}<br>NLI: %{x:.3f}<br>Rank: %{y}<extra></extra>",
                ),
                row=2,
                col=1,
            )
            figure.add_vline(x=selected_score, line_color="#E45756", annotation_text="Selected firm")

    figure.update_layout(
        title="NLI Distribution And Rank",
        bargap=0.05,
        legend={"orientation": "h"},
    )
    figure.update_xaxes(title_text="NLI score", row=2, col=1)
    figure.update_yaxes(title_text="Count", row=1, col=1)
    figure.update_yaxes(title_text="Rank (low to high)", row=2, col=1)
    return figure


def build_error_gap_figure(comparison: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if comparison.empty:
        return figure
    figure.add_trace(
        go.Bar(
            x=comparison["date"],
            y=comparison["error_gap"],
            name="ARIMA error - Chronos error",
        )
    )
    figure.update_layout(title="Model Error Gap", xaxis_title="Date", yaxis_title="Error gap")
    return figure


def build_eligibility_figure(summary: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if summary.empty:
        return figure
    counts = summary["exclusion_reason"].value_counts().reset_index()
    counts.columns = ["reason", "count"]
    figure.add_trace(
        go.Bar(
            x=counts["reason"],
            y=counts["count"],
            name="Entity count",
        )
    )
    figure.update_layout(title="Eligibility And Exclusion Reasons", xaxis_title="Reason", yaxis_title="Firms")
    return figure


def build_nli_quartile_figure(distribution: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if distribution.empty:
        return figure
    working = distribution.copy()
    working["quartile"] = pd.qcut(working["nli_score"], 4, labels=["Low", "Mid-Low", "Mid-High", "High"], duplicates="drop")
    counts = working["quartile"].value_counts().sort_index().reset_index()
    counts.columns = ["quartile", "count"]
    figure.add_trace(go.Bar(x=counts["quartile"], y=counts["count"], name="Firms"))
    figure.update_layout(title="NLI Quartile Distribution", xaxis_title="Quartile", yaxis_title="Firms")
    return figure


def build_tournament_gap_scatter(metrics: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if metrics.empty or "delta_mase" not in metrics.columns:
        return figure
    working = metrics.dropna(subset=["nli_score", "delta_mase"])
    if working.empty:
        return figure
    figure.add_trace(
        go.Scatter(
            x=working["nli_score"],
            y=working["delta_mase"],
            mode="markers",
            text=working["entity_label"],
            marker={"size": 9},
            hovertemplate="Firm: %{text}<br>NLI: %{x:.3f}<br>Delta MASE: %{y:.3f}<extra></extra>",
            name="Firms",
        )
    )
    figure.update_layout(title="NLI Vs Chronos Advantage", xaxis_title="NLI score", yaxis_title="ARIMA MASE - Chronos MASE")
    return figure


def build_tournament_leaderboard_figure(metrics: pd.DataFrame, top_n: int = 10) -> go.Figure:
    figure = go.Figure()
    if metrics.empty or "delta_mase" not in metrics.columns:
        return figure
    working = metrics.dropna(subset=["delta_mase"]).head(top_n)
    if working.empty:
        return figure
    figure.add_trace(
        go.Bar(
            x=working["delta_mase"],
            y=working["entity_label"],
            orientation="h",
            name="Delta MASE",
        )
    )
    figure.update_layout(title="Tournament Leaderboard", xaxis_title="ARIMA MASE - Chronos MASE", yaxis_title="Firm")
    return figure


def build_tournament_quartile_win_figure(metrics: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if metrics.empty:
        return figure
    working = metrics.dropna(subset=["nli_score"]).copy()
    if working.empty:
        return figure
    working["quartile"] = pd.qcut(working["nli_score"], 4, labels=["Low", "Mid-Low", "Mid-High", "High"], duplicates="drop")
    summary = (
        working.groupby("quartile", observed=False)["chronos_beats_arima"]
        .mean()
        .reset_index()
        .rename(columns={"chronos_beats_arima": "chronos_win_rate"})
    )
    figure.add_trace(go.Bar(x=summary["quartile"], y=summary["chronos_win_rate"], name="Chronos win rate"))
    figure.update_layout(title="Chronos Win Rate By NLI Quartile", xaxis_title="NLI quartile", yaxis_title="Win rate")
    return figure


def build_volatility_gap_figure(metrics: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if metrics.empty:
        return figure
    working = metrics.dropna(subset=["volatility", "delta_mase"])
    if working.empty:
        return figure
    figure.add_trace(
        go.Scatter(
            x=working["volatility"],
            y=working["delta_mase"],
            mode="markers",
            text=working["entity_label"],
            hovertemplate="Firm: %{text}<br>Volatility: %{x:.3f}<br>Delta MASE: %{y:.3f}<extra></extra>",
            name="Firms",
        )
    )
    figure.update_layout(title="Volatility Vs Chronos Advantage", xaxis_title="Volatility", yaxis_title="ARIMA MASE - Chronos MASE")
    return figure


def build_uncertainty_width_nli_figure(metrics: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if metrics.empty:
        return figure
    working = metrics.dropna(subset=["nli_score", "chronos_interval_width"])
    if working.empty:
        return figure
    figure.add_trace(
        go.Scatter(
            x=working["nli_score"],
            y=working["chronos_interval_width"],
            mode="markers",
            text=working["entity_label"],
            hovertemplate="Firm: %{text}<br>NLI: %{x:.3f}<br>Median interval width: %{y:.3f}<extra></extra>",
            name="Firms",
        )
    )
    figure.update_layout(
        title="Chronos Uncertainty Width Vs NLI",
        xaxis_title="NLI score",
        yaxis_title="Median interval width",
    )
    return figure


def build_coverage_quartile_figure(metrics: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if metrics.empty:
        return figure
    working = metrics.dropna(subset=["nli_score", "chronos_coverage_rate"]).copy()
    if working.empty:
        return figure
    working["quartile"] = pd.qcut(
        working["nli_score"],
        4,
        labels=["Low", "Mid-Low", "Mid-High", "High"],
        duplicates="drop",
    )
    summary = (
        working.groupby("quartile", observed=False)["chronos_coverage_rate"]
        .mean()
        .reset_index()
    )
    figure.add_trace(
        go.Bar(
            x=summary["quartile"],
            y=summary["chronos_coverage_rate"],
            name="Coverage rate",
        )
    )
    figure.update_layout(
        title="Chronos Coverage Rate By NLI Quartile",
        xaxis_title="NLI quartile",
        yaxis_title="Coverage rate",
    )
    return figure


def build_winner_distribution_figure(metrics: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    if metrics.empty or "winner" not in metrics.columns:
        return figure
    counts = metrics["winner"].value_counts().reset_index()
    counts.columns = ["winner", "count"]
    figure.add_trace(go.Bar(x=counts["winner"], y=counts["count"], name="Firms"))
    figure.update_layout(title="Tournament Winners", xaxis_title="Winning model", yaxis_title="Firms")
    return figure
