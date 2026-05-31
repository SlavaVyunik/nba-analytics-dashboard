import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.theme_utils import ensure_template

_TPL = ensure_template()   # register nba_dark before any chart is built

METRICS = ["PTS", "AST", "REB", "STL", "BLK", "FG_PCT", "FG3_PCT", "USG_PCT"]
METRIC_LABELS = {
    "PTS": "Очки/игра",
    "AST": "Передачи/игра",
    "REB": "Подборы/игра",
    "STL": "Перехваты/игра",
    "BLK": "Блок-шоты/игра",
    "FG_PCT": "% бросков",
    "FG3_PCT": "% трёхочковых",
    "FT_PCT": "% штрафных",
    "TOV": "Потери/игра",
    "USG_PCT": "% владения",
    "TS_PCT": "TS%",
    "PLUS_MINUS": "+/-",
    "efficiency": "Эффективность",
}


def umap_scatter(df):
    hover = {c: ":.1f" for c in ["PTS", "AST", "REB"] if c in df.columns}
    hover["umap_x"] = False
    hover["umap_y"] = False

    name_col = "name" if "name" in df.columns else ("PLAYER_NAME" if "PLAYER_NAME" in df.columns else None)

    fig = px.scatter(
        df,
        x="umap_x",
        y="umap_y",
        color="cluster_name",
        hover_name=name_col,
        hover_data=hover,
        title="Кластеризация игроков NBA 2025-26",
        labels={"cluster_name": "Тип игрока",
                "umap_x": "Компонента 1",
                "umap_y": "Компонента 2"},
        template="nba_dark",
        height=520,
    )
    fig.update_traces(marker=dict(
        size=9, opacity=0.82,
        line=dict(color="rgba(255,255,255,0.18)", width=0.8),
    ))
    return fig


def radar_chart(df, cluster_name):
    available = [m for m in METRICS if m in df.columns]
    labels = [METRIC_LABELS.get(m, m) for m in available]

    cluster_avg = df[df["cluster_name"] == cluster_name][available].mean()
    overall_avg = df[available].mean()
    max_vals = df[available].max().replace(0, 1)

    c_norm = (cluster_avg / max_vals).tolist() + [(cluster_avg / max_vals).tolist()[0]]
    o_norm = (overall_avg / max_vals).tolist() + [(overall_avg / max_vals).tolist()[0]]
    lbl = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=c_norm, theta=lbl, fill="toself", name=cluster_name,
        line_color="#e74c3c", fillcolor="rgba(231,76,60,0.2)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=o_norm, theta=lbl, fill="toself", name="Среднее по лиге",
        line_color="#3498db", fillcolor="rgba(52,152,219,0.1)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0c1220",
            radialaxis=dict(
                visible=True, range=[0, 1],
                showticklabels=False,
                gridcolor="rgba(255,255,255,0.10)",
                linecolor="rgba(255,255,255,0.08)",
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.10)",
                linecolor="rgba(255,255,255,0.12)",
                tickfont=dict(size=11, color="#8b9ab5"),
            ),
        ),
        title=f"Профиль: {cluster_name}",
        template="nba_dark",
        height=420,
    )
    return fig


def top_players_bar(df, cluster_name, metric="PTS", n=10):
    subset = df[df["cluster_name"] == cluster_name]
    if metric not in subset.columns:
        metric = "PTS" if "PTS" in subset.columns else subset.columns[0]
    subset = subset.nlargest(n, metric)
    name_col = "name" if "name" in subset.columns else "PLAYER_NAME"

    fig = px.bar(
        subset,
        x=metric,
        y=name_col,
        orientation="h",
        title=f"Топ-{n}: {cluster_name}",
        labels={metric: METRIC_LABELS.get(metric, metric), name_col: ""},
        template="nba_dark",
        height=380,
        color=metric,
        color_continuous_scale=[[0.0, "#0d1f3c"], [0.4, "#1D428A"], [0.75, "#C8102E"], [1.0, "#ff4d6d"]],
    )
    fig.update_coloraxes(
        colorbar=dict(
            thickness=10, len=0.7,
            tickfont=dict(color="#6b7a99", size=10),
            title=dict(font=dict(color="#6b7a99", size=11)),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.08)",
        )
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    fig.update_traces(
        texttemplate="%{x:.1f}",
        textposition="outside",
        textfont=dict(size=11, color="#8b9ab5"),
    )
    return fig


def pts_vs_ast(df):
    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    size_col = "REB" if "REB" in df.columns else None
    fig = px.scatter(
        df,
        x="PTS",
        y="AST",
        color="cluster_name",
        size=size_col,
        hover_name=name_col,
        title="Очки vs Передачи (за игру)",
        labels={"PTS": "Очки/игра", "AST": "Передачи/игра", "cluster_name": "Тип"},
        template="nba_dark",
        height=480,
    )
    fig.update_traces(marker=dict(
        opacity=0.78,
        line=dict(color="rgba(255,255,255,0.15)", width=0.7),
    ))
    return fig
