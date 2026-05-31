"""
Markowitz Mean-Variance Portfolio Optimization for NBA Roster Construction.

Applies Markowitz MVO concepts to team building:
  Maximize sum(trade_value) subject to:
      portfolio_risk = sqrt(mean(risk²)) <= risk_budget

The greedy optimizer builds the roster iteratively by ranking players on
their risk-adjusted trade value ratio and adding them as long as the
portfolio risk constraint is satisfied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.theme_utils import ensure_template

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TPL = ensure_template()   # register nba_dark if not yet registered
N_PLAYERS: int = 5

NBA_RED = "#C8102E"
NBA_BLUE = "#1D428A"
BG_CHART = "#0c1220"
TEXT_MUTED = "#8b9ab5"
GRID = "#141E30"
GOLD = "#FFC72C"

NBA_PALETTE = [
    "#C8102E", "#1D428A", "#FFC72C", "#27AE60",
    "#9B59B6", "#E67E22", "#1ABC9C", "#E74C3C",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_player_name(row: pd.Series) -> str:
    for col in ("PLAYER_NAME", "player_name", "name", "Name"):
        val = row.get(col)
        if val is not None and str(val).strip():
            return str(val)
    return "Unknown"


def _get_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that *df* has the required columns and return a copy with
    float columns ``_tv`` (trade value) and ``_risk`` (injury risk).

    Raises
    ------
    ValueError
        If either ``trade_value`` or ``injury_risk`` is missing.
    """
    missing = [c for c in ("trade_value", "injury_risk") if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}. "
            "Ensure 'trade_value' and 'injury_risk' are present."
        )
    out = df.copy()
    out["_tv"] = out["trade_value"].astype(float)
    out["_risk"] = out["injury_risk"].astype(float)
    return out


# ---------------------------------------------------------------------------
# Optimization functions
# ---------------------------------------------------------------------------

def optimize_team(
    df: pd.DataFrame,
    n_players: int = N_PLAYERS,
    risk_budget: float = 50.0,
    min_tv: float = 20.0,
) -> pd.DataFrame:
    """
    Greedy Markowitz-inspired team optimizer.

    Ranks players by ``_tv / (_risk + 1)`` (risk-adjusted value ratio),
    then greedily adds players while the portfolio risk constraint holds.

    Parameters
    ----------
    df : pd.DataFrame
        Player DataFrame with ``trade_value`` and ``injury_risk`` columns.
    n_players : int
        Target roster size.
    risk_budget : float
        Maximum allowed portfolio risk (sqrt of mean squared risks).
    min_tv : float
        Minimum trade value threshold; players below this are ignored.

    Returns
    -------
    pd.DataFrame
        Optimal team with original columns plus ``portfolio_risk`` and
        ``total_tv`` appended to every row for convenience.
    """
    data = _get_metrics(df)
    # Filter by minimum TV
    data = data[data["_tv"] >= min_tv].copy()

    # Compute risk-adjusted ratio and sort
    data["_ratio"] = data["_tv"] / (data["_risk"] + 1.0)
    data = data.sort_values("_ratio", ascending=False).reset_index(drop=True)

    selected_indices: list[int] = []
    for idx in range(len(data)):
        candidate_risks = [data.loc[i, "_risk"] for i in selected_indices] + [
            data.loc[idx, "_risk"]
        ]
        port_risk = float(np.sqrt(np.mean(np.array(candidate_risks) ** 2)))
        if port_risk <= risk_budget:
            selected_indices.append(idx)
        if len(selected_indices) >= n_players:
            break

    if not selected_indices:
        # Fallback: take the single lowest-risk player
        selected_indices = [int(data["_risk"].idxmin())]

    team = data.loc[selected_indices].drop(columns=["_ratio", "_tv", "_risk"]).copy()
    risks = data.loc[selected_indices, "_risk"].values
    tvs = data.loc[selected_indices, "trade_value"].values
    team["portfolio_risk"] = float(np.sqrt(np.mean(risks ** 2)))
    team["total_tv"] = float(np.sum(tvs))
    return team.reset_index(drop=True)


def compute_efficient_frontier(
    df: pd.DataFrame,
    n_points: int = 25,
    n_players: int = N_PLAYERS,
) -> pd.DataFrame:
    """
    Compute the efficient frontier by solving the portfolio problem at
    *n_points* equally-spaced risk budget levels between 10 and 90.

    Returns
    -------
    pd.DataFrame
        Columns: risk_budget, total_tv, avg_risk, n_players_selected.
    """
    data = _get_metrics(df)
    risk_levels = np.linspace(10.0, 90.0, n_points)
    records: list[dict] = []
    for rb in risk_levels:
        try:
            team = optimize_team(data, n_players=n_players, risk_budget=rb)
            records.append({
                "risk_budget": float(rb),
                "total_tv": float(team["total_tv"].iloc[0]),
                "avg_risk": float(data.loc[
                    data.index.isin(team.index), "_risk"
                ].mean()) if "_risk" in data.columns else float(rb),
                "n_players_selected": len(team),
            })
        except Exception:
            records.append({
                "risk_budget": float(rb),
                "total_tv": 0.0,
                "avg_risk": float(rb),
                "n_players_selected": 0,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    selected_team: pd.DataFrame | None = None,
    all_players_df: pd.DataFrame | None = None,
) -> go.Figure:
    """
    Efficient frontier chart with optional random-portfolio background and
    selected-team highlight.

    Parameters
    ----------
    frontier_df : pd.DataFrame
        Output of ``compute_efficient_frontier``.
    selected_team : pd.DataFrame or None
        If provided, marks the selected portfolio on the chart.
    all_players_df : pd.DataFrame or None
        If provided, used to generate 300 random background portfolios.
    """
    fig = go.Figure()

    # --- Background: random portfolios ---
    if all_players_df is not None:
        try:
            data_bg = _get_metrics(all_players_df)
            rng = np.random.default_rng(42)
            n = len(data_bg)
            rand_x, rand_y = [], []
            for _ in range(300):
                if n < 5:
                    break
                idx = rng.choice(n, size=min(5, n), replace=False)
                risks = data_bg["_risk"].values[idx]
                tvs = data_bg["_tv"].values[idx]
                rand_x.append(float(np.sqrt(np.mean(risks ** 2))))
                rand_y.append(float(np.sum(tvs)))
            fig.add_trace(go.Scatter(
                x=rand_x, y=rand_y,
                mode="markers",
                name="Случайные составы",
                marker=dict(
                    color="rgba(255,255,255,0.06)",
                    size=5,
                    line=dict(color="rgba(255,255,255,0.04)", width=0.3),
                ),
                hoverinfo="skip",
            ))
        except Exception:
            pass

    # --- Efficient frontier line ---
    fig.add_trace(go.Scatter(
        x=frontier_df["avg_risk"],
        y=frontier_df["total_tv"],
        mode="lines+markers",
        name="Эффективная граница",
        line=dict(color=NBA_RED, width=2.5),
        marker=dict(color=NBA_RED, size=6, line=dict(color="white", width=0.8)),
        hovertemplate=(
            "Бюджет риска: %{customdata:.0f}<br>"
            "Trade Value: %{y:.1f}<br>"
            "Avg Risk: %{x:.1f}<extra></extra>"
        ),
        customdata=frontier_df["risk_budget"],
    ))

    # --- Selected team marker ---
    if selected_team is not None and not selected_team.empty:
        try:
            port_risk = float(selected_team["portfolio_risk"].iloc[0])
            total_tv = float(selected_team["total_tv"].iloc[0])
            fig.add_trace(go.Scatter(
                x=[port_risk],
                y=[total_tv],
                mode="markers",
                name="Выбранный состав",
                marker=dict(
                    symbol="star",
                    color=GOLD,
                    size=18,
                    line=dict(color="white", width=1),
                ),
                hovertemplate=f"Состав: TV={total_tv:.1f}, Risk={port_risk:.1f}<extra></extra>",
            ))
        except Exception:
            pass

    fig.update_layout(
        title=dict(
            text="Эффективная граница Марковица — Составы NBA",
            font=dict(size=14, color="#e8eaf6"),
        ),
        height=480,
        template="nba_dark",
        xaxis=dict(title="Средний риск портфеля"),
        yaxis=dict(title="Суммарный Trade Value"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
    )
    return fig


def plot_risk_return_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Scatter of all players: injury_risk (X) vs trade_value (Y).

    Points sized by PTS, colored by trade_value.
    Quadrant lines at median risk and median TV with annotations.
    """
    data = _get_metrics(df)

    pts_col = next((c for c in ("PTS", "pts") if c in data.columns), None)
    pts_vals = data[pts_col].fillna(10).values if pts_col else np.full(len(data), 10)
    sizes = 6 + (pts_vals / pts_vals.max()) * 22 if pts_vals.max() > 0 else np.full(len(data), 10)

    name_col = next(
        (c for c in ("PLAYER_NAME", "player_name", "name", "Name") if c in data.columns),
        None,
    )
    names = data[name_col].values if name_col else [f"P{i}" for i in range(len(data))]

    cluster_col = "cluster_name" if "cluster_name" in data.columns else None

    median_risk = float(data["_risk"].median())
    median_tv = float(data["_tv"].median())

    customdata = []
    for i, row in data.iterrows():
        customdata.append([
            pts_vals[list(data.index).index(i)] if pts_col else 0,
            row.get("cluster_name", "") if cluster_col else "",
        ])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data["_risk"],
        y=data["_tv"],
        mode="markers",
        name="Игроки",
        marker=dict(
            color=data["_tv"],
            colorscale=[[0, "#0d1f3c"], [0.5, "#1D428A"], [1, "#C8102E"]],
            size=sizes,
            opacity=0.85,
            line=dict(color="rgba(255,255,255,0.15)", width=0.5),
            colorbar=dict(
                thickness=10,
                len=0.7,
                tickfont=dict(color="#6b7a99", size=10),
                title=dict(
                    text="Trade Value",
                    font=dict(color="#8b9ab5", size=11),
                ),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(255,255,255,0.06)",
            ),
        ),
        text=names,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Риск: %{x:.1f}<br>"
            "Trade Value: %{y:.1f}<br>"
            "PTS: %{customdata[0]:.1f}<br>"
            "Тип: %{customdata[1]}<extra></extra>"
        ),
        customdata=customdata,
    ))

    # Quadrant dividers
    fig.add_hline(
        y=median_tv,
        line=dict(color="rgba(255,255,255,0.20)", dash="dot", width=1),
    )
    fig.add_vline(
        x=median_risk,
        line=dict(color="rgba(255,255,255,0.20)", dash="dot", width=1),
    )

    # Quadrant annotations
    x_lo = float(data["_risk"].min())
    x_hi = float(data["_risk"].max())
    y_lo = float(data["_tv"].min())
    y_hi = float(data["_tv"].max())

    quadrant_labels = [
        (x_lo + (median_risk - x_lo) * 0.2, median_tv + (y_hi - median_tv) * 0.8,
         "Звёзды (высокий TV,<br>низкий риск)"),
        (median_risk + (x_hi - median_risk) * 0.6, median_tv + (y_hi - median_tv) * 0.8,
         "Рискованные звёзды"),
        (x_lo + (median_risk - x_lo) * 0.2, y_lo + (median_tv - y_lo) * 0.2,
         "Глубина состава"),
        (median_risk + (x_hi - median_risk) * 0.6, y_lo + (median_tv - y_lo) * 0.2,
         "Высокий риск,<br>низкий TV"),
    ]
    for ax, ay, text in quadrant_labels:
        fig.add_annotation(
            x=ax, y=ay,
            text=text,
            showarrow=False,
            font=dict(color="rgba(139,154,181,0.5)", size=9),
            align="center",
        )

    fig.update_layout(
        title=dict(
            text="Trade Value vs Риск травмы",
            font=dict(size=14, color="#e8eaf6"),
        ),
        height=500,
        template="nba_dark",
        xaxis=dict(title="Риск травмы"),
        yaxis=dict(title="Trade Value"),
    )
    return fig


def plot_team_composition(team_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal stacked bar showing each player's trade value contribution
    to the selected roster.
    """
    players: list[str] = []
    for _, row in team_df.iterrows():
        players.append(_get_player_name(row))

    tv_vals = team_df["trade_value"].astype(float).tolist()
    total_tv = sum(tv_vals)

    fig = go.Figure()

    cumulative = 0.0
    for i, (player, tv) in enumerate(zip(players, tv_vals)):
        color = NBA_PALETTE[i % len(NBA_PALETTE)]
        fig.add_trace(go.Bar(
            x=[tv],
            y=["Состав"],
            orientation="h",
            name=player,
            marker=dict(color=color, line=dict(width=0)),
            text=f"{player}<br>{tv:.0f}",
            textposition="inside",
            textfont=dict(size=10, color="white"),
            hovertemplate=f"<b>{player}</b><br>TV: {tv:.1f}<extra></extra>",
            base=cumulative,
        ))
        cumulative += tv

    # Total annotation
    fig.add_annotation(
        x=total_tv,
        y="Состав",
        text=f" Итого: {total_tv:.0f}",
        showarrow=False,
        xanchor="left",
        font=dict(color=GOLD, size=12),
    )

    fig.update_layout(
        title=dict(
            text="Состав: вклад каждого игрока",
            font=dict(size=13, color="#e8eaf6"),
        ),
        height=280,
        template="nba_dark",
        barmode="stack",
        xaxis=dict(title="Trade Value"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9),
        ),
        margin=dict(l=80, r=120, t=60, b=40),
    )
    return fig
