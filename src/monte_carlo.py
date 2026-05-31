"""
Monte Carlo season simulation module.

Simulates N_SIMS=8000 seasons per player using Normal(mu, sigma) where
sigma = mu * CV. CV increases for younger / less-played players so that
high-uncertainty profiles get wider distributions.

Uncertainty model:
  CV = 0.10 + max(0, (30 - gp) / 300) + age_adj
  where age_adj = 0.02 if age < 22, 0.01 if age > 34, else 0.

All random state is seeded from SEED=42 for reproducibility.
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
N_SIMS: int = 8000
SEED: int = 42

NBA_RED = "#C8102E"
NBA_BLUE = "#1D428A"
BG_CHART = "#0c1220"
TEXT_MUTED = "#8b9ab5"
GRID = "#141E30"
GOLD = "#FFC72C"

_STAT_LABELS: dict[str, str] = {
    "PTS": "Очки",
    "AST": "Передачи",
    "REB": "Подборы",
    "STL": "Перехваты",
    "BLK": "Блоки",
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_gp(row: pd.Series) -> float:
    """Return games played from whichever column is present."""
    for col in ("games_played", "GP", "G"):
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            return float(val)
    return 30.0


def _get_age(row: pd.Series) -> float:
    """Return player age from whichever column is present."""
    for col in ("PLAYER_AGE", "AGE", "age"):
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            return float(val)
    return 27.0


def _get_name(row: pd.Series) -> str:
    for col in ("PLAYER_NAME", "player_name", "name", "Name"):
        val = row.get(col)
        if val is not None and str(val).strip():
            return str(val)
    return "Unknown"


def _name_col_of(df: pd.DataFrame) -> str:
    """Return the player-name column that exists in *df*."""
    for c in ("PLAYER_NAME", "player_name", "name", "Name", "name_col"):
        if c in df.columns:
            return c
    return df.columns[0]


def _get_team(row: pd.Series) -> str:
    for col in ("TEAM_ABBREVIATION", "team", "Team", "TEAM"):
        val = row.get(col)
        if val is not None and str(val).strip():
            return str(val)
    return "N/A"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _cv_for_player(row: pd.Series) -> float:
    """
    Coefficient of variation for a player's stat distribution.

    CV = 0.10
       + max(0, (30 - gp) / 300)   # penalty for few games
       + age_adj                    # youth / veteran bonus uncertainty
    """
    gp: float = _get_gp(row)
    age: float = _get_age(row)

    cv = 0.10 + max(0.0, (30.0 - gp) / 300.0)
    if age < 22:
        cv += 0.02
    elif age > 34:
        cv += 0.01
    return float(cv)


def simulate_player(
    row: pd.Series,
    stat: str = "PTS",
    n_sims: int = N_SIMS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Simulate *n_sims* season outcomes for a single player for a given stat.

    Returns an array of shape (n_sims,) with values clipped to [0, inf).
    If the observed stat is <= 0, returns an array of zeros.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    mu = float(row.get(stat) or 0)
    if mu <= 0:
        return np.zeros(n_sims)

    sigma = mu * _cv_for_player(row)
    return np.clip(rng.normal(mu, sigma, n_sims), 0.0, None)


def compute_season_probabilities(
    df: pd.DataFrame,
    n_sims: int = N_SIMS,
) -> pd.DataFrame:
    """
    Run Monte Carlo simulations for every player in *df* and compute
    distribution statistics and threshold-crossing probabilities.

    Columns added to the output DataFrame:
      PTS_mean, PTS_p10, PTS_p90
      AST_mean, AST_p10, AST_p90
      REB_mean, REB_p10, REB_p90
      p_15plus, p_20plus, p_25plus, p_30plus   (% of sims where PTS >= threshold)
      p_allstar   (% of sims where PTS>=20 AND AST>=5 AND REB>=5)
      sim_cv      (CV * 100, percentage)
    """
    rng = np.random.default_rng(SEED)

    # Detect the player-name column so the output df preserves it
    _name_col = next(
        (c for c in ("PLAYER_NAME", "player_name", "name", "Name") if c in df.columns),
        "PLAYER_NAME",
    )

    records: list[dict] = []
    for _, row in df.iterrows():
        pts_sims = simulate_player(row, "PTS", n_sims, rng)
        ast_sims = simulate_player(row, "AST", n_sims, rng)
        reb_sims = simulate_player(row, "REB", n_sims, rng)

        n = float(n_sims)
        rec: dict = {
            _name_col: _get_name(row),
            "team": _get_team(row),
            "cluster_name": row.get("cluster_name", ""),
            # Distribution stats
            "PTS_mean": float(np.mean(pts_sims)),
            "PTS_p10":  float(np.percentile(pts_sims, 10)),
            "PTS_p90":  float(np.percentile(pts_sims, 90)),
            "AST_mean": float(np.mean(ast_sims)),
            "AST_p10":  float(np.percentile(ast_sims, 10)),
            "AST_p90":  float(np.percentile(ast_sims, 90)),
            "REB_mean": float(np.mean(reb_sims)),
            "REB_p10":  float(np.percentile(reb_sims, 10)),
            "REB_p90":  float(np.percentile(reb_sims, 90)),
            # Scoring probabilities
            "p_15plus": float(np.sum(pts_sims >= 15) / n * 100),
            "p_20plus": float(np.sum(pts_sims >= 20) / n * 100),
            "p_25plus": float(np.sum(pts_sims >= 25) / n * 100),
            "p_30plus": float(np.sum(pts_sims >= 30) / n * 100),
            # All-Star proxy
            "p_allstar": float(
                np.sum((pts_sims >= 20) & (ast_sims >= 5) & (reb_sims >= 5)) / n * 100
            ),
            "sim_cv": float(_cv_for_player(row) * 100),
        }
        records.append(rec)

    return pd.DataFrame(records)


def plot_player_simulation(
    row: pd.Series,
    stat: str = "PTS",
    n_sims: int = N_SIMS,
) -> go.Figure:
    """
    Histogram of simulated stat values for a single player.

    Shows:
    - Histogram of all simulated values (blue bars)
    - Shaded 80% prediction interval [p10, p90]
    - Vertical line for the observed (actual) value
    - Vertical line for the expected value E[X] = mu
    """
    rng = np.random.default_rng(SEED)
    sims = simulate_player(row, stat, n_sims, rng)

    mu = float(row.get(stat) or 0)
    actual = mu
    p10 = float(np.percentile(sims, 10))
    p90 = float(np.percentile(sims, 90))
    player_name = _get_name(row)
    stat_label = _STAT_LABELS.get(stat, stat)

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=sims,
        nbinsx=60,
        marker_color="rgba(29,66,138,0.55)",
        marker_line=dict(color="rgba(29,66,138,0.9)", width=0.5),
        name="Симуляции",
        hovertemplate=f"{stat_label}: %{{x:.1f}}<extra></extra>",
    ))

    # 80% interval shading
    fig.add_vrect(
        x0=p10, x1=p90,
        fillcolor="rgba(255,255,255,0.06)",
        line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"),
        annotation_text="80% интервал",
        annotation_position="top left",
        annotation_font=dict(color=TEXT_MUTED, size=10),
    )

    # Actual / observed value
    fig.add_vline(
        x=actual,
        line=dict(color=NBA_RED, dash="solid", width=2),
        annotation_text=f"Факт: {actual:.1f}",
        annotation_position="top right",
        annotation_font=dict(color=NBA_RED, size=11),
    )

    # Expected value E[X]
    fig.add_vline(
        x=mu,
        line=dict(color=GOLD, dash="dot", width=1.5),
        annotation_text=f"E[X]: {mu:.1f}",
        annotation_position="top left",
        annotation_font=dict(color=GOLD, size=11),
    )

    fig.update_layout(
        title=dict(
            text=f"Распределение симуляций — {player_name} · {stat_label}",
            font=dict(size=14, color="#e8eaf6"),
        ),
        height=360,
        template="nba_dark",
        showlegend=False,
        xaxis_title=stat_label,
        yaxis_title="Частота",
    )
    return fig


def plot_probability_leaderboard(
    df_probs: pd.DataFrame,
    metric: str = "p_allstar",
    n: int = 25,
) -> go.Figure:
    """
    Horizontal bar chart ranking players by a Monte Carlo probability metric.

    Color bands:
      >= 70%  →  NBA_RED   (elite)
      >= 40%  →  NBA_BLUE  (contender)
      <  40%  →  #4a566e   (developmental)
    """
    metric_labels: dict[str, str] = {
        "p_allstar": "All-Star (20pts / 5ast / 5reb)",
        "p_20plus":  "20+ очков",
        "p_25plus":  "25+ очков",
        "p_30plus":  "30+ очков",
        "p_15plus":  "15+ очков",
    }

    nc = _name_col_of(df_probs)
    top = (
        df_probs[[nc, metric]]
        .dropna()
        .sort_values(metric, ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

    colors = [
        NBA_RED if v >= 70 else NBA_BLUE if v >= 40 else "#4a566e"
        for v in top[metric]
    ]

    fig = go.Figure(go.Bar(
        x=top[metric],
        y=top[nc],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in top[metric]],
        textposition="outside",
        textfont=dict(color=TEXT_MUTED, size=10),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))

    label = metric_labels.get(metric, metric)
    fig.update_layout(
        title=dict(
            text=f"Вероятность достижения — {label} (Monte Carlo, 8,000 симуляций)",
            font=dict(size=13, color="#e8eaf6"),
        ),
        height=max(380, n * 22),
        template="nba_dark",
        xaxis=dict(title="Вероятность (%)", range=[0, max(top[metric].max() * 1.15, 10)]),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=160, r=60, t=60, b=40),
    )
    return fig


def plot_uncertainty_scatter(df_probs: pd.DataFrame) -> go.Figure:
    """
    Scatter plot: Expected PTS (X) vs Coefficient of Variation / Uncertainty (Y).

    Points sized by p_allstar probability.  Three groups are distinguished:
      - All-Star candidates (p_allstar >= 60%)
      - Potential starters  (30% <= p_allstar < 60%)
      - Others              (p_allstar < 30%)
    """
    df = df_probs.copy().dropna(subset=["PTS_mean", "sim_cv", "p_allstar"])
    nc = _name_col_of(df)

    group_defs = [
        ("All-Star кандидаты (≥60%)", df["p_allstar"] >= 60, NBA_RED),
        ("Потенциал (30-60%)",         (df["p_allstar"] >= 30) & (df["p_allstar"] < 60), "#FFC72C"),
        ("Остальные (<30%)",            df["p_allstar"] < 30,  "#4a566e"),
    ]

    fig = go.Figure()
    for group_name, mask, color in group_defs:
        sub = df[mask]
        if sub.empty:
            continue
        sizes = 8 + (sub["p_allstar"] / 100) * 24
        fig.add_trace(go.Scatter(
            x=sub["PTS_mean"],
            y=sub["sim_cv"],
            mode="markers+text",
            name=group_name,
            marker=dict(
                color=color,
                size=sizes,
                opacity=0.85,
                line=dict(color="rgba(255,255,255,0.2)", width=0.5),
            ),
            text=sub[nc].apply(lambda n: n.split()[-1] if " " in n else n),
            textposition="top center",
            textfont=dict(size=8, color="rgba(255,255,255,0.5)"),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "E[PTS]: %{x:.1f}<br>"
                "CV: %{y:.1f}%<br>"
                "P(All-Star): %{customdata[1]:.1f}%<extra></extra>"
            ),
            customdata=list(zip(sub[nc], sub["p_allstar"])),
        ))

    # High uncertainty threshold
    fig.add_hline(
        y=15,
        line=dict(color="rgba(255,255,255,0.3)", dash="dot", width=1),
        annotation_text="Высокая нестабильность",
        annotation_position="top right",
        annotation_font=dict(color=TEXT_MUTED, size=10),
    )

    fig.update_layout(
        title=dict(
            text="Ожидаемые очки vs Неопределённость",
            font=dict(size=14, color="#e8eaf6"),
        ),
        height=480,
        template="nba_dark",
        xaxis=dict(title="E[PTS] — ожидаемые очки за игру"),
        yaxis=dict(title="CV — коэффициент вариации (%)"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
    )
    return fig
