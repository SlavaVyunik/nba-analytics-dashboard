"""
Bayesian statistics for NBA player performance estimation.

Two conjugate models are implemented:
  1. Beta-Binomial for shooting percentages (FG_PCT, FG3_PCT, FT_PCT, TS_PCT).
     Prior parameters are calibrated to NBA league averages so that
     small-sample estimates are pulled toward the league mean.

  2. Normal-Normal conjugate for counting stats (PTS, AST, REB, STL, BLK).
     Uses a known-variance approximation: the posterior mean is a weighted
     combination of the observed mean and the prior mean, with weights
     determined by the effective sample sizes (kappa_0, n_games).

Shrinkage: players with fewer games played receive stronger shrinkage toward
the prior, naturally regularising sparse observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import beta as beta_dist, norm as norm_dist
from src.theme_utils import ensure_template

# ---------------------------------------------------------------------------
# Prior parameters
# ---------------------------------------------------------------------------
_TPL = ensure_template()   # register nba_dark if not yet registered
SHOOTING_PRIORS: dict[str, dict[str, float]] = {
    "FG_PCT":  {"alpha": 44.0, "beta": 56.0},
    "FG3_PCT": {"alpha": 35.0, "beta": 65.0},
    "FT_PCT":  {"alpha": 77.0, "beta": 23.0},
    "TS_PCT":  {"alpha": 56.0, "beta": 44.0},
}

COUNTING_PRIORS: dict[str, dict[str, float]] = {
    "PTS": {"mu_0": 10.5, "sigma_0": 5.0, "kappa_0": 10.0},
    "AST": {"mu_0": 2.5,  "sigma_0": 2.0, "kappa_0": 10.0},
    "REB": {"mu_0": 4.2,  "sigma_0": 2.5, "kappa_0": 10.0},
    "STL": {"mu_0": 0.9,  "sigma_0": 0.5, "kappa_0": 10.0},
    "BLK": {"mu_0": 0.5,  "sigma_0": 0.4, "kappa_0": 10.0},
}

# Approximate attempts per game for converting percentage stats
ATTEMPTS_PER_GAME: dict[str, int] = {
    "FG_PCT":  14,
    "FG3_PCT":  6,
    "FT_PCT":   4,
    "TS_PCT":  14,
}

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------
NBA_RED = "#C8102E"
NBA_BLUE = "#1D428A"
BG_CHART = "#0c1220"
TEXT_MUTED = "#8b9ab5"
GRID = "#141E30"
GOLD = "#FFC72C"

_STAT_LABELS: dict[str, str] = {
    "FG_PCT":  "FG% (точность с игры)",
    "FG3_PCT": "3PT% (точность из-за дуги)",
    "FT_PCT":  "FT% (штрафные броски)",
    "TS_PCT":  "TS% (истинная точность)",
    "PTS":     "Очки",
    "AST":     "Передачи",
    "REB":     "Подборы",
    "STL":     "Перехваты",
    "BLK":     "Блоки",
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_gp(row: pd.Series) -> float:
    for col in ("games_played", "GP", "G"):
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            return max(1.0, float(val))
    return 30.0


def _get_name(row: pd.Series) -> str:
    for col in ("PLAYER_NAME", "player_name", "name", "Name"):
        val = row.get(col)
        if val is not None and str(val).strip():
            return str(val)
    return "Unknown"


# ---------------------------------------------------------------------------
# Core Bayesian estimators
# ---------------------------------------------------------------------------

def bayesian_shooting(
    pct_obs: float,
    n_attempts: float,
    stat: str = "FG_PCT",
    ci: float = 0.95,
) -> dict:
    """
    Beta-Binomial Bayesian update for a shooting percentage.

    Parameters
    ----------
    pct_obs : float
        Observed shooting percentage (0–1 scale).
    n_attempts : float
        Total number of shot attempts.
    stat : str
        Which shooting stat to look up the prior for.
    ci : float
        Credible interval width (default 0.95 → 95% CI).

    Returns
    -------
    dict with keys:
        posterior_mean, ci_low, ci_high, shrinkage, alpha_n, beta_n
    """
    prior = SHOOTING_PRIORS.get(stat, {"alpha": 50.0, "beta": 50.0})
    alpha_0 = prior["alpha"]
    beta_0 = prior["beta"]

    # Clamp inputs
    pct_obs = float(np.clip(pct_obs, 0.0, 1.0))
    n_attempts = max(0.0, float(n_attempts))

    successes = pct_obs * n_attempts
    failures = (1.0 - pct_obs) * n_attempts

    alpha_n = alpha_0 + successes
    beta_n = beta_0 + failures

    posterior_mean = alpha_n / (alpha_n + beta_n)

    alpha_ci = (1.0 - ci) / 2.0
    ci_low = float(beta_dist.ppf(alpha_ci, alpha_n, beta_n))
    ci_high = float(beta_dist.ppf(1.0 - alpha_ci, alpha_n, beta_n))

    shrinkage = (alpha_0 + beta_0) / (alpha_0 + beta_0 + n_attempts)

    return {
        "posterior_mean": float(posterior_mean),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "shrinkage": float(shrinkage),
        "alpha_n": float(alpha_n),
        "beta_n": float(beta_n),
    }


def bayesian_counting(
    obs_mean: float,
    n_games: float,
    stat: str = "PTS",
    ci: float = 0.95,
) -> dict:
    """
    Normal-Normal conjugate Bayesian update for a counting stat.

    Parameters
    ----------
    obs_mean : float
        Observed per-game average.
    n_games : float
        Number of games played (effective sample size).
    stat : str
        Which counting stat to look up the prior for.
    ci : float
        Credible interval width.

    Returns
    -------
    dict with keys:
        posterior_mean, ci_low, ci_high, shrinkage
    """
    prior = COUNTING_PRIORS.get(stat, {"mu_0": 5.0, "sigma_0": 3.0, "kappa_0": 10.0})
    mu_0 = float(prior["mu_0"])
    sigma_0 = float(prior["sigma_0"])
    kappa_0 = float(prior["kappa_0"])

    n_games = max(0.0, float(n_games))
    obs_mean = max(0.0, float(obs_mean))

    kappa_n = kappa_0 + n_games
    mu_n = (kappa_0 * mu_0 + n_games * obs_mean) / kappa_n
    sigma_n = sigma_0 * np.sqrt(kappa_0 / kappa_n)

    alpha_ci = (1.0 - ci) / 2.0
    ci_low = float(norm_dist.ppf(alpha_ci, mu_n, sigma_n))
    ci_high = float(norm_dist.ppf(1.0 - alpha_ci, mu_n, sigma_n))

    shrinkage = kappa_0 / kappa_n

    return {
        "posterior_mean": float(mu_n),
        "ci_low": max(0.0, ci_low),
        "ci_high": ci_high,
        "shrinkage": float(shrinkage),
    }


# ---------------------------------------------------------------------------
# DataFrame-level computation
# ---------------------------------------------------------------------------

def compute_bayesian_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Bayesian estimation to all relevant columns in *df*.

    For each shooting stat found in *df*:
        Adds {stat}_bayes, {stat}_ci_low, {stat}_ci_high, {stat}_shrink

    For each counting stat found in *df*:
        Adds {stat}_bayes, {stat}_ci_low, {stat}_ci_high, {stat}_shrink

    Returns the augmented DataFrame (copy).
    """
    result = df.copy()

    def _process_row_shooting(row: pd.Series, stat: str) -> pd.Series:
        gp = _get_gp(row)
        att = gp * ATTEMPTS_PER_GAME.get(stat, 10)
        raw = float(row.get(stat) or 0.0)
        # Detect if value is in 0–100 scale; normalise to 0–1
        if raw > 1.0:
            raw = raw / 100.0
        est = bayesian_shooting(raw, att, stat)
        return pd.Series({
            f"{stat}_bayes": est["posterior_mean"],
            f"{stat}_ci_low": est["ci_low"],
            f"{stat}_ci_high": est["ci_high"],
            f"{stat}_shrink": est["shrinkage"] * 100,
        })

    def _process_row_counting(row: pd.Series, stat: str) -> pd.Series:
        gp = _get_gp(row)
        obs = float(row.get(stat) or 0.0)
        est = bayesian_counting(obs, gp, stat)
        return pd.Series({
            f"{stat}_bayes": est["posterior_mean"],
            f"{stat}_ci_low": est["ci_low"],
            f"{stat}_ci_high": est["ci_high"],
            f"{stat}_shrink": est["shrinkage"] * 100,
        })

    # Shooting stats
    for stat in SHOOTING_PRIORS:
        if stat in df.columns:
            extra = df.apply(_process_row_shooting, axis=1, stat=stat)
            for col in extra.columns:
                result[col] = extra[col].values

    # Counting stats
    for stat in ("PTS", "AST", "REB", "STL", "BLK"):
        if stat in df.columns:
            extra = df.apply(_process_row_counting, axis=1, stat=stat)
            for col in extra.columns:
                result[col] = extra[col].values

    return result


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_shrinkage_dotplot(
    df_bayes: pd.DataFrame,
    stat: str = "FG_PCT",
    n: int = 25,
) -> go.Figure:
    """
    Dot-and-interval plot comparing observed values to posterior estimates.

    For each player:
      - Horizontal blue line = 95% credible interval
      - Gray open circle = observed raw value
      - Red diamond = posterior mean
      - Yellow dotted vertical line = prior mean
    """
    bayes_col = f"{stat}_bayes"
    ci_lo_col = f"{stat}_ci_low"
    ci_hi_col = f"{stat}_ci_high"

    required = [stat, bayes_col, ci_lo_col, ci_hi_col]
    missing = [c for c in required if c not in df_bayes.columns]
    if missing:
        raise ValueError(f"Missing columns for shrinkage dotplot: {missing}")

    label = _STAT_LABELS.get(stat, stat)
    top = (
        df_bayes[["PLAYER_NAME" if "PLAYER_NAME" in df_bayes.columns else
                   next((c for c in ("player_name", "name", "Name") if c in df_bayes.columns), None)]
                + [stat, bayes_col, ci_lo_col, ci_hi_col]]
        if False else  # guard for dynamic column selection below
        None
    )

    # Dynamic name column selection
    name_col = next(
        (c for c in ("PLAYER_NAME", "player_name", "name", "Name") if c in df_bayes.columns),
        None,
    )
    cols = ([name_col] if name_col else []) + required
    top = (
        df_bayes[[c for c in cols if c in df_bayes.columns]]
        .dropna(subset=[stat])
        .sort_values(stat, ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

    names = top[name_col].tolist() if name_col else [f"P{i}" for i in range(len(top))]

    # Prior mean for vertical reference
    prior = SHOOTING_PRIORS.get(stat, {"alpha": 50.0, "beta": 50.0})
    prior_mean = prior["alpha"] / (prior["alpha"] + prior["beta"])

    fig = go.Figure()

    # CI intervals
    for i, row in top.iterrows():
        lo = float(row[ci_lo_col])
        hi = float(row[ci_hi_col])
        fig.add_shape(
            type="line",
            x0=lo, x1=hi,
            y0=names[i], y1=names[i],
            line=dict(color=NBA_BLUE, width=2),
        )

    # Observed values (gray open circles)
    raw_vals = top[stat].values.copy().astype(float)
    if raw_vals.max() > 1.0:
        raw_vals = raw_vals / 100.0

    fig.add_trace(go.Scatter(
        x=raw_vals,
        y=names,
        mode="markers",
        name="Наблюдаемое",
        marker=dict(
            symbol="circle-open",
            color="rgba(180,180,180,0.7)",
            size=10,
            line=dict(width=1.5, color="rgba(180,180,180,0.7)"),
        ),
        hovertemplate="Наблюдаемое: %{x:.3f}<extra></extra>",
    ))

    # Posterior means (red diamonds)
    fig.add_trace(go.Scatter(
        x=top[bayes_col].values,
        y=names,
        mode="markers",
        name="Апостериорная оценка",
        marker=dict(
            symbol="diamond",
            color=NBA_RED,
            size=9,
            line=dict(width=0.5, color="white"),
        ),
        hovertemplate="Байес: %{x:.3f}<extra></extra>",
    ))

    # Prior mean reference line
    fig.add_vline(
        x=prior_mean,
        line=dict(color=GOLD, dash="dot", width=1.5),
        annotation_text=f"Приор: {prior_mean:.3f}",
        annotation_position="top right",
        annotation_font=dict(color=GOLD, size=10),
    )

    fig.update_layout(
        title=dict(
            text=f"Байесовская регуляризация — {label} (топ-{n})",
            font=dict(size=13, color="#e8eaf6"),
        ),
        height=max(380, n * 18 + 80),
        template="nba_dark",
        xaxis=dict(title=label),
        yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        margin=dict(l=160, r=60, t=60, b=40),
    )
    return fig


def plot_shrinkage_scatter(
    df_bayes: pd.DataFrame,
    stat: str = "FG_PCT",
) -> go.Figure:
    """
    Scatter: observed stat (X) vs Bayesian posterior mean (Y).

    Points colored by shrinkage percentage.  Diagonal y=x reference line
    shows where raw observed == posterior (no shrinkage).
    """
    bayes_col = f"{stat}_bayes"
    shrink_col = f"{stat}_shrink"

    for col in (stat, bayes_col, shrink_col):
        if col not in df_bayes.columns:
            raise ValueError(f"Missing column '{col}' — run compute_bayesian_df first.")

    label = _STAT_LABELS.get(stat, stat)
    name_col = next(
        (c for c in ("PLAYER_NAME", "player_name", "name", "Name") if c in df_bayes.columns),
        None,
    )

    sub = df_bayes.dropna(subset=[stat, bayes_col, shrink_col]).copy()
    raw = sub[stat].values.astype(float)
    if raw.max() > 1.0:
        raw = raw / 100.0

    post = sub[bayes_col].values.astype(float)
    shrink = sub[shrink_col].values.astype(float)
    names = sub[name_col].tolist() if name_col else [f"P{i}" for i in range(len(sub))]

    x_min = float(min(raw.min(), post.min())) * 0.98
    x_max = float(max(raw.max(), post.max())) * 1.02

    fig = go.Figure()

    # Diagonal y = x
    fig.add_trace(go.Scatter(
        x=[x_min, x_max],
        y=[x_min, x_max],
        mode="lines",
        name="y = x (нет усадки)",
        line=dict(color="rgba(255,255,255,0.25)", dash="dot", width=1.2),
        hoverinfo="skip",
    ))

    # Main scatter
    fig.add_trace(go.Scatter(
        x=raw,
        y=post,
        mode="markers",
        name="Игроки",
        marker=dict(
            color=shrink,
            colorscale=[[0, "#C8102E"], [0.5, "#1D428A"], [1, "#00D4AA"]],
            size=8,
            opacity=0.85,
            line=dict(color="rgba(255,255,255,0.15)", width=0.5),
            colorbar=dict(
                thickness=10,
                len=0.7,
                tickfont=dict(color="#6b7a99", size=10),
                title=dict(
                    text="Усадка %",
                    font=dict(color="#8b9ab5", size=11),
                ),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(255,255,255,0.06)",
            ),
        ),
        text=names,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Наблюдаемое: %{x:.3f}<br>"
            "Байес: %{y:.3f}<br>"
            "Усадка: %{marker.color:.1f}%<extra></extra>"
        ),
    ))

    fig.add_annotation(
        x=x_max * 0.98,
        y=x_min + (x_max - x_min) * 0.08,
        text="← Притяжение к приору",
        showarrow=False,
        xanchor="right",
        font=dict(color=TEXT_MUTED, size=10),
    )

    fig.update_layout(
        title=dict(
            text=f"Наблюдаемое vs Байесовское — {label}",
            font=dict(size=13, color="#e8eaf6"),
        ),
        height=450,
        template="nba_dark",
        xaxis=dict(title=f"Наблюдаемое {label}"),
        yaxis=dict(title=f"Апостериорная оценка {label}"),
    )
    return fig


def plot_posterior_distribution(
    row: pd.Series,
    stat: str = "FG_PCT",
) -> go.Figure:
    """
    Full posterior Beta distribution curve for a single player's shooting stat.

    Shows:
    - Shaded 95% credible interval
    - Prior distribution curve (muted)
    - Posterior distribution curve
    - Vertical line for observed value
    - Vertical line for posterior mean
    """
    label = _STAT_LABELS.get(stat, stat)
    player_name = _get_name(row)

    prior = SHOOTING_PRIORS.get(stat, {"alpha": 50.0, "beta": 50.0})
    alpha_0 = float(prior["alpha"])
    beta_0 = float(prior["beta"])

    gp = _get_gp(row)
    att = gp * ATTEMPTS_PER_GAME.get(stat, 10)
    raw = float(row.get(stat) or 0.0)
    if raw > 1.0:
        raw = raw / 100.0
    raw = float(np.clip(raw, 0.0, 1.0))

    est = bayesian_shooting(raw, att, stat)
    alpha_n = est["alpha_n"]
    beta_n = est["beta_n"]
    posterior_mean = est["posterior_mean"]
    ci_lo = est["ci_low"]
    ci_hi = est["ci_high"]

    x = np.linspace(0.001, 0.999, 300)
    y_post = beta_dist.pdf(x, alpha_n, beta_n)
    y_prior = beta_dist.pdf(x, alpha_0, beta_0)

    # 95% CI shading mask
    ci_mask = (x >= ci_lo) & (x <= ci_hi)

    fig = go.Figure()

    # CI shading
    fig.add_trace(go.Scatter(
        x=np.concatenate([x[ci_mask], x[ci_mask][::-1]]),
        y=np.concatenate([y_post[ci_mask], np.zeros_like(x[ci_mask])]),
        fill="toself",
        fillcolor="rgba(29,66,138,0.20)",
        line=dict(width=0),
        name="95% доверительный интервал",
        hoverinfo="skip",
    ))

    # Prior distribution
    fig.add_trace(go.Scatter(
        x=x,
        y=y_prior,
        mode="lines",
        name="Приор",
        line=dict(color="rgba(139,154,181,0.40)", width=1.5, dash="dot"),
        hovertemplate="Приор: %{x:.3f}<extra></extra>",
    ))

    # Posterior distribution
    fig.add_trace(go.Scatter(
        x=x,
        y=y_post,
        mode="lines",
        name="Апостериор",
        line=dict(color=NBA_BLUE, width=2.5),
        hovertemplate="Апостериор: p=%{x:.3f}, плотность=%{y:.2f}<extra></extra>",
    ))

    # Observed value line
    y_max = float(y_post.max()) * 1.05
    fig.add_vline(
        x=raw,
        line=dict(color=NBA_RED, dash="solid", width=2),
        annotation_text=f"Факт: {raw:.3f}",
        annotation_position="top right",
        annotation_font=dict(color=NBA_RED, size=11),
    )

    # Posterior mean line
    fig.add_vline(
        x=posterior_mean,
        line=dict(color=GOLD, dash="dash", width=1.8),
        annotation_text=f"Баес. оценка: {posterior_mean:.3f}",
        annotation_position="top left",
        annotation_font=dict(color=GOLD, size=11),
    )

    fig.update_layout(
        title=dict(
            text=f"Апостериорное распределение — {player_name} · {label}",
            font=dict(size=13, color="#e8eaf6"),
        ),
        height=320,
        template="nba_dark",
        xaxis=dict(title=label, range=[0, 1]),
        yaxis=dict(title="Плотность вероятности"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
    )
    return fig
