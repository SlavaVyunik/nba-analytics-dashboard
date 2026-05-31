"""
Career survival analysis for NBA players.

Models career "decline events" as a survival problem:
  - Event: performance decline proxy, defined as (age > 31) AND (injury_risk > 58).
  - Survival time: synthetic duration generated from player risk profile.

All survival math (Kaplan-Meier, variance estimation, CI transforms) is
implemented from scratch using NumPy/SciPy — no lifelines.

A LogisticRegression is used as a Cox PH approximation to identify risk
factors, with bootstrap confidence intervals for hazard ratios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm as norm_dist
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.theme_utils import ensure_template

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TPL = ensure_template()   # register nba_dark if not yet registered

NBA_RED = "#C8102E"
NBA_BLUE = "#1D428A"
BG_CHART = "#0c1220"
TEXT_MUTED = "#8b9ab5"
GRID = "#141E30"
GOLD = "#FFC72C"

AGE_COL_OPTIONS = ("PLAYER_AGE", "AGE", "age")

_KM_COLORS = ["#C8102E", "#FFC72C", "#27AE60", "#1D428A", "#9B59B6"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_age(row: pd.Series) -> float:
    for col in AGE_COL_OPTIONS:
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


# ---------------------------------------------------------------------------
# Kaplan-Meier estimator from scratch
# ---------------------------------------------------------------------------

def kaplan_meier(
    times: np.ndarray,
    events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pure NumPy Kaplan-Meier survival estimator with 95% CI via log(-log) transform.

    Parameters
    ----------
    times : np.ndarray
        Observation times (non-negative).
    events : np.ndarray
        Binary event indicators (1 = event occurred, 0 = censored).

    Returns
    -------
    t_arr : np.ndarray
        Unique event times prepended with t=0.
    S_arr : np.ndarray
        Survival function S(t) estimates prepended with S(0)=1.
    S_lo_arr : np.ndarray
        Lower bound of 95% CI for S(t), clipped to [0, 1].
    S_hi_arr : np.ndarray
        Upper bound of 95% CI for S(t), clipped to [0, 1].
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)

    # Sort by time
    order = np.argsort(times)
    times = times[order]
    events = events[order]

    unique_times = np.unique(times[events == 1])
    if len(unique_times) == 0:
        # No events: trivial survival
        t_out = np.array([0.0])
        S_out = np.array([1.0])
        return t_out, S_out, S_out.copy(), S_out.copy()

    n_total = len(times)
    S = 1.0
    var_log = 0.0  # Greenwood accumulator for log(S)

    t_list: list[float] = [0.0]
    S_list: list[float] = [1.0]
    var_list: list[float] = [0.0]

    # Index pointer for at-risk count
    for t_j in unique_times:
        # At-risk: number of subjects with time >= t_j
        n_j = int(np.sum(times >= t_j))
        # Events at exactly t_j
        d_j = int(np.sum((times == t_j) & (events == 1)))

        if n_j == 0:
            continue

        S *= (1.0 - d_j / n_j)

        # Greenwood variance increment
        if n_j > d_j:
            var_log += d_j / (n_j * (n_j - d_j))

        t_list.append(float(t_j))
        S_list.append(float(S))
        var_list.append(float(var_log))

    t_arr = np.array(t_list)
    S_arr = np.array(S_list)
    var_arr = np.array(var_list)

    # 95% CI via log(-log(S)) = complementary log-log transform
    S_lo_list: list[float] = [1.0]
    S_hi_list: list[float] = [1.0]

    for i in range(1, len(S_arr)):
        s = S_arr[i]
        v = var_arr[i]
        if s <= 0 or s >= 1:
            S_lo_list.append(float(np.clip(s, 0.0, 1.0)))
            S_hi_list.append(float(np.clip(s, 0.0, 1.0)))
            continue

        log_s = np.log(s)
        if log_s >= 0 or v <= 0:
            S_lo_list.append(float(np.clip(s, 0.0, 1.0)))
            S_hi_list.append(float(np.clip(s, 0.0, 1.0)))
            continue

        theta = np.log(-log_s)
        se_theta = np.sqrt(v) / abs(log_s)
        half = 1.96 * se_theta

        s_lo = float(np.exp(-np.exp(theta + half)))
        s_hi = float(np.exp(-np.exp(theta - half)))
        S_lo_list.append(float(np.clip(s_lo, 0.0, 1.0)))
        S_hi_list.append(float(np.clip(s_hi, 0.0, 1.0)))

    S_lo_arr = np.array(S_lo_list)
    S_hi_arr = np.array(S_hi_list)

    return t_arr, S_arr, S_lo_arr, S_hi_arr


# ---------------------------------------------------------------------------
# Survival data preparation
# ---------------------------------------------------------------------------

def prepare_survival_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic survival times and event indicators from player profiles.

    Survival time model:
      base ~ Exponential(rate = risk / 500)
      * 1.4 if peak age (23–30)
      * 0.65 if veteran (> 33)

    Event definition:
      observed = 1 if (age > 31) AND (injury_risk > 58) else 0

    Stratification:
      risk_group: 🔴 Высокий риск / 🟡 Умеренный / 🟢 Низкий
      age_group:  Молодой (<25) / Расцвет (25-30) / Ветеран (>30)

    Returns a copy of *df* with added columns.
    """
    rng = np.random.default_rng(42)
    result = df.copy()

    ages: list[float] = []
    risks: list[float] = []
    times: list[float] = []
    observed: list[int] = []
    risk_groups: list[str] = []
    age_groups: list[str] = []

    risk_col = next(
        (c for c in ("injury_risk", "risk", "INJURY_RISK") if c in df.columns), None
    )

    for _, row in df.iterrows():
        age = _get_age(row)
        risk = float(row.get(risk_col, 50.0) if risk_col else 50.0)
        if np.isnan(risk):
            risk = 50.0

        # Survival time
        rate = risk / 500.0
        base_time = rng.exponential(1.0 / max(rate, 0.001))
        if 23 <= age <= 30:
            base_time *= 1.4
        elif age > 33:
            base_time *= 0.65

        # Event
        evt = 1 if (age > 31 and risk > 58) else 0

        # Risk group
        if risk >= 65:
            rg = "🔴 Высокий риск"
        elif risk >= 45:
            rg = "🟡 Умеренный"
        else:
            rg = "🟢 Низкий"

        # Age group
        if age < 25:
            ag = "Молодой (<25)"
        elif age <= 30:
            ag = "Расцвет (25-30)"
        else:
            ag = "Ветеран (>30)"

        ages.append(age)
        risks.append(risk)
        times.append(base_time)
        observed.append(evt)
        risk_groups.append(rg)
        age_groups.append(ag)

    result["age"] = ages
    result["injury_risk"] = risks
    result["time"] = times
    result["observed"] = observed
    result["risk_group"] = risk_groups
    result["age_group"] = age_groups

    # Preserve name and cluster columns
    name_col = next(
        (c for c in ("PLAYER_NAME", "player_name", "name", "Name") if c in df.columns), None
    )
    if name_col and "name_col" not in result.columns:
        result["name_col"] = df[name_col].values

    return result


# ---------------------------------------------------------------------------
# Plotting: Kaplan-Meier curves
# ---------------------------------------------------------------------------

def plot_km_by_group(
    df_surv: pd.DataFrame,
    group_col: str = "risk_group",
) -> go.Figure:
    """
    Kaplan-Meier survival curves stratified by *group_col*.

    Each group shows:
    - Step-function survival curve (shape="hv")
    - Shaded 95% CI band
    - Legend entry with group name, n, and median survival T₅₀
    """
    if group_col not in df_surv.columns:
        raise ValueError(f"Column '{group_col}' not found in df_surv.")

    groups = sorted(df_surv[group_col].dropna().unique().tolist())
    group_titles = {
        "risk_group": "Группа риска",
        "age_group":  "Возрастная группа",
    }
    title_suffix = group_titles.get(group_col, group_col)

    fig = go.Figure()

    for gi, group in enumerate(groups):
        color = _KM_COLORS[gi % len(_KM_COLORS)]
        sub = df_surv[df_surv[group_col] == group].dropna(subset=["time", "observed"])
        if len(sub) < 2:
            continue

        t_arr, S_arr, S_lo, S_hi = kaplan_meier(
            sub["time"].values, sub["observed"].values
        )
        n = len(sub)

        # Median survival time (first time S <= 0.5)
        below = np.where(S_arr <= 0.5)[0]
        median_t = float(t_arr[below[0]]) if len(below) > 0 else float(t_arr[-1])

        legend_label = f"{group} (n={n}, T₅₀={median_t:.1f})"

        # CI band
        rgba_fill = color.lstrip("#")
        r = int(rgba_fill[0:2], 16)
        g_val = int(rgba_fill[2:4], 16)
        b = int(rgba_fill[4:6], 16)

        # Build step function arrays for CI band
        ci_x = np.concatenate([t_arr, t_arr[::-1]])
        ci_y = np.concatenate([S_hi, S_lo[::-1]])
        fig.add_trace(go.Scatter(
            x=ci_x, y=ci_y,
            fill="toself",
            fillcolor=f"rgba({r},{g_val},{b},0.12)",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        ))

        # Survival curve
        fig.add_trace(go.Scatter(
            x=t_arr,
            y=S_arr,
            mode="lines",
            name=legend_label,
            line=dict(color=color, width=2.2, shape="hv"),
            hovertemplate=f"{group}<br>t=%{{x:.2f}}, S(t)=%{{y:.3f}}<extra></extra>",
        ))

    # S(t) = 0.5 reference
    fig.add_hline(
        y=0.5,
        line=dict(color="rgba(255,255,255,0.25)", dash="dot", width=1),
        annotation_text="S(t)=0.5",
        annotation_position="top right",
        annotation_font=dict(color=TEXT_MUTED, size=10),
    )

    fig.update_layout(
        title=dict(
            text=f"Кривые Каплана-Мейера — {title_suffix}",
            font=dict(size=14, color="#e8eaf6"),
        ),
        height=440,
        template="nba_dark",
        xaxis=dict(title="Время (условные единицы)"),
        yaxis=dict(title="S(t) — вероятность выживания", range=[-0.05, 1.05]),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Cox PH approximation
# ---------------------------------------------------------------------------

def cox_hazard_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Logistic-regression-based Cox PH approximation for career decline risk factors.

    Uses bootstrap (200 iterations) to compute 95% CI for hazard ratios.

    Feature list (standardised): age, injury_risk, efficiency (inverted),
    USG_PCT, MIN.  Missing features are silently dropped.

    Returns
    -------
    pd.DataFrame with columns:
        Фактор, Hazard Ratio, CI_2.5%, CI_97.5%, Direction
    """
    # All candidate feature columns
    feature_candidates = {
        "age": "Возраст",
        "injury_risk": "Риск травмы",
        "efficiency": "Эффективность",
        "USG_PCT": "% использования",
        "MIN": "Минуты",
    }

    # Only keep features actually present in df
    avail_features = [f for f in feature_candidates if f in df.columns]
    if not avail_features:
        return pd.DataFrame(columns=["Фактор", "Hazard Ratio", "CI_2.5%", "CI_97.5%", "Direction"])

    # Build X and y
    X_raw = df[avail_features].copy()

    # Invert efficiency: higher efficiency = lower hazard
    if "efficiency" in X_raw.columns:
        X_raw["efficiency"] = -X_raw["efficiency"]

    X_raw = X_raw.fillna(X_raw.median())

    # Target: decline event proxy
    age_col = "age" if "age" in df.columns else None
    risk_col_target = "injury_risk" if "injury_risk" in df.columns else None

    if age_col and risk_col_target:
        y = ((df[age_col] > 32) | (df[risk_col_target] > 60)).astype(int).values
    elif risk_col_target:
        y = (df[risk_col_target] > 60).astype(int).values
    elif age_col:
        y = (df[age_col] > 32).astype(int).values
    else:
        return pd.DataFrame(columns=["Фактор", "Hazard Ratio", "CI_2.5%", "CI_97.5%", "Direction"])

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw.values)
    n_features = X.shape[1]

    rng = np.random.default_rng(42)
    n_obs = len(y)
    n_boot = 200

    coef_boots = np.zeros((n_boot, n_features))
    for b in range(n_boot):
        idx = rng.integers(0, n_obs, size=n_obs)
        X_b, y_b = X[idx], y[idx]
        if len(np.unique(y_b)) < 2:
            coef_boots[b] = 0.0
            continue
        try:
            model = LogisticRegression(max_iter=300, solver="lbfgs", C=1.0)
            model.fit(X_b, y_b)
            coef_boots[b] = model.coef_[0]
        except Exception:
            coef_boots[b] = 0.0

    hr_boots = np.exp(coef_boots)       # (n_boot, n_features)
    hr_mean = np.exp(np.median(coef_boots, axis=0))
    hr_lo = np.percentile(hr_boots, 2.5, axis=0)
    hr_hi = np.percentile(hr_boots, 97.5, axis=0)

    labels = [feature_candidates[f] for f in avail_features]

    directions = []
    for hr in hr_mean:
        if hr > 1.1:
            directions.append("↑ Увеличивает риск")
        elif hr < 0.9:
            directions.append("↓ Снижает риск")
        else:
            directions.append("➡ Нейтрально")

    return pd.DataFrame({
        "Фактор": labels,
        "Hazard Ratio": hr_mean.tolist(),
        "CI_2.5%": hr_lo.tolist(),
        "CI_97.5%": hr_hi.tolist(),
        "Direction": directions,
    })


def plot_cox_forest(cox_df: pd.DataFrame) -> go.Figure:
    """
    Forest plot of Cox PH hazard ratios with 95% bootstrap CI.

    Each factor shown as a horizontal CI line with a diamond point estimate.
    X-axis uses log scale; HR=1 reference line added.
    Lines coloured by direction (red = increases risk, blue = reduces, gray = neutral).
    """
    if cox_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Cox PH — нет данных", template="nba_dark", height=280)
        return fig

    dir_colors = {
        "↑ Увеличивает риск": NBA_RED,
        "↓ Снижает риск": NBA_BLUE,
        "➡ Нейтрально": "#4a566e",
    }

    fig = go.Figure()

    # HR = 1 reference
    fig.add_vline(
        x=1.0,
        line=dict(color="rgba(255,255,255,0.30)", dash="dot", width=1.2),
        annotation_text="HR = 1",
        annotation_position="top",
        annotation_font=dict(color=TEXT_MUTED, size=10),
    )

    factors = cox_df["Фактор"].tolist()
    hr_vals = cox_df["Hazard Ratio"].tolist()
    ci_lo = cox_df["CI_2.5%"].tolist()
    ci_hi = cox_df["CI_97.5%"].tolist()
    dirs = cox_df["Direction"].tolist()

    for i, (factor, hr, lo, hi, direction) in enumerate(
        zip(factors, hr_vals, ci_lo, ci_hi, dirs)
    ):
        color = dir_colors.get(direction, "#4a566e")
        # CI line
        fig.add_trace(go.Scatter(
            x=[lo, hi],
            y=[factor, factor],
            mode="lines",
            line=dict(color=color, width=3),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Diamond point estimate
        fig.add_trace(go.Scatter(
            x=[hr],
            y=[factor],
            mode="markers",
            marker=dict(
                symbol="diamond",
                color=color,
                size=10,
                line=dict(color="white", width=0.8),
            ),
            name=direction,
            showlegend=(i == [d for d in dirs].index(direction)),
            hovertemplate=(
                f"<b>{factor}</b><br>"
                f"HR: {hr:.3f}<br>"
                f"95% CI: [{lo:.3f}, {hi:.3f}]<br>"
                f"{direction}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text="Cox PH — факторы риска карьерного спада",
            font=dict(size=13, color="#e8eaf6"),
        ),
        height=max(280, len(cox_df) * 55 + 80),
        template="nba_dark",
        xaxis=dict(title="Hazard Ratio (log scale)", type="log"),
        yaxis=dict(autorange="reversed"),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        margin=dict(l=140, r=60, t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Survival heatmap
# ---------------------------------------------------------------------------

def plot_survival_heatmap(df_surv: pd.DataFrame) -> go.Figure:
    """
    Heatmap: rows = risk_group (3 tiers), columns = age_group (3 buckets).
    Cell value = median survival time for that subgroup.
    Each cell annotated with the median value and count n.
    """
    age_groups_order = ["Молодой (<25)", "Расцвет (25-30)", "Ветеран (>30)"]
    risk_groups_order = ["🟢 Низкий", "🟡 Умеренный", "🔴 Высокий риск"]

    z_matrix = np.full((3, 3), np.nan)
    annot_matrix: list[list[str]] = [[""] * 3 for _ in range(3)]

    for ri, rg in enumerate(risk_groups_order):
        for ai, ag in enumerate(age_groups_order):
            sub = df_surv[
                (df_surv["risk_group"] == rg) & (df_surv["age_group"] == ag)
            ].dropna(subset=["time"])
            n = len(sub)
            if n == 0:
                annot_matrix[ri][ai] = "n=0"
                z_matrix[ri, ai] = 0.0
                continue
            t_arr, S_arr, _, _ = kaplan_meier(
                sub["time"].values, sub["observed"].values
            )
            below = np.where(S_arr <= 0.5)[0]
            median_t = float(t_arr[below[0]]) if len(below) > 0 else float(t_arr[-1])
            z_matrix[ri, ai] = median_t
            annot_matrix[ri][ai] = f"{median_t:.1f}<br>n={n}"

    # Build annotation text for heatmap
    fig = go.Figure(go.Heatmap(
        z=z_matrix,
        x=age_groups_order,
        y=risk_groups_order,
        colorscale=[[0, "#C8102E"], [0.5, "#1D428A"], [1, "#00D4AA"]],
        colorbar=dict(
            thickness=10,
            len=0.7,
            tickfont=dict(color="#6b7a99", size=10),
            title=dict(
                text="Медиана T",
                font=dict(color="#8b9ab5", size=11),
            ),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.06)",
        ),
        hovertemplate="Риск: %{y}<br>Возраст: %{x}<br>T₅₀: %{z:.2f}<extra></extra>",
    ))

    # Cell annotations
    for ri in range(3):
        for ai in range(3):
            fig.add_annotation(
                x=age_groups_order[ai],
                y=risk_groups_order[ri],
                text=annot_matrix[ri][ai],
                showarrow=False,
                font=dict(color="white", size=11),
                align="center",
            )

    fig.update_layout(
        title=dict(
            text="Медиана выживаемости по возрасту и риску",
            font=dict(size=14, color="#e8eaf6"),
        ),
        height=320,
        template="nba_dark",
        xaxis=dict(title="Возрастная группа"),
        yaxis=dict(title="Группа риска"),
        margin=dict(l=130, r=80, t=60, b=60),
    )
    return fig
