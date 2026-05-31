"""
Риск травмы игрока — составной индекс на основе возраста, нагрузки, эффективности.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Веса компонентов ────────────────────────────────────────────────────────
W_AGE        = 0.30
W_WORKLOAD   = 0.25
W_EFFICIENCY = 0.20
W_BODY       = 0.15
W_RECOVERY   = 0.10

RISK_TIERS = [
    (80, "🔴 Высокий",    "#e74c3c"),
    (60, "🟠 Повышенный", "#e67e22"),
    (40, "🟡 Умеренный",  "#f1c40f"),
    (20, "🟢 Низкий",     "#27ae60"),
    ( 0, "⚪ Минимальный","#95a5a6"),
]


def _tier(score):
    for threshold, label, color in RISK_TIERS:
        if score >= threshold:
            return label, color
    return "⚪ Минимальный", "#95a5a6"


def _safe_minmax(series, invert=False):
    """Min-max normalize series to [0, 100]; invert=True → higher raw = lower risk."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    norm = (series - mn) / (mx - mn) * 100
    return 100 - norm if invert else norm


def compute_injury_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет в DataFrame колонки:
      risk_age, risk_workload, risk_efficiency,
      risk_body, risk_recovery, injury_risk (0-100), risk_tier, risk_color
    """
    df = df.copy()

    # ── 1. Возрастной риск (выше у 30+) ────────────────────────────────────
    age_col = next((c for c in ["PLAYER_AGE", "AGE", "age"] if c in df.columns), None)
    if age_col:
        age = df[age_col].fillna(df[age_col].median())
        # Линейно растёт после 28: 0% в 18 лет, 100% в 40 лет
        df["risk_age"] = np.clip((age - 18) / (40 - 18) * 100, 0, 100)
    else:
        df["risk_age"] = 50.0

    # ── 2. Нагрузочный риск (много минут → выше риск) ──────────────────────
    min_col = next((c for c in ["MIN", "MP", "minutes"] if c in df.columns), None)
    gp_col  = next((c for c in ["GP", "G", "games"] if c in df.columns), None)
    if min_col and gp_col:
        total_min = df[min_col].fillna(0) * df[gp_col].fillna(0)
        df["risk_workload"] = _safe_minmax(total_min)
    elif min_col:
        df["risk_workload"] = _safe_minmax(df[min_col].fillna(df[min_col].median()))
    else:
        df["risk_workload"] = 50.0

    # ── 3. Риск по снижению эффективности (упала = усталость/риск) ─────────
    eff_col = next((c for c in ["efficiency", "EFF", "PER"] if c in df.columns), None)
    if eff_col:
        # Низкая эффективность → высокий риск
        df["risk_efficiency"] = _safe_minmax(
            df[eff_col].fillna(df[eff_col].median()), invert=True
        )
    else:
        df["risk_efficiency"] = 50.0

    # ── 4. Физический стресс (комбо: высокие очки + мало игр → перегруз) ──
    pts_col = next((c for c in ["PTS", "points"] if c in df.columns), None)
    if pts_col and gp_col:
        ppg = df[pts_col].fillna(0) / df[gp_col].fillna(1).replace(0, 1)
        df["risk_body"] = _safe_minmax(ppg)
    elif pts_col:
        df["risk_body"] = _safe_minmax(df[pts_col].fillna(0))
    else:
        df["risk_body"] = 50.0

    # ── 5. Восстановительный риск (возраст × нагрузка) ────────────────────
    df["risk_recovery"] = np.clip(
        (df["risk_age"] * 0.6 + df["risk_workload"] * 0.4), 0, 100
    )

    # ── Итоговый индекс ─────────────────────────────────────────────────────
    df["injury_risk"] = (
        W_AGE        * df["risk_age"]        +
        W_WORKLOAD   * df["risk_workload"]   +
        W_EFFICIENCY * df["risk_efficiency"] +
        W_BODY       * df["risk_body"]       +
        W_RECOVERY   * df["risk_recovery"]
    ).round(1)

    tiers = df["injury_risk"].apply(_tier)
    df["risk_tier"]  = tiers.apply(lambda x: x[0])
    df["risk_color"] = tiers.apply(lambda x: x[1])

    return df


# ── Визуализации ─────────────────────────────────────────────────────────────

def plot_risk_breakdown(row: pd.Series) -> go.Figure:
    """Горизонтальная гистограмма компонентов риска для одного игрока."""
    components = {
        "Возраст (30%)":        row.get("risk_age",        50),
        "Нагрузка (25%)":       row.get("risk_workload",   50),
        "Эффективность (20%)":  row.get("risk_efficiency", 50),
        "Физический стресс (15%)": row.get("risk_body",    50),
        "Восстановление (10%)": row.get("risk_recovery",   50),
    }
    names  = list(components.keys())
    values = [float(v) for v in components.values()]

    colors = []
    for v in values:
        if v >= 80:   colors.append("#e74c3c")
        elif v >= 60: colors.append("#e67e22")
        elif v >= 40: colors.append("#f1c40f")
        elif v >= 20: colors.append("#27ae60")
        else:         colors.append("#95a5a6")

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}" for v in values],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1f}<extra></extra>",
    ))

    overall = float(row.get("injury_risk", 50))
    tier_label = row.get("risk_tier", "")

    fig.update_layout(
        title=dict(
            text=f"Компоненты риска — <b>{row.get('name', row.get('PLAYER_NAME', ''))}</b>"
                 f"<br><sup>Итого: {overall:.0f} / 100 &nbsp;{tier_label}</sup>",
            font_size=14,
        ),
        xaxis=dict(title="Уровень риска (0-100)", range=[0, 115]),
        yaxis=dict(autorange="reversed"),
        template="nba_dark",
        height=320,
        margin=dict(t=70, b=40, l=190, r=30),
        showlegend=False,
    )
    return fig


def plot_risk_leaderboard(df: pd.DataFrame, n: int = 20) -> go.Figure:
    """Топ-N игроков с наибольшим риском травмы."""
    if "injury_risk" not in df.columns:
        df = compute_injury_risk(df)

    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    top = (
        df[[name_col, "injury_risk", "risk_tier", "risk_color"]]
        .dropna(subset=["injury_risk"])
        .sort_values("injury_risk", ascending=False)
        .head(n)
    )

    fig = go.Figure(go.Bar(
        x=top["injury_risk"],
        y=top[name_col],
        orientation="h",
        marker_color=top["risk_color"],
        text=top["risk_tier"],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Риск: %{x:.1f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"Топ-{n} игроков по риску травмы", font_size=15),
        xaxis=dict(title="Индекс риска (0-100)", range=[0, 115]),
        yaxis=dict(autorange="reversed"),
        template="nba_dark",
        height=max(350, n * 22),
        margin=dict(t=50, b=40, l=180, r=30),
        showlegend=False,
    )
    return fig


def plot_risk_vs_value(df: pd.DataFrame) -> go.Figure:
    """Скаттер: Trade Value (X) vs Риск травмы (Y), размер = очки/игру."""
    if "injury_risk" not in df.columns:
        df = compute_injury_risk(df)

    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    has_tv   = "trade_value" in df.columns
    pts_col  = next((c for c in ["PTS", "points"] if c in df.columns), None)
    gp_col   = next((c for c in ["GP", "G", "games"] if c in df.columns), None)

    plot_df = df.copy()
    if pts_col and gp_col:
        plot_df["_ppg"] = plot_df[pts_col].fillna(0) / plot_df[gp_col].fillna(1).replace(0, 1)
    else:
        plot_df["_ppg"] = 10

    x_col = "trade_value" if has_tv else (pts_col or "injury_risk")
    x_label = "Trade Value" if has_tv else "Очки"

    plot_df = plot_df.dropna(subset=[x_col, "injury_risk"])

    # Цвет по уровню риска
    color_map = {
        "🔴 Высокий":    "#e74c3c",
        "🟠 Повышенный": "#e67e22",
        "🟡 Умеренный":  "#f1c40f",
        "🟢 Низкий":     "#27ae60",
        "⚪ Минимальный":"#95a5a6",
    }

    fig = go.Figure()
    for tier_label, color in color_map.items():
        sub = plot_df[plot_df["risk_tier"] == tier_label]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub[x_col],
            y=sub["injury_risk"],
            mode="markers",
            name=tier_label,
            marker=dict(
                color=color,
                size=np.clip(sub["_ppg"] * 2.5, 6, 28),
                opacity=0.75,
                line=dict(color="white", width=0.8),
            ),
            text=sub[name_col],
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"{x_label}: %{{x:.1f}}<br>"
                "Риск: %{y:.1f}<extra></extra>"
            ),
        ))

    # Зоны опасности
    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(231,76,60,0.06)",  line_width=0)
    fig.add_hrect(y0=60, y1=80,  fillcolor="rgba(230,126,34,0.06)", line_width=0)

    fig.update_layout(
        title=dict(text=f"Риск травмы vs {x_label}", font_size=15),
        xaxis=dict(title=x_label),
        yaxis=dict(title="Индекс риска (0-100)", range=[0, 105]),
        template="nba_dark",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=50, l=60, r=20),
    )
    return fig


def plot_risk_distribution(df: pd.DataFrame) -> go.Figure:
    """Распределение индекса риска по лиге (гистограмма + KDE)."""
    if "injury_risk" not in df.columns:
        df = compute_injury_risk(df)

    scores = df["injury_risk"].dropna()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=25,
        marker_color="rgba(52,152,219,0.6)",
        marker_line=dict(color="rgba(52,152,219,1)", width=1),
        name="Игроки",
        hovertemplate="Риск %{x:.0f}: %{y} игроков<extra></extra>",
    ))

    # Зоны
    for x0, x1, color, label in [
        (0,  20,  "rgba(149,165,166,0.12)", "Минимальный"),
        (20, 40,  "rgba(39,174,96,0.12)",   "Низкий"),
        (40, 60,  "rgba(241,196,15,0.12)",  "Умеренный"),
        (60, 80,  "rgba(230,126,34,0.12)",  "Повышенный"),
        (80, 100, "rgba(231,76,60,0.12)",   "Высокий"),
    ]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=color, line_width=0,
                      annotation_text=label, annotation_position="top left",
                      annotation_font_size=9, annotation_font_color="#888")

    fig.update_layout(
        title=dict(text="Распределение риска травмы по лиге", font_size=14),
        xaxis=dict(title="Индекс риска (0-100)"),
        yaxis=dict(title="Количество игроков"),
        template="nba_dark",
        height=360,
        margin=dict(t=50, b=50, l=60, r=20),
        bargap=0.05,
    )
    return fig
