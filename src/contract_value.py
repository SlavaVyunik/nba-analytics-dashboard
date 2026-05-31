"""
Contract Value — сравнение Trade Value с синтетической рыночной зарплатой.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Параметры синтетического рынка ───────────────────────────────────────────
SALARY_BASE_M  = 1.5     # млн $ — минимальная зарплата
SALARY_MAX_M   = 45.0    # млн $ — максимальная зарплата звезды
TV_MAX         = 100.0   # максимум Trade Value

# Множители за позицию (условные)
POS_MULTIPLIER = {"PG": 1.10, "SG": 1.05, "SF": 1.00, "PF": 0.95, "C": 0.92}

CONTRACT_TIERS = [
    (35, "💰 Макс-контракт",  "#8e44ad"),
    (25, "⭐ Всезвёздный",    "#c8102e"),
    (15, "🔵 Стартер",        "#2980b9"),
    ( 7, "🟡 Ролевой",        "#d4ac0d"),
    ( 0, "⚪ Минималка",      "#7f8c8d"),
]


def _salary_tier(salary_m: float):
    for threshold, label, color in CONTRACT_TIERS:
        if salary_m >= threshold:
            return label, color
    return "⚪ Минималка", "#7f8c8d"


def compute_contract_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет колонки:
      market_salary_m   — синтетическая рыночная зарплата (млн $)
      contract_tier     — тир контракта
      contract_color
      value_surplus     — Trade Value - нормализованная зарплата (>0 = недооценён)
      value_label       — "Недооценён" / "Справедливо" / "Переоценён"
    """
    df = df.copy()

    # Trade Value обязателен
    if "trade_value" not in df.columns:
        from src.trade_value import compute_trade_value
        df = compute_trade_value(df)

    tv = df["trade_value"].fillna(df["trade_value"].median())

    # Нелинейная зависимость: зарплата ~ tv^1.4 (суперзвёзды зарабатывают непропорционально)
    tv_norm = (tv / TV_MAX).clip(0, 1)
    salary = SALARY_BASE_M + (SALARY_MAX_M - SALARY_BASE_M) * (tv_norm ** 1.4)

    # Поправка на позицию
    pos_col = next((c for c in ["POS", "POSITION", "position", "_pos"] if c in df.columns), None)
    if pos_col:
        mult = df[pos_col].apply(
            lambda p: POS_MULTIPLIER.get(str(p).upper()[:2], 1.0) if pd.notna(p) else 1.0
        )
        salary = salary * mult

    # Поправка на возраст (молодые ↓, опытный пик ↑, ветераны ↓)
    age_col = next((c for c in ["PLAYER_AGE", "AGE", "age"] if c in df.columns), None)
    if age_col:
        age = df[age_col].fillna(df[age_col].median()).clip(18, 42)
        age_factor = 1 - 0.3 * np.exp(-((age - 28) ** 2) / 50)
        salary = salary * age_factor

    df["market_salary_m"] = salary.round(2)

    tiers = df["market_salary_m"].apply(_salary_tier)
    df["contract_tier"]  = tiers.apply(lambda x: x[0])
    df["contract_color"] = tiers.apply(lambda x: x[1])

    # Профицит: сравниваем trade_value с «ожидаемым» TV для данной зарплаты
    expected_tv = TV_MAX * ((salary - SALARY_BASE_M) / (SALARY_MAX_M - SALARY_BASE_M)).clip(0, 1) ** (1/1.4)
    df["value_surplus"] = (tv - expected_tv).round(1)

    df["value_label"] = df["value_surplus"].apply(
        lambda s: "Недооценён" if s > 8 else ("Переоценён" if s < -8 else "Справедливо")
    )
    return df


# ── Визуализации ─────────────────────────────────────────────────────────────

def plot_salary_leaderboard(df: pd.DataFrame, n: int = 20) -> go.Figure:
    """Топ-N самых высокооплачиваемых игроков."""
    if "market_salary_m" not in df.columns:
        df = compute_contract_value(df)

    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    top = (
        df[[name_col, "market_salary_m", "contract_tier", "contract_color", "trade_value"]]
        .dropna(subset=["market_salary_m"])
        .sort_values("market_salary_m", ascending=False)
        .head(n)
    )

    fig = go.Figure(go.Bar(
        x=top["market_salary_m"],
        y=top[name_col],
        orientation="h",
        marker_color=top["contract_color"],
        text=[f"${v:.1f}M" for v in top["market_salary_m"]],
        textposition="outside",
        customdata=top[["trade_value", "contract_tier"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Зарплата: $%{x:.1f}M<br>"
            "Trade Value: %{customdata[0]:.1f}<br>"
            "%{customdata[1]}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(text=f"Топ-{n} по рыночной стоимости контракта", font_size=15),
        xaxis=dict(title="Зарплата (млн $)", range=[0, SALARY_MAX_M * 1.25]),
        yaxis=dict(autorange="reversed"),
        template="nba_dark",
        height=max(360, n * 22),
        margin=dict(t=50, b=40, l=180, r=30),
        showlegend=False,
    )
    return fig


def plot_value_vs_salary(df: pd.DataFrame) -> go.Figure:
    """Скаттер: Рыночная зарплата (X) vs Trade Value (Y)."""
    if "market_salary_m" not in df.columns:
        df = compute_contract_value(df)

    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    plot_df = df.dropna(subset=["market_salary_m", "trade_value"])

    color_map = {
        "Недооценён":  "#27ae60",
        "Справедливо": "#3498db",
        "Переоценён":  "#e74c3c",
    }

    fig = go.Figure()
    for label, color in color_map.items():
        sub = plot_df[plot_df["value_label"] == label]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["market_salary_m"],
            y=sub["trade_value"],
            mode="markers",
            name=label,
            marker=dict(
                color=color, size=9, opacity=0.75,
                line=dict(color="white", width=0.8),
            ),
            text=sub[name_col],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Зарплата: $%{x:.1f}M<br>"
                "Trade Value: %{y:.1f}<extra></extra>"
            ),
        ))

    # Линия справедливой цены
    sal_range = np.linspace(SALARY_BASE_M, SALARY_MAX_M, 100)
    tv_fair = TV_MAX * ((sal_range - SALARY_BASE_M) / (SALARY_MAX_M - SALARY_BASE_M)).clip(0, 1) ** (1/1.4)
    fig.add_trace(go.Scatter(
        x=sal_range, y=tv_fair,
        mode="lines",
        line=dict(color="#555", width=1.5, dash="dot"),
        name="Справедливая цена",
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=dict(text="Trade Value vs Рыночная зарплата", font_size=15),
        xaxis=dict(title="Зарплата (млн $)"),
        yaxis=dict(title="Trade Value (0-100)"),
        template="nba_dark",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=50, l=60, r=20),
    )
    return fig


def plot_value_surplus(df: pd.DataFrame, n: int = 30) -> go.Figure:
    """Топ недооценённых и переоценённых игроков (горизонтальный бар)."""
    if "value_surplus" not in df.columns:
        df = compute_contract_value(df)

    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    plot_df = (
        df[[name_col, "value_surplus", "value_label"]]
        .dropna(subset=["value_surplus"])
        .sort_values("value_surplus", ascending=False)
    )

    # Топ-15 недооценённых + Топ-15 переоценённых
    under = plot_df.head(min(15, n // 2))
    over  = plot_df.tail(min(15, n // 2))
    plot_df2 = pd.concat([over, under]).sort_values("value_surplus")

    colors = plot_df2["value_label"].map({
        "Недооценён": "#27ae60",
        "Справедливо": "#3498db",
        "Переоценён":  "#e74c3c",
    }).fillna("#3498db")

    fig = go.Figure(go.Bar(
        x=plot_df2["value_surplus"],
        y=plot_df2[name_col],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}" for v in plot_df2["value_surplus"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Профицит: %{x:+.1f}<extra></extra>",
    ))

    fig.add_vline(x=0, line_dash="solid", line_color="#aaa", line_width=1)

    fig.update_layout(
        title=dict(text="Недооценённые vs Переоценённые игроки", font_size=15),
        xaxis=dict(title="Профицит стоимости (↑ недооценён, ↓ переоценён)"),
        yaxis=dict(autorange="reversed"),
        template="nba_dark",
        height=max(400, len(plot_df2) * 22),
        margin=dict(t=50, b=50, l=180, r=40),
        showlegend=False,
    )
    return fig


def plot_contract_tiers_breakdown(df: pd.DataFrame) -> go.Figure:
    """Пай-чарт + бар по тирам контрактов."""
    if "contract_tier" not in df.columns:
        df = compute_contract_value(df)

    tier_counts = df["contract_tier"].value_counts().reset_index()
    tier_counts.columns = ["tier", "count"]

    color_dict = {t[1]: t[2] for t in CONTRACT_TIERS}
    tier_counts["color"] = tier_counts["tier"].map(color_dict).fillna("#aaa")

    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type": "pie"}, {"type": "bar"}]],
                        subplot_titles=["Распределение по тирам", "Количество игроков"])

    fig.add_trace(go.Pie(
        labels=tier_counts["tier"],
        values=tier_counts["count"],
        marker_colors=tier_counts["color"],
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} игроков<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=tier_counts["tier"],
        y=tier_counts["count"],
        marker_color=tier_counts["color"],
        text=tier_counts["count"],
        textposition="outside",
        hovertemplate="%{x}: %{y} игроков<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        title=dict(text="Структура контрактов в лиге", font_size=14),
        template="nba_dark",
        height=380,
        margin=dict(t=60, b=50, l=30, r=20),
    )
    return fig


def get_contract_summary(df: pd.DataFrame, player_name: str) -> dict:
    """Возвращает словарь с контрактной информацией об игроке."""
    if "market_salary_m" not in df.columns:
        df = compute_contract_value(df)

    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    rows = df[df[name_col] == player_name]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    return {
        "market_salary_m": round(float(row.get("market_salary_m", 0)), 2),
        "contract_tier":   row.get("contract_tier", ""),
        "contract_color":  row.get("contract_color", "#aaa"),
        "trade_value":     round(float(row.get("trade_value", 0)), 1),
        "value_surplus":   round(float(row.get("value_surplus", 0)), 1),
        "value_label":     row.get("value_label", "Справедливо"),
    }
