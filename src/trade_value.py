"""
Trade Value Score — интегральная оценка трансферной стоимости игрока.

Формула учитывает:
  • Результативность (PTS, AST, REB взвешенно)
  • Защиту (STL, BLK)
  • Эффективность (TS%, USG%, PLUS_MINUS)
  • Возрастной коэффициент (пик ~ 26-29 лет → 1.0, молодые ~ 0.9-1.0+, ветераны ↓)
  • «Ценность» контракта (прокси через оставшиеся пиковые годы)
  • Надёжность (игры, низкие потери)

Итоговый Trade Value: 0 – 100 (>= 85 — франчайз-игрок, 70-84 — топ-15, и т.д.)
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler


# Веса компонентов (сумма = 1.0)
WEIGHTS = {
    "scoring":    0.28,   # очки + доля создания
    "playmaking": 0.14,   # передачи
    "rebounding": 0.10,   # подборы
    "defense":    0.12,   # stl + blk + plus_minus
    "efficiency": 0.18,   # ts%, usg% (умеренный)
    "age_prime":  0.12,   # возрастной бонус
    "reliability":0.06,   # игры, потери
}

AGE_COL_OPTIONS = ["PLAYER_AGE", "AGE", "age"]


def _age_value_factor(age):
    """
    Возрастная ценность: максимум (~1.0) около 25-30 лет.
    Молодёжь получает «потенциальный» бонус, ветераны — штраф.
    """
    if pd.isna(age):
        return 0.70
    age = float(age)
    if age <= 19:  return 0.55   # очень молодой — непредсказуем
    if age <= 22:  return 0.75   # перспектива, но ещё сырой
    if age <= 25:  return 0.88   # растёт
    if age <= 29:  return 1.00   # пик
    if age <= 31:  return 0.93
    if age <= 33:  return 0.82
    if age <= 35:  return 0.68
    return                0.50   # закат карьеры


def _prime_years_remaining(age):
    """Примерное количество пиковых лет до 32."""
    if pd.isna(age): return 3
    age = float(age)
    return max(0, min(10, 32 - age))


def compute_trade_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитать Trade Value для каждого игрока.
    Возвращает df с колонками:
      tv_scoring, tv_playmaking, tv_rebounding, tv_defense,
      tv_efficiency, tv_age_prime, tv_reliability,
      trade_value  (0-100), tv_tier
    """
    df = df.copy()
    scaler = MinMaxScaler(feature_range=(0, 100))

    age_col = next((c for c in AGE_COL_OPTIONS if c in df.columns), None)

    # ── Компоненты ────────────────────────────────────────────────────────────

    # 1. Scoring (PTS + scoring_creation proxy)
    pts   = df.get("PTS", pd.Series(0, index=df.index)).fillna(0)
    ast   = df.get("AST", pd.Series(0, index=df.index)).fillna(0)
    scoring_raw = pts * 1.0 + ast * 0.4
    df["tv_scoring"] = scaler.fit_transform(scoring_raw.values.reshape(-1,1)).flatten()

    # 2. Playmaking (AST − TOV * 0.5)
    tov   = df.get("TOV", pd.Series(0, index=df.index)).fillna(0)
    pm_raw = ast - tov * 0.5
    df["tv_playmaking"] = scaler.fit_transform(pm_raw.values.reshape(-1,1)).flatten()

    # 3. Rebounding
    reb   = df.get("REB", pd.Series(0, index=df.index)).fillna(0)
    df["tv_rebounding"] = scaler.fit_transform(reb.values.reshape(-1,1)).flatten()

    # 4. Defense (STL + BLK*1.2 + PLUS_MINUS*0.3)
    stl   = df.get("STL",        pd.Series(0, index=df.index)).fillna(0)
    blk   = df.get("BLK",        pd.Series(0, index=df.index)).fillna(0)
    pm    = df.get("PLUS_MINUS", pd.Series(0, index=df.index)).fillna(0)
    def_raw = stl + blk * 1.2 + pm.clip(-10, 10) * 0.3
    df["tv_defense"] = scaler.fit_transform(def_raw.values.reshape(-1,1)).flatten()

    # 5. Efficiency (TS% * 50 + USG%_moderate)
    ts    = df.get("TS_PCT", pd.Series(0.50, index=df.index)).fillna(0.50)
    usg   = df.get("USG_PCT", pd.Series(0.20, index=df.index)).fillna(0.20)
    # Штрафуем слишком низкое и слишком высокое использование — ценны «умные» игроки
    usg_bonus = 100 - (usg * 100 - 22).abs() * 1.5   # пик у 22% USG
    eff_raw = ts * 100 * 0.7 + usg_bonus * 0.3
    df["tv_efficiency"] = scaler.fit_transform(eff_raw.values.reshape(-1,1)).flatten()

    # 6. Age / Prime value
    if age_col:
        age_vals = df[age_col].values
        prime_raw = np.array([
            _age_value_factor(a) * 50 + _prime_years_remaining(a) * 3
            for a in age_vals
        ])
    else:
        prime_raw = np.full(len(df), 50.0)
    df["tv_age_prime"] = scaler.fit_transform(prime_raw.reshape(-1,1)).flatten()

    # 7. Reliability (games_played% + low_tov bonus)
    gp    = df.get("games_played", pd.Series(60, index=df.index)).fillna(60)
    max_gp = max(82, gp.max())
    gp_pct = (gp / max_gp * 100).clip(0, 100)
    tov_penalty = (tov * 5).clip(0, 30)
    rel_raw = gp_pct - tov_penalty
    df["tv_reliability"] = scaler.fit_transform(rel_raw.values.reshape(-1,1)).flatten()

    # ── Итог взвешенный ──────────────────────────────────────────────────────
    tv = (
        df["tv_scoring"]     * WEIGHTS["scoring"]     +
        df["tv_playmaking"]  * WEIGHTS["playmaking"]  +
        df["tv_rebounding"]  * WEIGHTS["rebounding"]  +
        df["tv_defense"]     * WEIGHTS["defense"]     +
        df["tv_efficiency"]  * WEIGHTS["efficiency"]  +
        df["tv_age_prime"]   * WEIGHTS["age_prime"]   +
        df["tv_reliability"] * WEIGHTS["reliability"]
    )
    df["trade_value"] = tv.round(1)

    # Тиры
    def tier(v):
        if v >= 82: return "💎 Франчайз"
        if v >= 68: return "⭐ Топ-15"
        if v >= 54: return "🔵 Стартер"
        if v >= 38: return "🟡 Ролевой"
        return             "⚪ Резерв"

    df["tv_tier"] = df["trade_value"].apply(tier)

    return df


# ── Визуализации ──────────────────────────────────────────────────────────────

def plot_trade_leaderboard(df, top_n=25):
    """Горизонтальный bar chart топ-N по Trade Value."""
    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    if "trade_value" not in df.columns:
        df = compute_trade_value(df)

    top = df.nlargest(top_n, "trade_value").sort_values("trade_value")

    color_map = {
        "💎 Франчайз": "#1d428a",
        "⭐ Топ-15":   "#c8102e",
        "🔵 Стартер":  "#2980b9",
        "🟡 Ролевой":  "#f39c12",
        "⚪ Резерв":   "#95a5a6",
    }
    colors = top["tv_tier"].map(color_map).fillna("#95a5a6")

    fig = go.Figure(go.Bar(
        x=top["trade_value"],
        y=top[name_col],
        orientation="h",
        marker_color=colors,
        text=top["trade_value"].map("{:.1f}".format),
        textposition="outside",
        customdata=top[["tv_tier", "trade_value"]].values,
        hovertemplate="<b>%{y}</b><br>Trade Value: %{x:.1f}<br>Tier: %{customdata[0]}<extra></extra>",
    ))

    # Границы тиров
    tier_lines = [("💎 Франчайз", 82, "#1d428a"), ("⭐ Топ-15", 68, "#c8102e"),
                  ("🔵 Стартер", 54, "#2980b9"), ("🟡 Ролевой", 38, "#f39c12")]
    for name, val, color in tier_lines:
        if val <= top["trade_value"].max() + 5:
            fig.add_vline(x=val, line_dash="dot", line_color=color,
                          annotation_text=name, annotation_position="top right",
                          annotation_font_size=10)

    fig.update_layout(
        title="Trade Value — Топ игроков лиги",
        xaxis_title="Trade Value Score (0–100)",
        template="nba_dark",
        height=max(420, top_n * 22),
        margin=dict(l=160, r=60, t=60, b=40),
        xaxis=dict(range=[0, 105]),
    )
    return fig


def plot_trade_breakdown(row):
    """Radar/bar chart разбивки компонентов Trade Value для одного игрока."""
    components = {
        "⚡ Результ.":  float(row.get("tv_scoring",     50) or 50),
        "🎯 Пасы":      float(row.get("tv_playmaking",  50) or 50),
        "💪 Подборы":   float(row.get("tv_rebounding",  50) or 50),
        "🛡 Защита":    float(row.get("tv_defense",     50) or 50),
        "📊 Эффект.":   float(row.get("tv_efficiency",  50) or 50),
        "🎂 Возраст":   float(row.get("tv_age_prime",   50) or 50),
        "✅ Надёжн.":   float(row.get("tv_reliability", 50) or 50),
    }

    labels = list(components.keys())
    values = list(components.values())

    bar_colors = [
        "#1d428a" if v >= 70 else "#2980b9" if v >= 50 else
        "#f39c12" if v >= 30 else "#e74c3c"
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=bar_colors,
        text=[f"{v:.0f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="Разбивка Trade Value по компонентам", font_size=14),
        yaxis=dict(range=[0, 110], title="Балл (0–100)"),
        template="nba_dark",
        height=350,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    # Средняя линия
    fig.add_hline(y=50, line_dash="dot", line_color="#999",
                  annotation_text="Средний по лиге", annotation_position="right")
    return fig


def plot_value_vs_age(df):
    """Scatter: возраст vs Trade Value, цвет — кластер."""
    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    if "trade_value" not in df.columns:
        df = compute_trade_value(df)

    age_col = next((c for c in AGE_COL_OPTIONS if c in df.columns), None)
    if not age_col:
        return None

    fig = px.scatter(
        df, x=age_col, y="trade_value",
        color="cluster_name", hover_name=name_col,
        size="PTS", size_max=22,
        labels={age_col: "Возраст", "trade_value": "Trade Value",
                "cluster_name": "Тип игрока"},
        title="Trade Value vs Возраст",
        template="nba_dark",
        height=420,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    # Пик-зона
    fig.add_vrect(x0=25, x1=30, fillcolor="rgba(29,66,138,0.07)",
                  line_width=0, annotation_text="Пик (25-30)",
                  annotation_position="top left")
    return fig


def plot_trade_tiers(df):
    """Pie chart распределения по тирам."""
    if "trade_value" not in df.columns:
        df = compute_trade_value(df)

    tier_counts = df["tv_tier"].value_counts().reset_index()
    tier_counts.columns = ["Тир", "Игроков"]

    color_map = {
        "💎 Франчайз": "#1d428a",
        "⭐ Топ-15":   "#c8102e",
        "🔵 Стартер":  "#2980b9",
        "🟡 Ролевой":  "#f39c12",
        "⚪ Резерв":   "#bdc3c7",
    }
    colors = [color_map.get(t, "#999") for t in tier_counts["Тир"]]

    fig = go.Figure(go.Pie(
        labels=tier_counts["Тир"],
        values=tier_counts["Игроков"],
        marker_colors=colors,
        hole=0.45,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} игроков<extra></extra>",
    ))
    fig.update_layout(
        title="Распределение по Trade Value тирам",
        template="nba_dark",
        height=380,
        showlegend=False,
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


if __name__ == "__main__":
    df = pd.read_csv("data/processed/players_clustered.csv")
    df = compute_trade_value(df)
    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    print("Топ-10 по Trade Value:")
    cols = [name_col, "trade_value", "tv_tier", "PTS", "AST"]
    cols = [c for c in cols if c in df.columns]
    print(df.nlargest(10, "trade_value")[cols].to_string(index=False))
    print("\nРаспределение тиров:")
    print(df["tv_tier"].value_counts())
