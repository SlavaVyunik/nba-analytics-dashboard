"""
Оптимальный состав команды — подбор 5 лучших игроков по позициям и суммарной эффективности.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations

# ── Позиции ──────────────────────────────────────────────────────────────────
POSITIONS = ["PG", "SG", "SF", "PF", "C"]
POS_RU = {
    "PG": "Разыгрывающий",
    "SG": "Атакующий защитник",
    "SF": "Лёгкий форвард",
    "PF": "Тяжёлый форвард",
    "C":  "Центровой",
}
POS_COLORS = {
    "PG": "#3498db",
    "SG": "#9b59b6",
    "SF": "#27ae60",
    "PF": "#e67e22",
    "C":  "#e74c3c",
}

# Веса для суммарного рейтинга состава
LINEUP_WEIGHTS = {
    "PTS":        0.30,
    "AST":        0.20,
    "REB":        0.15,
    "efficiency": 0.25,
    "trade_value":0.10,
}


def _detect_position(row: pd.Series) -> str:
    """Определяет позицию игрока по доступным данным."""
    pos_col = next((c for c in ["POS", "POSITION", "position", "pos"] if c in row.index), None)
    if pos_col and not pd.isna(row.get(pos_col)):
        raw = str(row[pos_col]).upper().strip()
        for p in POSITIONS:
            if p in raw:
                return p
    # Фолбэк по статистике
    ast = float(row.get("AST", 0) or 0)
    reb = float(row.get("REB", 0) or 0)
    if ast > reb * 1.5:
        return "PG"
    if reb > ast * 2:
        return "C" if reb > 8 else "PF"
    return "SF"


def assign_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет колонку _pos, если её нет."""
    df = df.copy()
    pos_col = next((c for c in ["POS", "POSITION", "position"] if c in df.columns), None)
    if pos_col:
        df["_pos"] = df[pos_col].apply(lambda x: str(x).upper()[:2] if pd.notna(x) else "SF")
        df["_pos"] = df["_pos"].apply(
            lambda p: next((pos for pos in POSITIONS if pos in p), "SF")
        )
    else:
        df["_pos"] = df.apply(_detect_position, axis=1)
    return df


def _player_score(row: pd.Series) -> float:
    """Скалярный рейтинг игрока для формирования состава."""
    score = 0.0
    for stat, w in LINEUP_WEIGHTS.items():
        val = row.get(stat, 0)
        if pd.notna(val):
            score += float(val) * w
    return round(score, 3)


def build_optimal_lineup(
    df: pd.DataFrame,
    strategy: str = "balanced",
    locked: list = None,
) -> dict:
    """
    Строит оптимальный состав из 5 игроков (по одному на позицию).

    Parameters
    ----------
    df        : DataFrame с игроками
    strategy  : "balanced" | "offense" | "defense" | "young"
    locked    : список имён, которых нужно включить принудительно

    Returns
    -------
    dict с ключами: lineup (список dict), total_score, team_stats
    """
    df = assign_positions(df)
    name_col = "name" if "name" in df.columns else "PLAYER_NAME"

    # Стратегия меняет веса
    weights = dict(LINEUP_WEIGHTS)
    if strategy == "offense":
        weights.update({"PTS": 0.40, "AST": 0.25, "REB": 0.10, "efficiency": 0.20, "trade_value": 0.05})
    elif strategy == "defense":
        weights.update({"PTS": 0.15, "AST": 0.10, "REB": 0.30, "efficiency": 0.35, "trade_value": 0.10})
    elif strategy == "young":
        weights.update({"PTS": 0.25, "AST": 0.20, "REB": 0.15, "efficiency": 0.20, "trade_value": 0.20})

    df["_score"] = df.apply(
        lambda r: sum(float(r.get(s, 0) or 0) * w for s, w in weights.items()), axis=1
    )

    age_col = next((c for c in ["PLAYER_AGE", "AGE", "age"] if c in df.columns), None)
    if strategy == "young" and age_col:
        df = df[df[age_col] <= 26].copy()

    locked = locked or []
    lineup = {}
    used_names = set()

    # Сначала прописываем заблокированных
    for name in locked:
        rows = df[df[name_col] == name]
        if rows.empty:
            continue
        row = rows.iloc[0]
        pos = row["_pos"]
        if pos not in lineup:
            lineup[pos] = row
            used_names.add(name)

    # Затем лучших по позициям
    for pos in POSITIONS:
        if pos in lineup:
            continue
        candidates = df[
            (df["_pos"] == pos) & (~df[name_col].isin(used_names))
        ].sort_values("_score", ascending=False)
        if candidates.empty:
            # Разрешим любую позицию
            candidates = df[~df[name_col].isin(used_names)].sort_values("_score", ascending=False)
        if not candidates.empty:
            best = candidates.iloc[0]
            lineup[pos] = best
            used_names.add(best[name_col])

    # Формируем результат
    lineup_list = []
    for pos in POSITIONS:
        if pos not in lineup:
            continue
        row = lineup[pos]
        lineup_list.append({
            "pos":          pos,
            "pos_ru":       POS_RU[pos],
            "name":         row.get(name_col, "?"),
            "score":        round(float(row["_score"]), 2),
            "pts":          round(float(row.get("PTS", 0) or 0), 1),
            "ast":          round(float(row.get("AST", 0) or 0), 1),
            "reb":          round(float(row.get("REB", 0) or 0), 1),
            "efficiency":   round(float(row.get("efficiency", 0) or 0), 1),
            "trade_value":  round(float(row.get("trade_value", 0) or 0), 1),
            "age":          row.get(age_col) if age_col else None,
            "color":        POS_COLORS[pos],
        })

    total = round(sum(p["score"] for p in lineup_list), 2)

    # Командная суммарная статистика
    team_stats = {}
    for stat in ["pts", "ast", "reb", "efficiency"]:
        team_stats[stat] = round(sum(p[stat] for p in lineup_list), 1)
    team_stats["avg_age"] = (
        round(float(np.mean([p["age"] for p in lineup_list if p["age"] is not None])), 1)
        if any(p["age"] is not None for p in lineup_list) else None
    )

    return {"lineup": lineup_list, "total_score": total, "team_stats": team_stats}


# ── Визуализации ──────────────────────────────────────────────────────────────

def plot_lineup_court(lineup_result: dict) -> go.Figure:
    """Рисует игроков на схеме баскетбольной площадки."""
    lineup = lineup_result.get("lineup", [])

    # Позиции на площадке (условные координаты 0-10)
    pos_coords = {
        "PG": (5.0, 1.5),
        "SG": (2.0, 3.5),
        "SF": (8.0, 3.5),
        "PF": (2.5, 7.0),
        "C":  (7.5, 7.0),
    }

    fig = go.Figure()

    # Фон площадки
    fig.add_shape(type="rect", x0=0, y0=0, x1=10, y1=9.4,
                  fillcolor="#f5deb3", line=dict(color="#c8a870", width=2), layer="below")
    # Трёхочковая (полукруг)
    theta = np.linspace(0, np.pi, 100)
    fig.add_trace(go.Scatter(
        x=5 + 4.2 * np.cos(theta), y=0.5 + 4.2 * np.sin(theta),
        mode="lines", line=dict(color="#8b7355", width=1.5),
        showlegend=False, hoverinfo="skip"))
    # Краска
    fig.add_shape(type="rect", x0=3.2, y0=0, x1=6.8, y1=5.8,
                  fillcolor="rgba(180,140,80,0.3)", line=dict(color="#8b7355", width=1.5), layer="below")
    # Штрафной круг
    theta2 = np.linspace(0, 2*np.pi, 80)
    fig.add_trace(go.Scatter(
        x=5 + 1.0 * np.cos(theta2), y=5.8 + 1.0 * np.sin(theta2),
        mode="lines", line=dict(color="#8b7355", width=1.2),
        showlegend=False, hoverinfo="skip"))
    # Центральный круг (центр зоны)
    fig.add_shape(type="circle", x0=4.4, y0=5.2, x1=5.6, y1=6.4,
                  fillcolor="rgba(180,140,80,0.4)", line=dict(color="#8b7355", width=1))

    # Игроки
    for player in lineup:
        pos = player["pos"]
        if pos not in pos_coords:
            continue
        x, y = pos_coords[pos]
        color = player["color"]
        name  = player["name"]
        pos_ru = player["pos_ru"]

        # Круг игрока
        fig.add_shape(type="circle",
                      x0=x-0.55, y0=y-0.55, x1=x+0.55, y1=y+0.55,
                      fillcolor=color, line=dict(color="white", width=2))

        # Аббревиатура позиции внутри
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="text",
            text=[pos],
            textfont=dict(size=12, color="white", family="Arial Black"),
            showlegend=False, hoverinfo="skip"))

        # Подпись снизу
        short_name = name.split()[-1] if " " in name else name
        fig.add_trace(go.Scatter(
            x=[x], y=[y - 0.9],
            mode="text",
            text=[f"<b>{short_name}</b>"],
            textfont=dict(size=9, color="#333"),
            showlegend=False,
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"{pos_ru}<br>"
                f"ОЧК: {player['pts']} | ПАС: {player['ast']} | ПОД: {player['reb']}<br>"
                f"Рейтинг: {player['score']}<extra></extra>"
            ),
        ))

    fig.update_layout(
        xaxis=dict(range=[-0.3, 10.3], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.5, 10.5], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x", scaleratio=1),
        template="nba_dark",
        height=460,
        margin=dict(t=20, b=20, l=20, r=20),
        plot_bgcolor="#f5deb3",
        paper_bgcolor="#0f1520",
        showlegend=False,
    )
    return fig


def plot_lineup_radar(lineup_result: dict) -> go.Figure:
    """Радарная диаграмма суммарных характеристик состава."""
    team_stats = lineup_result.get("team_stats", {})
    lineup     = lineup_result.get("lineup", [])

    cats   = ["Очки", "Передачи", "Подборы", "Эффективность"]
    keys   = ["pts",  "ast",      "reb",     "efficiency"]
    values = [team_stats.get(k, 0) for k in keys]

    # Нормализуем на 1 игрока
    if lineup:
        values = [round(v / len(lineup), 1) for v in values]

    values_closed = values + [values[0]]
    cats_closed   = cats + [cats[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=cats_closed,
        fill="toself",
        fillcolor="rgba(200,16,46,0.15)",
        line=dict(color="#c8102e", width=2),
        name="Состав",
        hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="#0c1220",
            radialaxis=dict(
                visible=True, showticklabels=True,
                gridcolor="rgba(255,255,255,0.10)",
                linecolor="rgba(255,255,255,0.08)",
                tickfont=dict(color="#6b7a99", size=9),
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.10)",
                linecolor="rgba(255,255,255,0.12)",
                tickfont=dict(size=12, color="#8b9ab5"),
            ),
        ),
        template="nba_dark",
        height=360,
        margin=dict(t=40, b=40, l=40, r=40),
        showlegend=False,
    )
    return fig


def plot_lineup_bars(lineup_result: dict) -> go.Figure:
    """Сравнение игроков состава по ключевым статам."""
    lineup = lineup_result.get("lineup", [])
    if not lineup:
        return go.Figure().update_layout(title="Нет данных")

    names  = [p["name"].split()[-1] for p in lineup]
    colors = [p["color"] for p in lineup]
    stats  = ["pts", "ast", "reb", "efficiency"]
    labels = {"pts": "Очки", "ast": "Передачи", "reb": "Подборы", "efficiency": "Эффективность"}

    fig = make_subplots(rows=1, cols=4,
                        subplot_titles=[labels[s] for s in stats],
                        horizontal_spacing=0.08)

    for i, stat in enumerate(stats):
        vals = [p[stat] for p in lineup]
        fig.add_trace(go.Bar(
            x=names, y=vals,
            marker_color=colors,
            showlegend=False,
            hovertemplate=f"%{{x}}: %{{y:.1f}}<extra></extra>",
        ), row=1, col=i+1)

    fig.update_layout(
        title=dict(text="Статистика игроков состава", font_size=14),
        template="nba_dark",
        height=320,
        margin=dict(t=50, b=50, l=40, r=20),
    )
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=11, color="#8b9ab5"))
    return fig
