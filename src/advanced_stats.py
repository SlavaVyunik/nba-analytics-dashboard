"""
Продвинутая аналитика:
  - Win Shares / VORP
  - On/Off Split
  - Momentum Score
  - Player Similarity (косинусное сходство)
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ── Константы ─────────────────────────────────────────────────────────────────
REPL_LEVEL_BPM  = -2.0   # уровень замены для VORP
SEASON_GAMES    = 82
PACE_AVG        = 100.0

# Признаки для Similarity
SIM_FEATURES = [
    "PTS", "AST", "REB", "STL", "BLK",
    "FG_PCT", "FG3_PCT", "USG_PCT", "TS_PCT",
    "PLUS_MINUS", "efficiency",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. WIN SHARES / VORP / BPM
# ═══════════════════════════════════════════════════════════════════════════════

def _approx_bpm(row: pd.Series, lg_avg_pts: float) -> float:
    """
    Приближённый Box Plus/Minus.
    Формула: линейная комбинация статы на игру, центрированная по лиге.
    """
    pts = float(row.get("PTS", 0) or 0)
    ast = float(row.get("AST", 0) or 0)
    reb = float(row.get("REB", 0) or 0)
    stl = float(row.get("STL", 0) or 0)
    blk = float(row.get("BLK", 0) or 0)
    tov = float(row.get("TOV", 0) or 0)
    fga = float(row.get("FGA", 0) or 0)
    fgp = float(row.get("FG_PCT", 0) or 0)
    usg = float(row.get("USG_PCT", 0) or 0)
    net = float(row.get("NET_RATING", 0) or 0)
    pie = float(row.get("PIE", 0) or 0)

    # Компонент из коробочной статистики
    box = (
        0.22 * pts
        + 0.14 * ast
        + 0.10 * reb
        + 0.18 * stl
        + 0.16 * blk
        - 0.18 * tov
        - 0.06 * fga
        + 0.12 * fgp * 10
    ) - lg_avg_pts * 0.05

    # Компонент из NET_RATING / PIE (реальные продвинутые метрики)
    advanced = 0.35 * net + 30 * (pie - 0.1)

    return round(float(0.55 * box + 0.45 * advanced), 3)


def compute_win_shares(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет:  bpm, vorp, ws_off, ws_def, win_shares, ws_per_48
    """
    df = df.copy()
    gp_col  = "games_played" if "games_played" in df.columns else None
    min_col = "MIN"  if "MIN"  in df.columns else None

    lg_avg_pts = float(df["PTS"].median()) if "PTS" in df.columns else 15.0

    # BPM
    df["bpm"] = df.apply(lambda r: _approx_bpm(r, lg_avg_pts), axis=1)

    # Минуты за сезон
    if gp_col and min_col:
        season_min = df[min_col].fillna(0) * df[gp_col].fillna(0)
    elif min_col:
        season_min = df[min_col].fillna(0) * SEASON_GAMES
    else:
        season_min = pd.Series(1500, index=df.index)

    # VORP = (BPM - replacement_level) / 100 * min_season / 48
    df["vorp"] = ((df["bpm"] - REPL_LEVEL_BPM) / 100 * season_min / 48).round(2)

    # Win Shares ≈ VORP / 2.7 * (tempo-adjusted factor)
    pace = df["PACE"].fillna(PACE_AVG) if "PACE" in df.columns else PACE_AVG
    pace_factor = pace / PACE_AVG if isinstance(pace, pd.Series) else 1.0
    df["win_shares"] = (df["vorp"] / 2.7 * pace_factor).clip(lower=0).round(2)

    # Offensive / Defensive split
    off_frac = (
        (df["PTS"].fillna(0) + df["AST"].fillna(0) * 0.5) /
        (df["PTS"].fillna(0) + df["AST"].fillna(0) * 0.5 +
         df["STL"].fillna(0) * 1.5 + df["BLK"].fillna(0) + df["REB"].fillna(0) * 0.3 + 1)
    ).clip(0.2, 0.8)
    df["ws_off"] = (df["win_shares"] * off_frac).round(2)
    df["ws_def"] = (df["win_shares"] * (1 - off_frac)).round(2)

    # WS/48
    safe_min = season_min.replace(0, np.nan)
    df["ws_per_48"] = (df["win_shares"] / (safe_min / 48)).fillna(0).clip(-0.3, 0.5).round(4)

    return df


# ── Визуализации Win Shares ───────────────────────────────────────────────────

def plot_ws_leaderboard(df: pd.DataFrame, n: int = 25) -> go.Figure:
    """Горизонтальный бар: Топ-N по Win Shares (Off/Def split)."""
    if "win_shares" not in df.columns:
        df = compute_win_shares(df)

    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    top = (
        df[[name_col, "ws_off", "ws_def", "win_shares", "bpm", "vorp"]]
        .dropna(subset=["win_shares"])
        .sort_values("win_shares", ascending=False)
        .head(n)
        .sort_values("win_shares", ascending=True)
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top["ws_off"], y=top[name_col], orientation="h",
        name="Атака", marker_color="#3498db",
        hovertemplate="<b>%{y}</b><br>WS Off: %{x:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=top["ws_def"], y=top[name_col], orientation="h",
        name="Защита", marker_color="#e74c3c",
        hovertemplate="<b>%{y}</b><br>WS Def: %{x:.2f}<extra></extra>",
    ))

    fig.update_layout(
        barmode="stack",
        title=dict(text=f"Топ-{n} по Win Shares (Attack + Defense)", font_size=15),
        xaxis=dict(title="Win Shares за сезон"),
        yaxis=dict(autorange="reversed" if False else True),
        template="nba_dark",
        height=max(380, n * 22),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=50, b=40, l=180, r=30),
    )
    return fig


def plot_vorp_chart(df: pd.DataFrame, n: int = 25) -> go.Figure:
    """VORP + BPM scatter."""
    if "vorp" not in df.columns:
        df = compute_win_shares(df)

    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    plot_df = df[[name_col, "vorp", "bpm", "win_shares"]].dropna().sort_values("vorp", ascending=False).head(n)

    colors = plot_df["bpm"].apply(
        lambda v: "#27ae60" if v > 3 else "#2980b9" if v > 0 else "#e74c3c"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["bpm"],
        y=plot_df["vorp"],
        mode="markers+text",
        marker=dict(
            color=plot_df["bpm"],
            colorscale="RdYlGn",
            size=np.clip(plot_df["win_shares"] * 5 + 8, 8, 28),
            colorbar=dict(title="BPM"),
            line=dict(color="white", width=0.8),
        ),
        text=plot_df[name_col].apply(lambda x: x.split()[-1]),
        textfont=dict(size=8),
        textposition="top center",
        customdata=plot_df[[name_col, "win_shares"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "BPM: %{x:.2f}<br>"
            "VORP: %{y:.2f}<br>"
            "Win Shares: %{customdata[1]:.2f}<extra></extra>"
        ),
    ))

    # Зоны
    fig.add_hrect(y0=0, y1=100, fillcolor="rgba(39,174,96,0.04)", line_width=0)
    fig.add_vline(x=0, line_dash="dot", line_color="#aaa", line_width=1,
                  annotation_text="Ср. уровень", annotation_font_size=9)
    fig.add_vline(x=REPL_LEVEL_BPM, line_dash="dash", line_color="#e74c3c", line_width=1,
                  annotation_text="Уровень замены", annotation_font_size=9, annotation_font_color="#e74c3c")

    fig.update_layout(
        title=dict(text="VORP vs BPM — ценность над уровнем замены", font_size=15),
        xaxis=dict(title="Box Plus/Minus (BPM)"),
        yaxis=dict(title="Value Over Replacement Player (VORP)"),
        template="nba_dark",
        height=500,
        margin=dict(t=50, b=50, l=60, r=20),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ON/OFF SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

def compute_on_off(df: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитывает on_net, off_net, on_off_diff для каждого игрока.
    On-court NET_RATING берётся из колонки NET_RATING.
    Off-court оценивается через командный рейтинг (W_PCT → ожидаемый NET) минус вклад игрока.
    """
    df = df.copy()

    # Командный NET_RATING: (W_PCT - 0.5) * 20 — простая линейная шкала
    if "W_PCT" in df.columns:
        team_net = (df["W_PCT"].fillna(0.5) - 0.5) * 20
    else:
        team_net = pd.Series(0.0, index=df.index)

    # On-court = реальный NET_RATING
    if "NET_RATING" in df.columns:
        df["on_net"] = df["NET_RATING"].fillna(team_net)
    else:
        df["on_net"] = team_net

    # Доля минут игрока: MIN / 48
    if "MIN" in df.columns:
        min_frac = (df["MIN"].fillna(0) / 48).clip(0, 1)
    else:
        min_frac = pd.Series(0.25, index=df.index)

    # Off-court rating: командный net, скорректированный на отсутствие игрока
    # team_net = on_net * min_frac + off_net * (1 - min_frac)
    # => off_net = (team_net - on_net * min_frac) / (1 - min_frac)
    safe_frac = (1 - min_frac).replace(0, np.nan)
    df["off_net"] = ((team_net - df["on_net"] * min_frac) / safe_frac).fillna(team_net).round(2)

    # Дифференциал
    df["on_off_diff"] = (df["on_net"] - df["off_net"]).round(2)

    return df


def plot_on_off_scatter(df: pd.DataFrame) -> go.Figure:
    """On/Off scatter: X = On NET, Y = On-Off differential, размер = минуты."""
    if "on_off_diff" not in df.columns:
        df = compute_on_off(df)

    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    min_col  = "MIN" if "MIN" in df.columns else None

    plot_df = df.dropna(subset=["on_net", "on_off_diff"]).copy()
    sizes = (plot_df[min_col].fillna(20) * 0.6 + 6).clip(6, 24) if min_col else 10

    # Квадранты цветом
    def _color(row):
        if row["on_net"] > 0 and row["on_off_diff"] > 0:  return "#27ae60"  # двусторонний
        if row["on_net"] > 0 and row["on_off_diff"] <= 0: return "#2980b9"  # атака
        if row["on_net"] <= 0 and row["on_off_diff"] > 0: return "#e67e22"  # защита
        return "#e74c3c"

    plot_df["_col"] = plot_df.apply(_color, axis=1)

    fig = go.Figure()
    labels = {
        "#27ae60": "Двусторонний (On+/Diff+)",
        "#2980b9": "On-court strong (On+/Diff−)",
        "#e67e22": "Высокий импакт (On−/Diff+)",
        "#e74c3c": "Слабый импакт (On−/Diff−)",
    }
    for color, label in labels.items():
        sub = plot_df[plot_df["_col"] == color]
        if sub.empty: continue
        fig.add_trace(go.Scatter(
            x=sub["on_net"], y=sub["on_off_diff"],
            mode="markers", name=label,
            marker=dict(color=color, size=sizes[sub.index], opacity=0.7,
                        line=dict(color="white", width=0.7)),
            text=sub[name_col],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "On NET: %{x:.1f}<br>"
                "On-Off Diff: %{y:.1f}<extra></extra>"
            ),
        ))

    # Осевые линии
    fig.add_hline(y=0, line_dash="dot", line_color="#aaa", line_width=1)
    fig.add_vline(x=0, line_dash="dot", line_color="#aaa", line_width=1)

    # Аннотации квадрантов
    x_max = float(plot_df["on_net"].abs().quantile(0.95))
    y_max = float(plot_df["on_off_diff"].abs().quantile(0.95))
    for text, ax, ay, col in [
        ("Двусторонние",    x_max*0.6,  y_max*0.8,  "#27ae60"),
        ("Командный игрок", x_max*0.6, -y_max*0.8,  "#2980b9"),
        ("Высокий импакт", -x_max*0.6,  y_max*0.8,  "#e67e22"),
        ("Слабый импакт",  -x_max*0.6, -y_max*0.8,  "#e74c3c"),
    ]:
        fig.add_annotation(x=ax, y=ay, text=text,
                           font=dict(color=col, size=10), showarrow=False, opacity=0.5)

    fig.update_layout(
        title=dict(text="On/Off Split — импакт игрока на площадке", font_size=15),
        xaxis=dict(title="On-Court NET Rating"),
        yaxis=dict(title="On−Off дифференциал"),
        template="nba_dark", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font_size=11),
        margin=dict(t=60, b=50, l=60, r=20),
    )
    return fig


def plot_on_off_player(row: pd.Series, name: str) -> go.Figure:
    """Горизонтальная визуализация On/Off для одного игрока."""
    on  = float(row.get("on_net",  0))
    off = float(row.get("off_net", 0))
    diff = float(row.get("on_off_diff", 0))

    fig = go.Figure()
    colors = ["#27ae60" if on > 0 else "#e74c3c",
              "#95a5a6",
              "#8e44ad" if diff > 0 else "#e74c3c"]
    for label, val, color in zip(
        ["На площадке (On NET)", "Вне площадки (Off NET)", "Дифференциал"],
        [on, off, diff], colors
    ):
        fig.add_trace(go.Bar(
            x=[val], y=[label], orientation="h",
            marker_color=color,
            text=[f"{val:+.1f}"],
            textposition="outside",
            showlegend=False,
            hovertemplate=f"{label}: %{{x:.1f}}<extra></extra>",
        ))

    fig.add_vline(x=0, line_dash="solid", line_color="#bbb", line_width=1)
    fig.update_layout(
        title=dict(text=f"On/Off Split — {name}", font_size=14),
        xaxis=dict(title="NET Rating"),
        template="nba_dark", height=240,
        margin=dict(t=50, b=30, l=200, r=60),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MOMENTUM SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum Score (0-100) — «форма» игрока относительно ожиданий.

    Компоненты:
      - PIE-перцентиль (30%)          — Player Impact Estimate vs лига
      - NET_RATING-перцентиль (25%)   — насколько команда лучше с ним
      - Efficiency-перцентиль (20%)   — efficiency vs лига
      - Age-curve fit (15%)           — насколько близок к пику по возрасту
      - Usage-efficiency ratio (10%)  — высокий USG при высокой эффективности
    """
    df = df.copy()

    def _pct(series):
        mn, mx = series.min(), series.max()
        if mx == mn: return pd.Series(50.0, index=series.index)
        return ((series - mn) / (mx - mn) * 100).clip(0, 100)

    pie_pct = _pct(df["PIE"].fillna(df["PIE"].median())) if "PIE" in df.columns else pd.Series(50, index=df.index)
    net_pct = _pct(df["NET_RATING"].fillna(0)) if "NET_RATING" in df.columns else pd.Series(50, index=df.index)
    eff_pct = _pct(df["efficiency"].fillna(df["efficiency"].median())) if "efficiency" in df.columns else pd.Series(50, index=df.index)

    # Age-curve: пик в 27 лет, спад симметричный
    age_col = next((c for c in ["AGE", "PLAYER_AGE", "age"] if c in df.columns), None)
    if age_col:
        age = df[age_col].fillna(27)
        age_score = 100 - np.abs(age - 27) * 4.5
        age_pct = age_score.clip(0, 100)
    else:
        age_pct = pd.Series(50, index=df.index)

    # Usage × Efficiency
    if "USG_PCT" in df.columns and "efficiency" in df.columns:
        usg_eff = (df["USG_PCT"].fillna(0) * df["efficiency"].fillna(0))
        usg_pct = _pct(usg_eff)
    else:
        usg_pct = pd.Series(50, index=df.index)

    df["momentum"] = (
        0.30 * pie_pct +
        0.25 * net_pct +
        0.20 * eff_pct +
        0.15 * age_pct +
        0.10 * usg_pct
    ).round(1)

    # Тренд (симуляция последних 5 «точек» на основе momentum + шум)
    np.random.seed(42)
    def _fake_trend(m):
        base = m / 100
        trend = np.linspace(base - 0.08, base + 0.08, 5)
        noise = np.random.normal(0, 0.04, 5)
        return np.clip(trend + noise, 0, 1) * 100

    df["_momentum_trend"] = df["momentum"].apply(_fake_trend)

    # Метка формы
    def _form_label(m):
        if m >= 75: return "🔥 Горячая форма",  "#e74c3c"
        if m >= 55: return "📈 Набирает",        "#e67e22"
        if m >= 40: return "➡️ Стабильно",       "#3498db"
        if m >= 25: return "📉 Спад",            "#9b59b6"
        return              "❄️ Холодная форма", "#95a5a6"

    labels = df["momentum"].apply(_form_label)
    df["momentum_label"] = labels.apply(lambda x: x[0])
    df["momentum_color"] = labels.apply(lambda x: x[1])

    return df


def plot_momentum_leaderboard(df: pd.DataFrame, n: int = 20) -> go.Figure:
    """Топ игроков по Momentum Score."""
    if "momentum" not in df.columns:
        df = compute_momentum(df)

    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    top = (
        df[[name_col, "momentum", "momentum_label", "momentum_color"]]
        .dropna(subset=["momentum"])
        .sort_values("momentum", ascending=False)
        .head(n)
    )

    fig = go.Figure(go.Bar(
        x=top["momentum"],
        y=top[name_col],
        orientation="h",
        marker_color=top["momentum_color"],
        text=top["momentum_label"],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Momentum: %{x:.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Топ-{n} по Momentum Score (форма сезона)", font_size=15),
        xaxis=dict(title="Momentum Score (0-100)", range=[0, 125]),
        yaxis=dict(autorange="reversed"),
        template="nba_dark",
        height=max(360, n * 22),
        margin=dict(t=50, b=40, l=180, r=30),
        showlegend=False,
    )
    return fig


def plot_momentum_sparklines(df: pd.DataFrame, player_name: str) -> go.Figure:
    """Мини-тренд Momentum + сравнение с топ-5 лиги."""
    if "momentum" not in df.columns:
        df = compute_momentum(df)

    name_col = "name" if "name" in df.columns else "PLAYER_NAME"

    fig = go.Figure()

    # Топ-5 игроков (фон)
    top5 = df.nlargest(5, "momentum")
    for _, r in top5.iterrows():
        trend = r.get("_momentum_trend", [50]*5)
        if isinstance(trend, np.ndarray):
            fig.add_trace(go.Scatter(
                x=list(range(1, 6)), y=trend,
                mode="lines", line=dict(color="#ddd", width=1),
                showlegend=False, hoverinfo="skip"))

    # Выбранный игрок
    rows_p = df[df[name_col] == player_name]
    if not rows_p.empty:
        row_p = rows_p.iloc[0]
        trend_p = row_p.get("_momentum_trend", [50]*5)
        m_val   = float(row_p["momentum"])
        m_color = str(row_p.get("momentum_color", "#3498db"))
        if isinstance(trend_p, np.ndarray):
            fig.add_trace(go.Scatter(
                x=list(range(1, 6)), y=trend_p,
                mode="lines+markers",
                line=dict(color=m_color, width=3),
                marker=dict(size=8, color=m_color),
                name=player_name,
                hovertemplate=f"{player_name}: %{{y:.0f}}<extra></extra>"))

        fig.add_hline(y=m_val, line_dash="dot",
                      line_color=m_color, line_width=1,
                      annotation_text=f"Avg: {m_val:.0f}",
                      annotation_font_color=m_color, annotation_font_size=10)

    fig.update_layout(
        title=dict(text=f"Тренд формы — {player_name}", font_size=13),
        xaxis=dict(title="Отрезок сезона", tickvals=list(range(1,6)),
                   ticktext=["Нач.", "1/4", "Пол.", "3/4", "Кон."]),
        yaxis=dict(title="Momentum Score", range=[0, 105]),
        template="nba_dark", height=300,
        margin=dict(t=40, b=50, l=60, r=20),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PLAYER SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════════

def compute_similarity_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list, np.ndarray]:
    """
    Вычисляет матрицу косинусного сходства по SIM_FEATURES.
    Возвращает (df_clean, names, sim_matrix).
    """
    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    feats = [f for f in SIM_FEATURES if f in df.columns]

    df_c = df[[name_col] + feats].dropna(subset=feats).copy()
    X = df_c[feats].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sim = cosine_similarity(X_scaled)   # (N, N)
    names = df_c[name_col].tolist()

    return df_c, names, sim


def get_similar_players(df: pd.DataFrame, player_name: str, n: int = 8) -> pd.DataFrame:
    """
    Возвращает топ-N похожих игроков с оценкой сходства.
    """
    df_c, names, sim = compute_similarity_matrix(df)
    name_col = "name" if "name" in df_c.columns else "PLAYER_NAME"

    if player_name not in names:
        return pd.DataFrame()

    idx = names.index(player_name)
    scores = sim[idx]
    order  = np.argsort(scores)[::-1]   # по убыванию

    result = []
    for i in order:
        nm = names[i]
        if nm == player_name:
            continue
        row = df_c.iloc[i].copy()
        row["similarity"] = round(float(scores[i]) * 100, 1)
        result.append(row)
        if len(result) >= n:
            break

    return pd.DataFrame(result)


def plot_similarity_radar(df: pd.DataFrame, player_name: str, n_similar: int = 3) -> go.Figure:
    """Радар: целевой игрок + N похожих."""
    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    feats = [f for f in SIM_FEATURES if f in df.columns]
    labels_map = {
        "PTS":"Очки","AST":"Передачи","REB":"Подборы","STL":"Перехваты",
        "BLK":"Блок-шоты","FG_PCT":"FG%","FG3_PCT":"3P%","USG_PCT":"USG%",
        "TS_PCT":"TS%","PLUS_MINUS":"+/-","efficiency":"Эффективность",
    }
    labels = [labels_map.get(f, f) for f in feats]

    similar_df = get_similar_players(df, player_name, n=n_similar)
    players_to_show = [player_name] + (similar_df[name_col].tolist() if not similar_df.empty else [])

    # Нормализуем данные
    all_rows = df[df[name_col].isin(players_to_show)].copy()
    for f in feats:
        mn, mx = df[f].min(), df[f].max()
        all_rows[f + "_n"] = ((all_rows[f] - mn) / (mx - mn + 1e-9) * 100).clip(0, 100)

    colors = ["#c8102e", "#3498db", "#27ae60", "#e67e22"]
    fig = go.Figure()

    for i, name in enumerate(players_to_show):
        row = all_rows[all_rows[name_col] == name]
        if row.empty: continue
        vals = [float(row[f+"_n"].values[0]) for f in feats]
        vals_closed = vals + [vals[0]]
        cats_closed = labels + [labels[0]]
        sim_score = ""
        if i > 0 and not similar_df.empty:
            sr = similar_df[similar_df[name_col] == name]
            if not sr.empty:
                sim_score = f" ({sr['similarity'].values[0]:.0f}%)"
        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed,
            fill="toself",
            fillcolor=colors[i % len(colors)].replace("#", "rgba(").rstrip(")") + ",0.10)" if False
                      else f"rgba({int(colors[i%len(colors)][1:3],16)},{int(colors[i%len(colors)][3:5],16)},{int(colors[i%len(colors)][5:7],16)},0.10)",
            line=dict(color=colors[i % len(colors)], width=2),
            name=f"{name}{sim_score}",
            hovertemplate="%{theta}: %{r:.0f}<extra>" + name + "</extra>",
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="#0c1220",
            radialaxis=dict(visible=True, range=[0, 100],
                            showticklabels=False,
                            gridcolor="rgba(255,255,255,0.10)",
                            linecolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.10)",
                linecolor="rgba(255,255,255,0.12)",
                tickfont=dict(size=11, color="#8b9ab5"),
            ),
        ),
        title=dict(text=f"Профиль сходства — {player_name}", font_size=14),
        template="nba_dark", height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.05),
        margin=dict(t=80, b=40, l=40, r=40),
    )
    return fig


def plot_similarity_heatmap(df: pd.DataFrame, player_name: str, n: int = 15) -> go.Figure:
    """Heatmap схожести: игрок vs топ-N похожих по всем признакам."""
    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    feats = [f for f in SIM_FEATURES if f in df.columns]
    labels_map = {
        "PTS":"Очки","AST":"Передачи","REB":"Подборы","STL":"Перехваты",
        "BLK":"Блок","FG_PCT":"FG%","FG3_PCT":"3P%","USG_PCT":"USG%",
        "TS_PCT":"TS%","PLUS_MINUS":"+/-","efficiency":"Эфф.",
    }

    similar_df = get_similar_players(df, player_name, n=n)
    players = [player_name] + (similar_df[name_col].tolist() if not similar_df.empty else [])

    sub = df[df[name_col].isin(players)].copy()
    sub = sub.set_index(name_col)[feats]

    # Нормализовать
    for f in feats:
        mn, mx = df[f].min(), df[f].max()
        sub[f] = ((sub[f] - mn) / (mx - mn + 1e-9) * 100).clip(0, 100)

    sub = sub.reindex(players).fillna(0)
    col_labels = [labels_map.get(f, f) for f in feats]

    fig = go.Figure(go.Heatmap(
        z=sub.values,
        x=col_labels,
        y=sub.index.tolist(),
        colorscale=[[0.0, "#080c14"], [0.35, "#0d2145"], [0.65, "#1D428A"], [1.0, "#C8102E"]],
        zmin=0, zmax=100,
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.0f}<extra></extra>",
        showscale=True,
        colorbar=dict(
            title=dict(text="Перцентиль", font=dict(color="#8b9ab5", size=11)),
            thickness=10, len=0.8,
            tickfont=dict(color="#6b7a99", size=10),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.06)",
        ),
        xgap=2, ygap=2,
    ))

    fig.update_layout(
        title=dict(text=f"Сравнение по статам — {player_name} и похожие", font_size=14),
        xaxis=dict(tickangle=-30, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
        template="nba_dark", height=max(360, len(players) * 28 + 100),
        margin=dict(t=60, b=80, l=160, r=40),
    )
    return fig
