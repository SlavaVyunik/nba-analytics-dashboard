"""
Кривая развития игрока — прогресс по возрасту + пик карьеры.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

AGE_COL_OPTIONS = ["PLAYER_AGE", "AGE", "age"]
KEY_STATS = ["PTS", "AST", "REB", "efficiency"]


def _get_age_col(df):
    return next((c for c in AGE_COL_OPTIONS if c in df.columns), None)


def player_career_phase(age):
    if age is None or pd.isna(age):
        return "Неизвестно", "❓", "Нет данных"
    age = float(age)
    if age < 22: return "Прорыв",       "🚀", "Быстрый рост, максимум потенциала"
    if age < 25: return "Рост",         "📈", "Приближается к пику"
    if age < 29: return "Пик",          "⭐", "Лучшие годы карьеры"
    if age < 32: return "Стабильность", "🔄", "Небольшой спад"
    if age < 35: return "Спад",         "📉", "Снижение физических данных"
    return             "Ветеран",       "🏅", "Опыт важнее физики"


def fit_age_curve(df, stat="PTS", degree=3):
    age_col = _get_age_col(df)
    if age_col is None or stat not in df.columns:
        return None, None, None
    sub = df[[age_col, stat]].dropna()
    sub = sub[(sub[age_col] >= 18) & (sub[age_col] <= 42)]
    if len(sub) < 15:
        return None, None, None
    X = sub[age_col].values.reshape(-1, 1)
    y = sub[stat].values
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("lr",   LinearRegression()),
    ])
    model.fit(X, y)
    ages_smooth = np.linspace(18, 42, 200)
    pred = np.clip(model.predict(ages_smooth.reshape(-1, 1)), 0, None)
    sub2 = sub.copy()
    sub2["age_bin"] = sub2[age_col].round(0)
    std_map = sub2.groupby("age_bin")[stat].std().reindex(np.arange(18, 43))
    std_map = std_map.ffill().bfill().fillna(1)
    std_smooth = np.interp(ages_smooth, std_map.index.values, std_map.values)
    return ages_smooth, pred, std_smooth


def _peak_age(ages, pred):
    if pred is None: return None
    return float(ages[np.argmax(pred)])


def get_development_summary(df, player_name):
    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    age_col  = _get_age_col(df)
    rows = df[df[name_col] == player_name]
    if rows.empty: return {}
    row = rows.iloc[0]
    age = float(row[age_col]) if age_col and not pd.isna(row.get(age_col, np.nan)) else None
    phase, emoji, desc = player_career_phase(age)
    peaks = {}
    for stat in KEY_STATS:
        if stat in df.columns:
            ages, pred, _ = fit_age_curve(df, stat)
            if ages is not None:
                peaks[stat] = round(_peak_age(ages, pred), 1)
    avg_peak = round(float(np.mean(list(peaks.values()))), 1) if peaks else None
    years_to_peak = round(avg_peak - age, 1) if (avg_peak and age) else None
    return {
        "age": age, "phase": phase, "phase_emoji": emoji,
        "phase_desc": desc, "peak_age": avg_peak,
        "years_to_peak": years_to_peak, "stat_peaks": peaks,
    }


def plot_development_curve(df, player_name, stat="PTS"):
    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    age_col  = _get_age_col(df)
    stat_labels = {"PTS":"Очки","AST":"Передачи","REB":"Подборы","efficiency":"Эффективность"}
    label = stat_labels.get(stat, stat)

    ages, pred, std = fit_age_curve(df, stat)
    if ages is None:
        return go.Figure().update_layout(title="Недостаточно данных")

    peak = _peak_age(ages, pred)
    fig = go.Figure()

    # Цветные зоны фаз
    zones = [(18,22,"rgba(39,174,96,0.07)","Прорыв"),
             (22,25,"rgba(41,128,185,0.07)","Рост"),
             (25,29,"rgba(142,68,173,0.09)","Пик"),
             (29,32,"rgba(230,126,34,0.07)","Стабильность"),
             (32,42,"rgba(231,76,60,0.07)","Спад")]
    for x0, x1, col, nm in zones:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=col, line_width=0,
                      annotation_text=nm, annotation_position="top left",
                      annotation_font_size=9, annotation_font_color="#999")

    # Полоса ± std
    fig.add_trace(go.Scatter(
        x=np.concatenate([ages, ages[::-1]]),
        y=np.concatenate([pred+std, (pred-std)[::-1]]),
        fill="toself", fillcolor="rgba(100,100,100,0.10)",
        line=dict(width=0), name="± Разброс", hoverinfo="skip"))

    # Средняя кривая лиги
    fig.add_trace(go.Scatter(
        x=ages, y=pred,
        line=dict(color="#555", width=2),
        name="Среднее лиги",
        hovertemplate="Возраст %{x:.0f}: %{y:.1f}<extra></extra>"))

    # Пик
    if peak:
        fig.add_vline(x=peak, line_dash="dot", line_color="#8e44ad",
                      annotation_text=f"Пик ≈ {peak:.0f} лет",
                      annotation_font_color="#8e44ad", annotation_font_size=11)

    # Точка игрока
    prows = df[df[name_col] == player_name]
    if not prows.empty and age_col and stat in df.columns:
        row = prows.iloc[0]
        p_age = row.get(age_col)
        p_val = row.get(stat)
        if p_age is not None and p_val is not None and not pd.isna(p_age) and not pd.isna(p_val):
            phase, emoji, _ = player_career_phase(p_age)
            fig.add_trace(go.Scatter(
                x=[float(p_age)], y=[float(p_val)],
                mode="markers+text",
                marker=dict(size=16, color="#c8102e", symbol="star",
                            line=dict(color="white", width=2)),
                text=[f"{emoji} {player_name}"],
                textposition="top right",
                textfont=dict(size=11, color="#c8102e"),
                name=player_name,
                hovertemplate=f"<b>{player_name}</b><br>Возраст: {float(p_age):.0f}<br>{label}: {float(p_val):.1f}<extra></extra>"))
            fig.add_hline(y=float(p_val), line_dash="dot",
                          line_color="rgba(200,16,46,0.25)", line_width=1)

    fig.update_layout(
        title=dict(text=f"Кривая развития — <b>{label}</b>", font_size=15),
        xaxis=dict(title="Возраст", range=[18, 42], dtick=2),
        yaxis=dict(title=label),
        template="nba_dark", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=40, l=50, r=20),
    )
    return fig


def plot_multi_stat_curves(df, player_name):
    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    age_col  = _get_age_col(df)
    stats  = [s for s in KEY_STATS if s in df.columns]
    labels = {"PTS":"Очки","AST":"Передачи","REB":"Подборы","efficiency":"Эффективность"}
    n = len(stats)
    rows_n = (n + 1) // 2
    fig = make_subplots(rows=rows_n, cols=2,
                        subplot_titles=[labels.get(s,s) for s in stats],
                        vertical_spacing=0.15)

    prows = df[df[name_col] == player_name]
    player_row = prows.iloc[0] if not prows.empty else None

    for i, stat in enumerate(stats):
        r, c = divmod(i, 2)
        ages, pred, std = fit_age_curve(df, stat)
        if ages is None: continue

        fig.add_trace(go.Scatter(
            x=np.concatenate([ages, ages[::-1]]),
            y=np.concatenate([pred+std, (pred-std)[::-1]]),
            fill="toself", fillcolor="rgba(100,100,100,0.09)",
            line=dict(width=0), showlegend=False, hoverinfo="skip"),
            row=r+1, col=c+1)

        fig.add_trace(go.Scatter(
            x=ages, y=pred, line=dict(color="#555", width=1.5),
            showlegend=(i==0), name="Среднее лиги",
            hovertemplate="Возраст %{x:.0f}: %{y:.1f}<extra></extra>"),
            row=r+1, col=c+1)

        if player_row is not None and age_col and stat in df.columns:
            p_age = player_row.get(age_col)
            p_val = player_row.get(stat)
            if p_age is not None and p_val is not None and not pd.isna(p_age) and not pd.isna(p_val):
                fig.add_trace(go.Scatter(
                    x=[float(p_age)], y=[float(p_val)], mode="markers",
                    marker=dict(size=12, color="#c8102e", symbol="star",
                                line=dict(color="white", width=1.5)),
                    showlegend=(i==0), name=player_name,
                    hovertemplate=f"{player_name}: %{{y:.1f}}<extra></extra>"),
                    row=r+1, col=c+1)

    fig.update_layout(
        title=dict(text=f"Статистика по возрасту — <b>{player_name}</b>", font_size=14),
        template="nba_dark", height=440,
        margin=dict(t=60, b=30),
    )
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=11, color="#8b9ab5"))
    return fig
