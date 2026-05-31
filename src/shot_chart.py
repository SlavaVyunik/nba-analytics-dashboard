"""
Карта бросков (Shot Chart) — синтетическая генерация на основе статистики игрока.
Рисуем площадку NBA в Plotly и наносим гексагональную тепловую карту бросков.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ── Размеры площадки (NBA, футы, начало координат — центр щита) ───────────────
COURT_W  = 50    # ширина площадки (−25 ... +25)
COURT_H  = 47    # полуплощадка — ограничение по глубине атакующей зоны
KEY_W    = 16    # ширина «краски»
KEY_H    = 19    # высота «краски»
RIM_R    = 0.75  # радиус кольца
FT_R     = 6     # радиус штрафного круга
THREE_R  = 23.75 # дуга трёхочковой
THREE_CO = 22    # угловые три (corner three cutoff y)
BACKBOARD= 3     # длина щита

# Цвета тем — NBA Dark
COURT_COLOR   = "#1a2035"   # тёмно-синий паркет
LINE_COLOR    = "#3a4a6b"   # приглушённые линии
PAINT_COLOR   = "#162040"   # тёмная краска
RIM_COLOR     = "#C8102E"   # красное кольцо
HOOP_BG       = "rgba(200,16,46,0.20)"
PAPER_COLOR   = "#111622"   # фон фигуры


# ── Построение форм площадки ──────────────────────────────────────────────────
def _arc(cx, cy, r, theta1, theta2, n=60):
    """Дуга окружности: возвращает (x_list, y_list)."""
    angles = np.linspace(np.radians(theta1), np.radians(theta2), n)
    return cx + r * np.cos(angles), cy + r * np.sin(angles)


def _court_shapes():
    """
    Возвращает список shape-dict для Plotly.
    Заливки → layer='below'  (под тепловой картой).
    Линии разметки → layer='above' (поверх данных).
    """
    below = []   # фон, краска — рисуются ПОД данными
    above = []   # линии, дуги   — рисуются НАД данными

    # ── Фон площадки (под данными) ────────────────────────────────────────────
    below.append(dict(type="rect", x0=-25, y0=-2, x1=25, y1=COURT_H,
                      fillcolor=COURT_COLOR, layer="below",
                      line=dict(color=LINE_COLOR, width=2)))

    # ── Краска (под данными) ──────────────────────────────────────────────────
    below.append(dict(type="rect",
                      x0=-KEY_W/2, y0=0, x1=KEY_W/2, y1=KEY_H,
                      fillcolor=PAINT_COLOR, layer="below",
                      line=dict(color="rgba(0,0,0,0)", width=0)))

    # ── Штрафной круг — нижняя заливка (под данными) ─────────────────────────
    ax, ay = _arc(0, KEY_H, FT_R, 0, 180)
    path_ft = "M" + " L".join(f"{x:.2f},{y:.2f}" for x,y in zip(ax, ay)) + " Z"
    below.append(dict(type="path", path=path_ft,
                      fillcolor=PAINT_COLOR, layer="below",
                      line=dict(color="rgba(0,0,0,0)", width=0)))

    # ── Линии разметки (НАД данными) ─────────────────────────────────────────
    # Граница площадки
    above.append(dict(type="rect", x0=-25, y0=-2, x1=25, y1=COURT_H,
                      fillcolor="rgba(0,0,0,0)", layer="above",
                      line=dict(color=LINE_COLOR, width=2)))

    # Краска
    above.append(dict(type="rect",
                      x0=-KEY_W/2, y0=0, x1=KEY_W/2, y1=KEY_H,
                      fillcolor="rgba(0,0,0,0)", layer="above",
                      line=dict(color=LINE_COLOR, width=1.5)))

    # Штрафной круг
    ax2, ay2 = _arc(0, KEY_H, FT_R, 0, 360)
    path_ft2 = "M" + " L".join(f"{x:.2f},{y:.2f}" for x,y in zip(ax2, ay2)) + " Z"
    above.append(dict(type="path", path=path_ft2,
                      fillcolor="rgba(0,0,0,0)", layer="above",
                      line=dict(color=LINE_COLOR, width=1.5)))

    # Кольцо
    above.append(dict(type="circle",
                      x0=-RIM_R, y0=-RIM_R, x1=RIM_R, y1=RIM_R,
                      fillcolor="rgba(0,0,0,0)", layer="above",
                      line=dict(color=RIM_COLOR, width=2.5)))

    # Щит
    above.append(dict(type="rect",
                      x0=-BACKBOARD, y0=-0.2, x1=BACKBOARD, y1=0.15,
                      fillcolor=LINE_COLOR, layer="above",
                      line=dict(color=LINE_COLOR, width=1)))

    # Трёхочковая дуга
    ax3, ay3 = _arc(0, 0, THREE_R, 22, 158, n=120)
    path_3 = (f"M -25,{THREE_CO:.1f}"
              + f" L {ax3[0]:.2f},{ay3[0]:.2f}"
              + " " + " ".join(f"L{x:.2f},{y:.2f}" for x,y in zip(ax3[1:], ay3[1:]))
              + f" L 25,{THREE_CO:.1f}")
    above.append(dict(type="path", path=path_3,
                      fillcolor="rgba(0,0,0,0)", layer="above",
                      line=dict(color=LINE_COLOR, width=1.8)))

    # Центральный круг
    above.append(dict(type="circle",
                      x0=-6, y0=COURT_H-6, x1=6, y1=COURT_H+6,
                      fillcolor="rgba(0,0,0,0)", layer="above",
                      line=dict(color=LINE_COLOR, width=1.5)))

    # No-charge arc
    ax_nc, ay_nc = _arc(0, 0, 4, 0, 180)
    path_nc = "M" + " L".join(f"{x:.2f},{y:.2f}" for x,y in zip(ax_nc, ay_nc))
    above.append(dict(type="path", path=path_nc,
                      fillcolor="rgba(0,0,0,0)", layer="above",
                      line=dict(color=LINE_COLOR, width=1)))

    return below + above


# ── Генерация синтетических бросков ──────────────────────────────────────────
def _generate_shots(row, n_shots=300):
    """
    Синтетически генерируем броски на основе архетипа и статистики игрока.
    Возвращает DataFrame: x, y, made (bool).
    """
    pts      = float(row.get("PTS",  10) or 10)
    ast      = float(row.get("AST",   3) or 3)
    fg_pct   = float(row.get("FG_PCT", 0.45) or 0.45)
    fg3_pct  = float(row.get("FG3_PCT", 0.35) or 0.35)
    usg      = float(row.get("USG_PCT", 0.20) or 0.20)
    cluster  = str(row.get("cluster_name", ""))

    rng = np.random.default_rng(42 + int(pts * 10) % 100)

    # Пропорции зон бросков зависят от типа игрока
    if "Звезда" in cluster or "Scorе" in cluster:
        p_rim, p_mid, p_corner3, p_arc3 = 0.30, 0.18, 0.15, 0.37
    elif "Плеймейкер" in cluster:
        p_rim, p_mid, p_corner3, p_arc3 = 0.30, 0.20, 0.12, 0.38
    elif "Big Man" in cluster or "Большой" in cluster:
        p_rim, p_mid, p_corner3, p_arc3 = 0.55, 0.28, 0.04, 0.13
    elif "3-and-D" in cluster:
        p_rim, p_mid, p_corner3, p_arc3 = 0.22, 0.08, 0.28, 0.42
    else:  # Ролевой
        p_rim, p_mid, p_corner3, p_arc3 = 0.35, 0.22, 0.18, 0.25

    zones = rng.choice(["rim","mid","corner3","arc3"],
                       size=n_shots,
                       p=[p_rim, p_mid, p_corner3, p_arc3])

    xs, ys, made = [], [], []

    for z in zones:
        if z == "rim":
            r    = rng.uniform(0, 3)
            ang  = rng.uniform(0, np.pi)
            x, y = r * np.cos(ang), r * np.sin(ang)
            hit  = rng.random() < min(fg_pct + 0.1, 0.72)
        elif z == "mid":
            r    = rng.uniform(3, 16)
            ang  = rng.uniform(0, np.pi)
            x, y = r * np.cos(ang), r * np.sin(ang)
            hit  = rng.random() < max(fg_pct - 0.06, 0.30)
        elif z == "corner3":
            side = rng.choice([-1, 1])
            x    = side * rng.uniform(21, 24.5)
            y    = rng.uniform(0, THREE_CO)
            hit  = rng.random() < fg3_pct
        else:  # arc3
            ang  = rng.uniform(np.radians(22), np.radians(158))
            r    = rng.uniform(22.5, 25)
            x, y = r * np.cos(ang), r * np.sin(ang)
            hit  = rng.random() < fg3_pct

        # Не выходим за границы площадки
        x = np.clip(x, -24.5, 24.5)
        y = np.clip(y, 0, COURT_H - 1)
        xs.append(x)
        ys.append(y)
        made.append(hit)

    return pd.DataFrame({"x": xs, "y": ys, "made": made})


# ── Зональная тепловая карта (scatter circles) ───────────────────────────────
def _zone_trace(shots, bin_size=3.5):
    """
    Рисует зоны бросков как цветные круги переменного размера.
    Размер = объём бросков, цвет = FG%.
    Работает надёжно в любом Plotly/Streamlit окружении.
    """
    bx = np.round(shots["x"] / bin_size) * bin_size
    by = np.round(shots["y"] / bin_size) * bin_size
    shots = shots.copy()
    shots["bx"] = bx
    shots["by"] = by

    agg = (shots.groupby(["bx", "by"])
               .agg(attempts=("made", "count"), pct=("made", "mean"))
               .reset_index())

    agg = agg[agg["attempts"] >= 1].copy()
    if agg.empty:
        return go.Scatter(x=[], y=[], mode="markers", showlegend=False)

    # Размер кружка: 14–42 пикселей (немного уменьшил чтобы не вылезали за края)
    max_att = max(agg["attempts"].max(), 1)
    agg["sz"] = ((agg["attempts"] / max_att) ** 0.5 * 28 + 14).clip(14, 42)

    # Цвет → синий (холодная зона) → красный (горячая)
    COLORSCALE = [
        [0.00, "#1a237e"],  # тёмно-синий
        [0.25, "#1565c0"],  # синий
        [0.50, "#ff8f00"],  # оранжевый (средне)
        [0.75, "#e53935"],  # красный
        [1.00, "#b71c1c"],  # тёмно-красный
    ]

    return go.Scatter(
        x=agg["bx"],
        y=agg["by"],
        mode="markers",
        marker=dict(
            symbol="circle",
            size=agg["sz"].tolist(),
            sizemode="diameter",
            color=agg["pct"],
            colorscale=COLORSCALE,
            cmin=0.25, cmax=0.65,
            colorbar=dict(
                title=dict(text="FG%", side="right"),
                thickness=14, len=0.55,
                tickformat=".0%",
                tickvals=[0.25, 0.35, 0.45, 0.55, 0.65],
            ),
            opacity=1.0,
            line=dict(color="rgba(255,255,255,0.7)", width=1.0),
        ),
        customdata=np.stack([agg["attempts"], agg["pct"]], axis=1),
        hovertemplate=(
            "Бросков: <b>%{customdata[0]:.0f}</b><br>"
            "FG%%: <b>%{customdata[1]:.1%%}</b><extra></extra>"
        ),
        showlegend=False,
    )


# ── Hex scatter (альтернативный режим) ───────────────────────────────────────
def _hexbin_trace(shots, hex_size=3.0):
    """Гексагональный режим — alias на _zone_trace с hex-символом."""
    bx = np.round(shots["x"] / hex_size) * hex_size
    by = np.round(shots["y"] / hex_size) * hex_size
    shots = shots.copy()
    shots["bx"] = bx
    shots["by"] = by
    agg = (shots.groupby(["bx", "by"])
               .agg(attempts=("made", "count"), pct=("made", "mean"))
               .reset_index())
    agg = agg[agg["attempts"] >= 1].copy()
    if agg.empty:
        return go.Scatter(x=[], y=[], mode="markers", showlegend=False)
    max_att = max(agg["attempts"].max(), 1)
    agg["sz"] = ((agg["attempts"] / max_att) ** 0.5 * 28 + 10).clip(10, 38)
    return go.Scatter(
        x=agg["bx"], y=agg["by"], mode="markers",
        marker=dict(
            symbol="hexagon",
            size=agg["sz"].tolist(),
            sizemode="diameter",
            color=agg["pct"],
            colorscale=[[0,"#1a237e"],[0.5,"#ff8f00"],[1,"#b71c1c"]],
            cmin=0.25, cmax=0.65,
            colorbar=dict(title=dict(text="FG%"), thickness=14,
                          tickformat=".0%", len=0.55),
            opacity=1.0,
            line=dict(color="rgba(255,255,255,0.6)", width=0.8),
        ),
        customdata=np.stack([agg["attempts"], agg["pct"]], axis=1),
        hovertemplate="Бросков: %{customdata[0]:.0f}<br>FG%%: %{customdata[1]:.1%%}<extra></extra>",
        showlegend=False,
    )


# ── Точки (сделанные / промахи) ───────────────────────────────────────────────
def _dot_traces(shots):
    made   = shots[shots["made"]]
    missed = shots[~shots["made"]]

    t_made = go.Scatter(
        x=made["x"], y=made["y"], mode="markers", name="Попадание",
        marker=dict(color="#27ae60", size=5, opacity=0.55,
                    line=dict(color="white", width=0.3)),
    )
    t_miss = go.Scatter(
        x=missed["x"], y=missed["y"], mode="markers", name="Промах",
        marker=dict(color="#e74c3c", size=5, opacity=0.4, symbol="x",
                    line=dict(color="#e74c3c", width=0.5)),
    )
    return t_made, t_miss


# ── Главная функция ───────────────────────────────────────────────────────────
def shot_chart(row, mode="hex", n_shots=350):
    """
    Построить карту бросков для игрока.

    Parameters
    ----------
    row    : pandas Series — строка DataFrame с данными игрока
    mode   : 'hex' (тепловая карта) | 'dots' (точки сделано/промах)
    n_shots: количество синтетических бросков

    Returns
    -------
    plotly.graph_objects.Figure
    """
    player_name = row.get("name") or row.get("PLAYER_NAME") or "Игрок"
    fg_pct  = float(row.get("FG_PCT",  0.45) or 0.45) * 100
    fg3_pct = float(row.get("FG3_PCT", 0.35) or 0.35) * 100

    shots = _generate_shots(row, n_shots=n_shots)

    fig = go.Figure()
    fig.update_layout(shapes=_court_shapes())

    if mode == "dots":
        t1, t2 = _dot_traces(shots)
        fig.add_trace(t2)
        fig.add_trace(t1)
    elif mode == "hex":
        fig.add_trace(_hexbin_trace(shots))
    else:  # "heat" — зональные круги (основной режим)
        fig.add_trace(_zone_trace(shots))

    # Подписи зон
    zone_labels = [
        dict(x=0,    y=2.5,  text="У кольца",  font_size=9),
        dict(x=0,    y=11,   text="Средняя",   font_size=9),
        dict(x=-24,  y=10,   text="Corner 3",  font_size=9),
        dict(x=24,   y=10,   text="Corner 3",  font_size=9),
        dict(x=0,    y=27,   text="Arc 3",     font_size=9),
    ]
    for lbl in zone_labels:
        fig.add_annotation(
            x=lbl["x"], y=lbl["y"],
            text=lbl["text"],
            showarrow=False,
            font=dict(size=lbl["font_size"], color="#4a6080"),
            opacity=0.8,
        )

    made_pct = shots["made"].mean() * 100
    total    = len(shots)

    fig.update_layout(
        title=dict(
            text=f"<b style='color:#ffffff'>{player_name}</b>"
                 f"<span style='font-size:12px;color:#6b7a99'>"
                 f"  ·  FG {fg_pct:.1f}%  ·  3P {fg3_pct:.1f}%  ·  "
                 f"{total} бросков</span>",
            font=dict(size=15, color="#e8eaf6",
                      family="'Segoe UI', Barlow, Arial, sans-serif"),
        ),
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=COURT_COLOR,
        xaxis=dict(range=[-28, 28], showgrid=False, zeroline=False,
                   showticklabels=False),
        yaxis=dict(range=[-4, COURT_H + 2], showgrid=False, zeroline=False,
                   showticklabels=False),
        height=500,
        width=590,
        margin=dict(l=10, r=65, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0.5,
                    xanchor="center",
                    font=dict(color="#e8eaf6", size=11)),
        hoverlabel=dict(
            bgcolor="#1e2740", bordercolor="#C8102E",
            font=dict(color="white", size=12),
        ),
    )

    return fig


# ── Сравнение карт двух игроков ───────────────────────────────────────────────
def shot_chart_compare(row1, row2):
    """Два shot chart рядом для Head-to-Head."""
    from plotly.subplots import make_subplots

    name1 = row1.get("name") or row1.get("PLAYER_NAME") or "Игрок 1"
    name2 = row2.get("name") or row2.get("PLAYER_NAME") or "Игрок 2"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"🏀 {name1}", f"🏀 {name2}"],
        horizontal_spacing=0.04,
    )

    for col_idx, (row, shapes_key) in enumerate([(row1, "x"), (row2, "x2")], start=1):
        shots = _generate_shots(row, n_shots=300)
        trace = _hexbin_trace(shots)
        trace.showlegend = False
        # colorbar только для второго
        if col_idx == 1:
            trace.marker.colorbar = None
        fig.add_trace(trace, row=1, col=col_idx)

    # Формы площадки для обоих subplot'ов
    s1 = _court_shapes()
    s2 = [{**s, "xref": "x2", "yref": "y2"} for s in _court_shapes()]

    fig.update_layout(
        shapes=s1 + s2,
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=COURT_COLOR,
        height=500,
        margin=dict(l=5, r=5, t=55, b=5),
        hoverlabel=dict(
            bgcolor="#1e2740", bordercolor="#C8102E",
            font=dict(color="white", size=12),
        ),
    )
    # Style subplot titles (annotations) for dark theme
    for ann in fig.layout.annotations:
        ann.update(
            font=dict(size=13, color="#e8eaf6",
                      family="'Segoe UI', Inter, Arial, sans-serif"),
        )
    for ax in ["xaxis", "xaxis2"]:
        fig.update_layout(**{ax: dict(range=[-25.5,25.5], showgrid=False,
                                      zeroline=False, showticklabels=False,
                                      scaleanchor="y" if ax=="xaxis" else "y2")})
    for ax in ["yaxis", "yaxis2"]:
        fig.update_layout(**{ax: dict(range=[-2,COURT_H], showgrid=False,
                                      zeroline=False, showticklabels=False)})
    for s in fig.layout.shapes:
        s.update({"fillcolor": s.get("fillcolor", COURT_COLOR)})

    return fig


if __name__ == "__main__":
    test_row = {
        "name": "Test Player", "cluster_name": "Звезда лиги",
        "PTS": 28.5, "AST": 5.2, "REB": 4.1,
        "FG_PCT": 0.51, "FG3_PCT": 0.38, "USG_PCT": 0.32,
    }
    fig = shot_chart(pd.Series(test_row))
    fig.show()
