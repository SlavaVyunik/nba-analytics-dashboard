import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
from google import genai as google_genai
from src.plots import umap_scatter, radar_chart, top_players_bar, pts_vs_ast

# ── NBA Theme ─────────────────────────────────────────────────────────────────
from app.nba_theme import register_nba_template, NBA_CSS, page_header, section_title, kpi_row
register_nba_template()

# ── Конфиг ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Analytics",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(NBA_CSS, unsafe_allow_html=True)

# ── Данные ────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "players_clustered.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

if not os.path.exists(DATA_PATH):
    st.error("Запусти: fetch_data.py → features.py → clustering.py")
    st.stop()

df = load_data()
name_col = "name" if "name" in df.columns else "PLAYER_NAME"

# ── Вспомогательные функции ───────────────────────────────────────────────────
def get_photo_url(player_id):
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{int(player_id)}.png"

def player_img(player_id, width=200, radius=12, fallback="👤"):
    """Render player headshot via client-side <img> (avoids server-side SSL timeouts)."""
    url = get_photo_url(player_id)
    placeholder = (
        f"<div style='width:{width}px;height:{int(width*0.75)}px;"
        f"display:flex;align-items:center;justify-content:center;"
        f"font-size:{width//3}px;background:rgba(255,255,255,0.05);"
        f"border-radius:{radius}px'>{fallback}</div>"
    )
    st.html(
        f"<img src='{url}' width='{width}' "
        f"style='border-radius:{radius}px;display:block;' "
        f"onerror=\"this.style.display='none';this.nextElementSibling.style.display='flex'\"> "
        f"{placeholder.replace('display:flex', 'display:none')}"
    )

def percentile_of(series, value):
    return float((series < value).mean() * 100)

def pct_color(p):
    if p >= 75: return "#27ae60"
    if p >= 50: return "#f39c12"
    if p >= 25: return "#e67e22"
    return "#e74c3c"

def stat_bar_html(label, value, pct, fmt=".1f"):
    color = pct_color(pct)
    bar_w = max(2, int(pct * 1.8))
    return f"""
    <div class="stat-row">
      <span class="stat-label">{label}</span>
      <span class="stat-value">{value:{fmt}}</span>
      <span class="bar-bg"><span class="bar-fill" style="width:{bar_w}px;background:{color}"></span></span>
      <span class="pct-label" style="color:{color}">{pct:.0f}°</span>
    </div>"""

def find_similar(df, player_name, n=6):
    row = df[df[name_col] == player_name].iloc[0]
    others = df[df[name_col] != player_name].copy()
    others["_dist"] = np.sqrt(
        (others["umap_x"] - row["umap_x"])**2 +
        (others["umap_y"] - row["umap_y"])**2
    )
    cols = [name_col, "team", "cluster_name", "PTS", "AST", "REB", "_dist"]
    cols = [c for c in cols if c in others.columns]
    return others.nsmallest(n, "_dist")[cols]

def compare_stat(v1, v2, higher_is_better=True):
    if higher_is_better:
        return "🟢" if v1 > v2 else ("🔴" if v1 < v2 else "⚪")
    else:
        return "🟢" if v1 < v2 else ("🔴" if v1 > v2 else "⚪")

def page_hint(lines: list) -> None:
    """Collapsible info-box explaining what a page does and why."""
    _hint_key = f"hint_{abs(hash(lines[0][:40] if lines else ''))}"
    with st.expander("💡 Что здесь и зачем?", expanded=False, key=_hint_key):
        st.markdown("\n\n".join(lines))


# ── ML-кэш ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="🤖 Обучаем модель прогноза...")
def get_prediction_model(data_hash):
    from src.prediction import build_model, predict_next_season
    model, scaler, feats, metrics, y_cv = build_model(df)
    df_pred = predict_next_season(df, model, scaler, feats)
    return model, scaler, feats, metrics, y_cv, df_pred

@st.cache_data(show_spinner="🔍 Анализируем аномалии...")
def get_anomalies(data_hash):
    from src.anomaly import detect_anomalies, get_hidden_gems, get_overrated
    df_a = detect_anomalies(df)
    gems = get_hidden_gems(df_a, top_n=20)
    over = get_overrated(df_a, top_n=15)
    return df_a, gems, over

@st.cache_data(show_spinner="💰 Считаем Trade Value...")
def get_trade_values(data_hash):
    from src.trade_value import compute_trade_value
    return compute_trade_value(df)

@st.cache_data(show_spinner="🏥 Считаем риск травм...")
def get_injury_risks(data_hash):
    from src.injury_risk import compute_injury_risk
    return compute_injury_risk(df)

@st.cache_data(show_spinner="💵 Считаем контрактные стоимости...")
def get_contract_values(data_hash):
    from src.contract_value import compute_contract_value
    df_tv = get_trade_values(data_hash)
    return compute_contract_value(df_tv)

@st.cache_data(show_spinner="🔬 Считаем продвинутую аналитику...")
def get_advanced_stats(data_hash):
    from src.advanced_stats import compute_win_shares, compute_on_off, compute_momentum
    df_tv = get_trade_values(data_hash)
    d = compute_win_shares(df_tv)
    d = compute_on_off(d)
    d = compute_momentum(d)
    return d

@st.cache_data(show_spinner="🎲 Запускаем Monte Carlo...")
def get_monte_carlo(data_hash):
    from src.monte_carlo import compute_season_probabilities
    return compute_season_probabilities(df)

@st.cache_data(show_spinner="📐 Оптимизируем портфели...")
def get_portfolio_data(data_hash):
    from src.trade_value import compute_trade_value
    from src.injury_risk import compute_injury_risk
    d = compute_trade_value(df)
    d = compute_injury_risk(d)
    return d

@st.cache_data(show_spinner="🧮 Считаем байесовскую статистику...")
def get_bayesian_stats(data_hash):
    from src.bayes_stats import compute_bayesian_df
    return compute_bayesian_df(df)

@st.cache_data(show_spinner="🕸 Строим граф схожести...", ttl=600)
def get_network_data(data_hash):
    from src.network import compute_network
    return compute_network(df)

@st.cache_data(show_spinner="⚰️ Строим кривые выживаемости...")
def get_survival_data(data_hash):
    from src.injury_risk import compute_injury_risk
    from src.survival import prepare_survival_data, cox_hazard_table
    d = compute_injury_risk(df)
    surv = prepare_survival_data(d)
    cox  = cox_hazard_table(d)
    return surv, cox

# Хэш для инвалидации кэша
_data_hash = str(len(df)) + str(df.columns.tolist())

# ── Navigation state ──────────────────────────────────────────────────────────
if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 Обзор лиги"

CATEGORIES = {
    "📊 ОБЗОР ЛИГИ":          ["🏠 Обзор лиги", "🏆 Лидеры", "📊 Кластеры"],
    "👤 ИГРОКИ":               ["🏀 Карточка игрока", "⚔️ Head-to-Head"],
    "🤖 ML & АНАЛИТИКА":      ["💎 Аномалии", "🔮 Прогноз", "🔬 Продвинутая аналитика", "🤖 AI Скаут", "🔄 AI Трейд"],
    "💰 ТРЕЙД & ДЕНЬГИ":      ["💰 Trade Value", "💵 Контракт"],
    "📈 РАЗВИТИЕ & ЗДОРОВЬЕ": ["📈 Развитие", "🏥 Риск травмы"],
    "🏗️ ТАКТИКА":             ["🏗️ Состав"],
    "🧮 МАТЕМАТИКА":          ["🎲 Monte Carlo", "📐 Марковиц", "🧮 Байес", "🕸 Сеть игроков", "⚰️ Выживаемость"],
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.html("""
<div style="text-align:center; padding: 8px 0 16px;">
    <img src="https://cdn.nba.com/logos/leagues/logo-nba.svg" width="54"
         style="filter: drop-shadow(0 2px 8px rgba(200,16,46,0.5));">
    <div style="font-family:'Barlow Condensed','Arial Black',sans-serif;
                font-size:22px; font-weight:900; color:#ffffff;
                letter-spacing:1px; margin-top:8px; text-transform:uppercase;">
        NBA Analytics
    </div>
    <div style="display:inline-block; background:linear-gradient(135deg,#C8102E,#1D428A);
                border-radius:20px; padding:3px 12px; font-size:11px;
                font-weight:700; color:white; letter-spacing:0.5px; margin-top:4px;">
        Сезон 2025-26
    </div>
</div>
""")
st.sidebar.divider()

clusters = ["Все"] + sorted(df["cluster_name"].dropna().unique().tolist())
sel_cluster = st.sidebar.selectbox("🗂 Кластер", clusters)
min_games = st.sidebar.slider("🎮 Мин. игр", 20,
    int(df["games_played"].max()) if ("games_played" in df.columns and not pd.isna(df["games_played"].max())) else 82, 30)
st.sidebar.divider()

import re as _re
def _nav_label(p: str) -> str:
    """Strip leading emoji + space for clean button display."""
    return _re.sub(r'^[\U0001F000-\U0001FFFF\u2600-\u27BF\u2900-\u2BFF]\uFE0F?\u20E3?\s*', '', p).strip()

with st.sidebar:
    for _cat_name, _cat_pages in CATEGORIES.items():
        _cat_active = st.session_state.current_page in _cat_pages
        with st.expander(_cat_name, expanded=_cat_active):
            for _p in _cat_pages:
                _is_active = (st.session_state.current_page == _p)
                if st.button(
                    _nav_label(_p),
                    key=f"nav_{_p}",
                    use_container_width=True,
                    type="primary" if _is_active else "secondary",
                ):
                    st.session_state.current_page = _p
                    st.rerun()

page = st.session_state.current_page

# ── Sidebar footer ─────────────────────────────────────────────────────────────
st.sidebar.html("""
<div style="margin-top:24px;padding:12px 8px 4px;
            border-top:1px solid rgba(200,16,46,0.1);">
    <div style="text-align:center;">
        <div style="font-size:9px;font-weight:800;color:#1e2d42;
                    letter-spacing:2.5px;text-transform:uppercase;">
            NBA Analytics Platform
        </div>
        <div style="font-size:8px;color:#162030;margin-top:3px;font-weight:500;">
            ML · UMAP · XGBoost · v2.0
        </div>
    </div>
</div>
""")

# Фильтрация
df_f = df.copy()
if "games_played" in df_f.columns:
    df_f = df_f[df_f["games_played"] >= min_games]
if sel_cluster != "Все":
    df_f = df_f[df_f["cluster_name"] == sel_cluster]


# ════════════════════════════════════════════════════════════════════════════════
# ОБЗОР ЛИГИ
# ════════════════════════════════════════════════════════════════════════════════
if page == "🏠 Обзор лиги":
    page_header(
        "🏀", "ОБЗОР ЛИГИ",
        "ML Pipeline: PCA → UMAP → K-Means · XGBoost · IsolationForest · 13 модулей",
        badge="Сезон 2025-26", badge_color="#1D428A",
    )
    page_hint([
        "**Что это?** Дашборд-резюме всей лиги за текущий сезон: UMAP-карта всех игроков, KPI, топ-5 по очкам / передачам / подборам.",
        "**Как работает?** Данные загружаются через NBA Stats API → очищаются → 13 признаков нормализуются → UMAP сжимает до 2D для карты · K-Means выдаёт 5 архетипов.",
        "**Зачем?** Быстрый обзор: кто в форме, какие команды доминируют, где аномалии. Фильтры в левой панели сужают обзор до кластера или минимального числа матчей.",
        "**Ключевые термины:** PTS — очки · AST — передачи · REB — подборы · Кластер — архетип игрока по статпрофилю.",
    ])

    kpi_row([
        {"icon": "👥", "label": "Игроков в лиге",  "value": str(len(df_f))},
        {"icon": "🗂", "label": "Кластеров",        "value": str(df_f["cluster_name"].nunique())},
        {"icon": "📊", "label": "Ср. очков/игра",  "value": f"{df_f['PTS'].mean():.1f}" if "PTS" in df_f.columns else "—"},
        {"icon": "🎯", "label": "Ср. передач/игра","value": f"{df_f['AST'].mean():.1f}" if "AST" in df_f.columns else "—"},
        {"icon": "💪", "label": "Ср. подборов",    "value": f"{df_f['REB'].mean():.1f}" if "REB" in df_f.columns else "—"},
    ])

    st.plotly_chart(umap_scatter(df_f), use_container_width=True, key="umap_main")

    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(pts_vs_ast(df_f), use_container_width=True, key="pts_ast_main")
    with col_r:
        if len(df_f) > 0:
            target = (
                sel_cluster if sel_cluster != "Все"
                else df_f.groupby("cluster_name")["PTS"].mean().idxmax()
            )
            st.plotly_chart(top_players_bar(df_f, target), use_container_width=True, key="top_bar_main")
        else:
            st.info("Нет данных для выбранных фильтров — измените параметры в боковой панели.")

    st.divider()
    section_title("Лидеры по категориям", "⭐")
    cats = [("PTS","🏆 Очки","#C8102E"), ("AST","🎯 Передачи","#1D428A"),
            ("REB","💪 Подборы","#27AE60"), ("STL","🤚 Перехваты","#9B59B6"),
            ("BLK","🛡 Блок-шоты","#E67E22")]
    cols = st.columns(len(cats))
    for col, (stat, label, color) in zip(cols, cats):
        if stat in df_f.columns:
            top = df_f.nlargest(5, stat)[[name_col, stat]]
            rows_html = ""
            for rank, (_, r) in enumerate(top.iterrows(), 1):
                opacity = 1.0 - (rank - 1) * 0.15
                rows_html += (
                    f"<div style='display:flex;align-items:center;gap:8px;"
                    f"padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04);opacity:{opacity:.2f}'>"
                    f"<span style='font-size:10px;color:{color};font-weight:800;"
                    f"min-width:14px;text-align:right'>#{rank}</span>"
                    f"<span style='font-family:\"Barlow Condensed\",sans-serif;font-size:13px;"
                    f"font-weight:700;color:#c8d0e0;flex:1;white-space:nowrap;"
                    f"overflow:hidden;text-overflow:ellipsis'>{r[name_col]}</span>"
                    f"<span style='font-family:\"Barlow Condensed\",sans-serif;font-size:15px;"
                    f"font-weight:900;color:#fff'>{r[stat]:.1f}</span>"
                    f"</div>"
                )
            col.html(
                f"<div style='background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);"
                f"border-top:2px solid {color};border-radius:10px;padding:12px 14px;'>"
                f"<div style='font-size:11px;font-weight:800;color:{color};"
                f"text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px'>{label}</div>"
                f"{rows_html}</div>"
            )


# ════════════════════════════════════════════════════════════════════════════════
# КАРТОЧКА ИГРОКА
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🏀 Карточка игрока":
    page_header(
        "🏀", "КАРТОЧКА ИГРОКА",
        "Полная статистика · Shot Chart · Trade Value · AI скаутинг-отчёт",
        badge="Профиль", badge_color="#1D428A",
    )
    page_hint([
        "**Что это?** Полный профиль игрока: статистика, процентильный ранк по лиге, radar-диаграмма, схема бросков, Trade Value, Injury Risk и AI-скаутинг.",
        "**Как работает?** Выберите игрока в левой панели. Все показатели сравниваются с медианой лиги; цвет — зелёный (топ-25%), жёлтый (50%), красный (низ 25%).",
        "**Зачем?** Один экран заменяет 30–60 минут ручного анализа скаута: сильные стороны, зоны роста, риск, ценность для трейда и AI-вердикт.",
        "**Ключевые термины:** Процентиль — позиция в распределении по лиге (90° = лучше 90% игроков) · TV — Trade Value 0–100 · IR — Injury Risk 0–100.",
    ])

    player_name = st.selectbox("Выбери игрока", sorted(df[name_col].dropna().unique()))

    if player_name:
        row = df[df[name_col] == player_name].iloc[0]
        league_df = df.copy()

        # ── Шапка ─────────────────────────────────────────────────────────────
        col_photo, col_info, col_radar = st.columns([1, 1.2, 2])

        with col_photo:
            pid = row.get("PLAYER_ID", None)
            if pid and not pd.isna(pid):
                player_img(pid, width=200)
            else:
                st.markdown("### 👤")

            st.html(f"""
            <div style="background:linear-gradient(135deg,#1d428a,#c8102e);
                        border-radius:12px;padding:14px;color:white;text-align:center;margin-top:8px">
                <div style="font-size:18px;font-weight:700">{player_name}</div>
                <div style="font-size:13px;opacity:0.85">{row.get('team','—')}</div>
                <div style="display:inline-block;background:rgba(255,255,255,0.2);
                            border-radius:20px;padding:3px 12px;font-size:12px;margin-top:6px">
                    {row['cluster_name']}
                </div>
            </div>""")

        with col_info:
            st.markdown("#### 📋 Статистика сезона")
            stats = [
                ("Очки/игра",     row.get("PTS", 0),        "PTS",       ".1f", True),
                ("Передачи/игра", row.get("AST", 0),        "AST",       ".1f", True),
                ("Подборы/игра",  row.get("REB", 0),        "REB",       ".1f", True),
                ("Перехваты/игра",row.get("STL", 0),        "STL",       ".1f", True),
                ("Блок-шоты/игра",row.get("BLK", 0),        "BLK",       ".1f", True),
                ("% бросков",     row.get("FG_PCT",0)*100,  "FG_PCT",    ".1f", True),
                ("% трёхочков",   row.get("FG3_PCT",0)*100, "FG3_PCT",   ".1f", True),
                ("USG%",          row.get("USG_PCT",0)*100, "USG_PCT",   ".1f", True),
                ("+/-",           row.get("PLUS_MINUS",0),  "PLUS_MINUS","+.1f",True),
                ("Потери/игра",   row.get("TOV", 0),        "TOV",       ".1f", False),
            ]

            html = ""
            for label, val, col_key, fmt, higher in stats:
                if col_key in league_df.columns:
                    src = league_df[col_key]
                    if col_key in ["FG_PCT","FG3_PCT","FT_PCT","USG_PCT"]:
                        src = src * 100
                    p = percentile_of(src, val)
                    if not higher: p = 100 - p
                    html += stat_bar_html(label, val, p, fmt)
            st.html(html)
            st.caption("Перцентиль относительно всех игроков лиги")

        with col_radar:
            st.plotly_chart(radar_chart(df, row["cluster_name"]),
                            use_container_width=True, key=f"radar_player_{player_name}")

        # ── Вкладки: Shot Chart / Trade Value / Похожие / AI ─────────────────
        st.divider()
        tab_shot, tab_tv, tab_sim, tab_ai = st.tabs(
            ["🏀 Карта бросков", "💰 Trade Value", "🔍 Похожие", "🤖 AI Отчёт"]
        )

        with tab_shot:
            from src.shot_chart import shot_chart
            sc_mode = st.radio("Режим отображения",
                               ["🌡 Тепловая карта", "⬡ Гексагоны", "🎯 Точки попаданий"],
                               horizontal=True, key="sc_mode_player")
            mode = "heat" if "Тепловая" in sc_mode else ("hex" if "Гексагоны" in sc_mode else "dots")
            n_shots = st.slider("Симулированных бросков", 100, 600, 350, 50,
                                key="sc_n_player")
            fig_sc = shot_chart(row, mode=mode, n_shots=n_shots)
            st.plotly_chart(fig_sc, use_container_width=False, key=f"shot_chart_{player_name}")
            st.caption(
                "⚠️ Карта бросков синтетическая — построена на основе статистики "
                "и архетипа игрока. Реальные координаты бросков доступны через "
                "NBA ShotChartDetail API при наличии сети."
            )

        with tab_tv:
            from src.trade_value import compute_trade_value, plot_trade_breakdown
            df_tv = get_trade_values(_data_hash)
            row_tv = df_tv[df_tv[name_col] == player_name].iloc[0]
            tv_val = float(row_tv["trade_value"])
            tv_tier = str(row_tv["tv_tier"])

            tier_colors = {
                "💎 Франчайз": "#1d428a",
                "⭐ Топ-15":   "#c8102e",
                "🔵 Стартер":  "#2980b9",
                "🟡 Ролевой":  "#f39c12",
                "⚪ Резерв":   "#95a5a6",
            }
            badge_color = tier_colors.get(tv_tier, "#95a5a6")

            tv_c1, tv_c2, tv_c3 = st.columns(3)
            tv_c1.metric("💰 Trade Value", f"{tv_val:.1f} / 100")
            tv_c2.html(
                f"<div style='padding:10px'>"
                f"<div style='font-size:12px;color:#666'>Тир</div>"
                f"<div style='background:{badge_color};color:white;border-radius:20px;"
                f"padding:4px 14px;display:inline-block;font-weight:700;font-size:15px'>"
                f"{tv_tier}</div></div>"
            )
            rank = int((df_tv["trade_value"] > tv_val).sum()) + 1
            tv_c3.metric("📊 Ранг в лиге", f"#{rank} из {len(df_tv)}")

            fig_tv = plot_trade_breakdown(row_tv)
            st.plotly_chart(fig_tv, use_container_width=True, key=f"tv_breakdown_{player_name}")

        with tab_sim:
            similar = find_similar(df, player_name, n=6)
            sim_cols = st.columns(3)
            for i, (_, sr) in enumerate(similar.iterrows()):
                with sim_cols[i % 3]:
                    pid2 = df[df[name_col] == sr[name_col]]["PLAYER_ID"].values
                    if len(pid2) > 0 and not pd.isna(pid2[0]):
                        player_img(pid2[0], width=80, radius=8)
                    pts = sr.get("PTS", 0)
                    ast = sr.get("AST", 0)
                    reb = sr.get("REB", 0)
                    st.html(f"""
                    <div class="similar-card">
                        <b>{sr[name_col]}</b><br>
                        <span style="color:#888;font-size:12px">{sr.get('team','—')} · {sr['cluster_name']}</span><br>
                        <span style="font-size:13px">{pts:.1f} pts · {ast:.1f} ast · {reb:.1f} reb</span>
                    </div>""")

        with tab_ai:
            section_title("Скаутинг-отчёт (Gemini)", "🤖")
            cluster_avg = df[df["cluster_name"] == row["cluster_name"]].mean(numeric_only=True)
            # Trade value context
            df_tv2 = get_trade_values(_data_hash)
            row_tv2 = df_tv2[df_tv2[name_col] == player_name].iloc[0]

            if st.button("✨ Сгенерировать отчёт", type="primary", key="ai_btn_player"):
                try:
                    client = google_genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                    prompt = f"""Ты — ведущий NBA аналитик и скаут. Напиши профессиональный скаутинг-отчёт.

Игрок: {player_name}
Команда: {row.get('team','—')} | Тип: {row['cluster_name']} | Игр: {row.get('games_played','—')}
Trade Value: {row_tv2.get('trade_value',0):.1f}/100 ({row_tv2.get('tv_tier','—')})

Статистика 2025-26 (per game):
• Очки: {row.get('PTS',0):.1f} | Передачи: {row.get('AST',0):.1f} | Подборы: {row.get('REB',0):.1f}
• Перехваты: {row.get('STL',0):.1f} | Блок-шоты: {row.get('BLK',0):.1f} | Потери: {row.get('TOV',0):.1f}
• FG%: {row.get('FG_PCT',0)*100:.1f}% | 3P%: {row.get('FG3_PCT',0)*100:.1f}% | FT%: {row.get('FT_PCT',0)*100:.1f}%
• USG%: {row.get('USG_PCT',0)*100:.1f}% | +/-: {row.get('PLUS_MINUS',0):+.1f} | TS%: {row.get('TS_PCT',0)*100:.1f}%

Средние по кластеру "{row['cluster_name']}":
• Очки: {cluster_avg.get('PTS',0):.1f} | Передачи: {cluster_avg.get('AST',0):.1f} | Подборы: {cluster_avg.get('REB',0):.1f}

Структура отчёта:
**1. Сильные стороны** — что делает игрока ценным
**2. Зоны роста** — слабые стороны и как их устранить
**3. Роль в команде** — идеальная роль, с какими игроками сочетается
**4. Рыночная оценка** — трансферная ценность, контракт

Пиши по-русски, кратко и точно. Используй NBA-терминологию. Без вводных фраз."""

                    with st.spinner("Анализируем..."):
                        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                        report = resp.text

                    st.markdown(report)
                    st.download_button("📥 Скачать отчёт", report,
                        file_name=f"scout_{player_name.replace(' ','_')}.txt",
                        mime="text/plain", key="dl_btn_ai")

                except KeyError:
                    st.error("GEMINI_API_KEY не найден в .streamlit/secrets.toml")
                except Exception as e:
                    st.error(f"Ошибка: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# HEAD-TO-HEAD
# ════════════════════════════════════════════════════════════════════════════════
elif page == "⚔️ Head-to-Head":
    page_header(
        "⚔️", "HEAD-TO-HEAD",
        "Прямое сравнение двух игроков по всем показателям · Radar · Shot Charts",
        badge="Дуэль", badge_color="#9B59B6",
    )
    page_hint([
        "**Что это?** Прямое сравнение двух игроков: stat-таблица с цветовой разметкой победителя по каждому показателю, radar и схемы бросков.",
        "**Как работает?** Зелёный 🟢 = лучше по данному показателю, красный 🔴 = хуже. Общий счёт считается по сумме «зелёных» метрик.",
        "**Зачем?** Принять решение о трейде: кто из двух игроков объективно сильнее, в чём компромисс, кому выгоднее обмен.",
        "**Совет:** Используйте фильтр «Кластер» в боковой панели, чтобы сравнивать игроков одного архетипа — это честнее, чем сравнивать центрового с разыгрывающим.",
    ])

    all_players = sorted(df[name_col].dropna().unique())
    col_a, col_b = st.columns(2)
    with col_a:
        p1 = st.selectbox("🔵 Игрок 1", all_players, index=0)
    with col_b:
        p2 = st.selectbox("🔴 Игрок 2", all_players, index=min(1, len(all_players)-1))

    if p1 and p2 and p1 != p2:
        r1 = df[df[name_col] == p1].iloc[0]
        r2 = df[df[name_col] == p2].iloc[0]

        # Фото + карточки
        fc1, fc2 = st.columns(2)
        for col, row, color, pname in [(fc1,r1,"#1d428a",p1),(fc2,r2,"#c8102e",p2)]:
            with col:
                pid = row.get("PLAYER_ID", None)
                if pid and not pd.isna(pid):
                    player_img(pid, width=160)
                st.html(f"""
                <div style="background:{color};border-radius:10px;padding:12px;
                            color:white;text-align:center">
                    <b style="font-size:16px">{pname}</b><br>
                    <span style="font-size:13px">{row.get('team','—')}</span><br>
                    <span style="font-size:12px;opacity:0.8">{row['cluster_name']}</span>
                </div>""")

        st.divider()

        # Trade Value сравнение
        df_tv = get_trade_values(_data_hash)
        tv1 = df_tv[df_tv[name_col] == p1].iloc[0] if len(df_tv[df_tv[name_col] == p1]) > 0 else None
        tv2 = df_tv[df_tv[name_col] == p2].iloc[0] if len(df_tv[df_tv[name_col] == p2]) > 0 else None
        if tv1 is not None and tv2 is not None:
            tvc1, tvc_mid, tvc2 = st.columns([2, 1, 2])
            with tvc1:
                st.metric(f"💰 Trade Value — {p1}", f"{tv1['trade_value']:.1f}", tv1["tv_tier"])
            with tvc_mid:
                st.html("<div style='margin:8px 0;text-align:center;font-size:24px'>⚖️</div>")
            with tvc2:
                st.metric(f"💰 Trade Value — {p2}", f"{tv2['trade_value']:.1f}", tv2["tv_tier"])

        # Таблица сравнения
        compare_stats = [
            ("Очки/игра",    "PTS",       ".1f", True),
            ("Передачи/игра","AST",       ".1f", True),
            ("Подборы/игра", "REB",       ".1f", True),
            ("Перехваты",    "STL",       ".1f", True),
            ("Блок-шоты",    "BLK",       ".1f", True),
            ("FG%",          "FG_PCT",    ".1%", True),
            ("3P%",          "FG3_PCT",   ".1%", True),
            ("FT%",          "FT_PCT",    ".1%", True),
            ("USG%",         "USG_PCT",   ".1%", True),
            ("+/-",          "PLUS_MINUS","+.1f",True),
            ("Потери",       "TOV",       ".1f", False),
            ("TS%",          "TS_PCT",    ".1%", True),
        ]

        rows_data = []
        for label, stat, fmt, higher in compare_stats:
            if stat in df.columns:
                v1 = r1.get(stat, 0) or 0
                v2 = r2.get(stat, 0) or 0
                icon = compare_stat(v1, v2, higher)
                rows_data.append({
                    p1: f"{v1:{fmt}}",
                    "": icon,
                    "Показатель": label,
                    " ": icon,
                    p2: f"{v2:{fmt}}",
                })

        cmp_df = pd.DataFrame(rows_data).set_index("Показатель")
        st.dataframe(cmp_df, use_container_width=True)

        # Вкладки: Radar / Shot Charts
        tab_rad, tab_sc = st.tabs(["📊 Radar-профили", "🏀 Карты бросков"])

        with tab_rad:
            rc1, rc2 = st.columns(2)
            with rc1:
                st.plotly_chart(radar_chart(df, r1["cluster_name"]),
                                use_container_width=True, key=f"radar_h2h_p1_{p1}")
            with rc2:
                st.plotly_chart(radar_chart(df, r2["cluster_name"]),
                                use_container_width=True, key=f"radar_h2h_p2_{p2}")

        with tab_sc:
            from src.shot_chart import shot_chart
            sc1, sc2 = st.columns(2)
            with sc1:
                st.plotly_chart(shot_chart(r1, mode="heat", n_shots=300),
                                use_container_width=False, key=f"sc_h2h_{p1}")
            with sc2:
                st.plotly_chart(shot_chart(r2, mode="heat", n_shots=300),
                                use_container_width=False, key=f"sc_h2h_{p2}")


# ════════════════════════════════════════════════════════════════════════════════
# ЛИДЕРЫ
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Лидеры":
    page_header(
        "🏆", "ЛИДЕРЫ ЛИГИ",
        "Топ игроков по категориям · Очки · Передачи · Подборы · Защита · Эффективность",
        badge="2025-26", badge_color="#FFC72C",
        accent="#FFC72C",
    )
    page_hint([
        "**Что это?** Таблица лидеров лиги по восьми статистическим категориям с интерактивной сортировкой.",
        "**Как работает?** Данные ранжируются по выбранному показателю; подсвечиваются топ-10. Фильтр «Мин. игр» исключает игроков с маленькой выборкой.",
        "**Зачем?** Быстро найти лучших на рынке по нужной метрике — при поиске снайпера, плеймейкера или защитника.",
        "**Ключевые термины:** TS% — True Shooting % (реальная эффективность броска) · USG% — процент владений, завершённых игроком · +/- — вклад в разницу очков на площадке.",
    ])

    tab_pts, tab_ast, tab_reb, tab_def, tab_eff = st.tabs(
        ["🏀 Очки", "🎯 Передачи", "💪 Подборы", "🛡 Защита", "⚡ Эффективность"]
    )

    def leader_table(stat, n=15):
        cols = [name_col, "team", "cluster_name", "games_played", stat]
        cols = [c for c in cols if c in df_f.columns]
        return df_f.nlargest(n, stat)[cols].reset_index(drop=True)

    with tab_pts:
        t = leader_table("PTS"); t.index += 1
        st.dataframe(t, use_container_width=True)

    with tab_ast:
        t = leader_table("AST"); t.index += 1
        st.dataframe(t, use_container_width=True)

    with tab_reb:
        t = leader_table("REB"); t.index += 1
        st.dataframe(t, use_container_width=True)

    with tab_def:
        if "STL" in df_f.columns and "BLK" in df_f.columns:
            df_def = df_f.copy()
            df_def["STL+BLK"] = df_def["STL"] + df_def["BLK"]
            show_cols = [c for c in [name_col, "team", "cluster_name", "games_played",
                                      "STL", "BLK", "STL+BLK"] if c in df_def.columns]
            t = df_def.nlargest(15, "STL+BLK")[show_cols].reset_index(drop=True)
            t.index += 1
            st.dataframe(t, use_container_width=True)

    with tab_eff:
        if "efficiency" in df_f.columns:
            t = leader_table("efficiency"); t.index += 1
            st.dataframe(t, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# КЛАСТЕРЫ
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📊 Кластеры":
    page_header(
        "📊", "АНАЛИЗ КЛАСТЕРОВ",
        "K-Means · PCA · UMAP · Архетипы игроков · Radar-профили кластеров",
        badge="ML Clustering", badge_color="#3498DB",
        accent="#3498DB",
    )
    page_hint([
        "**Что это?** Визуализация пяти архетипов (кластеров) игроков: UMAP-карта, средний статпрофиль каждого кластера, radar-диаграммы.",
        "**Как работает?** 13 признаков → StandardScaler → UMAP (2D) → K-Means (k=5, силуэт > 0.65). Кластер — это не позиция, а *функциональный архетип*: «звезда», «плеймейкер», «3-and-D» и т.д.",
        "**Зачем?** Найти похожих игроков для замены на трейде, понять, какой архетип нужен команде, и выявить нетипичных игроков, не вписывающихся в шаблон.",
        "**Ключевые термины:** Силуэт — метрика качества кластеризации (1 = идеально) · UMAP — Uniform Manifold Approximation (нелинейное снижение размерности).",
    ])

    show_cols = [c for c in ["PTS","AST","REB","STL","BLK","FG_PCT","FG3_PCT",
                              "USG_PCT","PLUS_MINUS"] if c in df.columns]
    cluster_stats = df.groupby("cluster_name")[show_cols].mean().round(2)

    section_title("Средние показатели по кластерам", "📋")
    st.dataframe(cluster_stats.style.background_gradient(cmap="Blues"),
                 use_container_width=True)

    section_title("Распределение игроков", "📊")
    sizes = df.groupby("cluster_name").size().reset_index(name="Игроков")
    import plotly.express as _px2
    _fig_sizes = _px2.bar(
        sizes.sort_values("Игроков", ascending=False),
        x="cluster_name", y="Игроков",
        color="Игроков", color_continuous_scale=["#1D428A", "#C8102E"],
        template="nba_dark", height=320,
        labels={"cluster_name": "Кластер"},
    )
    _fig_sizes.update_layout(coloraxis_showscale=False, xaxis_title="", showlegend=False)
    st.plotly_chart(_fig_sizes, use_container_width=True, key="cluster_sizes_bar")

    section_title("Radar-профили кластеров", "🎯")
    selected = st.multiselect("Выбери кластеры",
        df["cluster_name"].dropna().unique().tolist(),
        default=list(df["cluster_name"].dropna().unique())[:3])
    rcols = st.columns(min(len(selected), 3)) if selected else []
    for i, cl in enumerate(selected):
        with rcols[i % 3]:
            st.plotly_chart(radar_chart(df, cl), use_container_width=True,
                            key=f"radar_cluster_{i}_{cl}")


# ════════════════════════════════════════════════════════════════════════════════
# АНОМАЛИИ — IsolationForest
# ════════════════════════════════════════════════════════════════════════════════
elif page == "💎 Аномалии":
    page_header(
        "💎", "АНОМАЛИИ",
        "IsolationForest · contamination=8% · Недооценённые и переоценённые игроки",
        badge="AI Detection", badge_color="#27AE60",
        accent="#27AE60",
    )
    page_hint([
        "**Что это?** Детектор статистических аномалий: «скрытые жемчужины» (сильные игроки с низкой зарплатой/ценой) и «переоценённые» (слабая реальная польза при высоком имидже).",
        "**Как работает?** IsolationForest обучается на 13 признаках и помечает ~8% игроков как «аномальных». Жемчужина = аномалия с высокой реальной результативностью. Переоценённый = высокий контракт при низком вкладе.",
        "**Зачем?** Найти игроков с рыночным неэффективностью — классическая задача Moneyball. Аномалии IsolationForest часто коррелируют с будущими прорывами или откатами.",
        "**Ключевые термины:** Anomaly Score — чем ниже (отрицательнее), тем «аномальнее» игрок · contamination=8% — ожидаемая доля аномалий в данных.",
    ])

    df_a, gems, over = get_anomalies(_data_hash)

    # Метрики
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🔍 Аномалий всего",
              int((df_a["anomaly_label"] != "normal").sum()))
    m2.metric("💎 Недооценённых",
              int((df_a["anomaly_label"] == "underrated").sum()))
    m3.metric("⚠️ Переоценённых",
              int((df_a["anomaly_label"] == "overrated").sum()))
    m4.metric("✅ Нормальных",
              int((df_a["anomaly_label"] == "normal").sum()))

    tab_gems, tab_over, tab_scatter = st.tabs(
        ["💎 Hidden Gems", "⚠️ Переоценённые", "🗺 Карта аномалий"]
    )

    with tab_gems:
        section_title("Топ недооценённых игроков", "💎", color="#27AE60")
        st.markdown(
            "_Высокая эффективность при низком использовании — "
            "кандидаты на повышение роли или выгодный трейд._"
        )
        if not gems.empty:
            # Цветная таблица
            st.dataframe(
                gems.reset_index(drop=True).style
                    .background_gradient(subset=["anomaly_score"], cmap="Greens")
                    .background_gradient(subset=["PTS"] if "PTS" in gems.columns else [], cmap="Blues")
                    .format({"anomaly_score": "{:.1f}", "PTS": "{:.1f}",
                             "AST": "{:.1f}", "REB": "{:.1f}",
                             "efficiency": "{:.1f}", "USG_PCT": "{:.3f}"}),
                use_container_width=True,
            )

            # Bar chart топ-15
            import plotly.express as px
            fig_gems = px.bar(
                gems.head(15).sort_values("anomaly_score"),
                x="anomaly_score", y=name_col if name_col in gems.columns else gems.columns[0],
                orientation="h", color="anomaly_score",
                color_continuous_scale=[[0.0,"#0d2b0d"],[0.5,"#27AE60"],[1.0,"#2ECC71"]],
                title="Топ недооценённых (Anomaly Score)",
                labels={"anomaly_score": "Anomaly Score", name_col: "Игрок"},
                template="nba_dark", height=420,
            )
            fig_gems.update_coloraxes(showscale=False)
            st.plotly_chart(fig_gems, use_container_width=True, key="gems_bar")
        else:
            st.info("Нет игроков-аномалий в выбранном наборе данных.")

    with tab_over:
        section_title("Топ переоценённых игроков", "⚠️", color="#E74C3C")
        st.markdown(
            "_Высокое использование при низкой эффективности — "
            "риск при трейде или перегруженность роли._"
        )
        if not over.empty:
            st.dataframe(
                over.reset_index(drop=True).style
                    .background_gradient(subset=["anomaly_score"], cmap="Reds")
                    .format({"anomaly_score": "{:.1f}", "PTS": "{:.1f}",
                             "USG_PCT": "{:.3f}", "efficiency": "{:.1f}"}),
                use_container_width=True,
            )

            import plotly.express as px
            fig_over = px.bar(
                over.sort_values("anomaly_score"),
                x="anomaly_score", y=name_col if name_col in over.columns else over.columns[0],
                orientation="h", color="anomaly_score",
                color_continuous_scale=[[0.0,"#2b0d0d"],[0.5,"#C0392B"],[1.0,"#E74C3C"]],
                title="Переоценённые (Anomaly Score)",
                template="nba_dark", height=380,
            )
            fig_over.update_coloraxes(showscale=False)
            st.plotly_chart(fig_over, use_container_width=True, key="over_bar")
        else:
            st.info("Переоценённых не обнаружено.")

    with tab_scatter:
        section_title("UMAP — карта аномалий", "🗺")
        import plotly.express as px
        color_map = {"underrated": "#27ae60", "overrated": "#e74c3c", "normal": "#bdc3c7"}
        size_map  = {"underrated": 10, "overrated": 10, "normal": 5}

        df_plot = df_a.copy()
        df_plot["_size"] = df_plot["anomaly_label"].map(size_map).fillna(5)

        if "umap_x" in df_plot.columns and "umap_y" in df_plot.columns:
            fig_sc = px.scatter(
                df_plot, x="umap_x", y="umap_y",
                color="anomaly_label",
                hover_name=name_col,
                hover_data={"anomaly_score": ":.1f", "PTS": ":.1f",
                             "umap_x": False, "umap_y": False},
                color_discrete_map=color_map,
                size="_size", size_max=14,
                title="UMAP — Normal / Underrated / Overrated",
                labels={"anomaly_label": "Статус"},
                template="nba_dark", height=480,
            )
            st.plotly_chart(fig_sc, use_container_width=True, key="anomaly_umap")
        else:
            st.warning("UMAP-координаты не найдены в данных.")

        # Anomaly score распределение
        import plotly.express as px
        fig_hist = px.histogram(
            df_a, x="anomaly_score", color="anomaly_label",
            color_discrete_map=color_map,
            nbins=30, barmode="overlay", opacity=0.7,
            title="Распределение Anomaly Score",
            template="nba_dark", height=320,
        )
        st.plotly_chart(fig_hist, use_container_width=True, key="anomaly_hist")


# ════════════════════════════════════════════════════════════════════════════════
# ПРОГНОЗ — XGBoost / GradientBoosting
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Прогноз":
    page_header(
        "🔮", "ПРОГНОЗ СЕЗОНА 2026-27",
        "XGBoost · GradientBoosting fallback · 5-fold CV · Возрастные коэффициенты роста/спада",
        badge="ML Forecast",
    )
    page_hint([
        "**Что это?** ML-прогноз результативности (PTS) каждого игрока на сезон 2026-27 с учётом возраста и нагрузки.",
        "**Как работает?** XGBoost Regressor (300 деревьев, max_depth=4, lr=0.05) обучается на 13 признаках. 5-fold кросс-валидация гарантирует отсутствие утечки данных. Для игроков <22 и >34 лет применяется поправочный коэффициент.",
        "**Зачем?** Контракт стоит подписывать игроку на подъёме, а не спаде. Прогноз + возраст + CV даёт полную картину для решения GM.",
        "**Ключевые термины:** MAE = 0.69 — средняя ошибка в очках · R² = 0.977 — доля объяснённой дисперсии (1.0 = идеально) · Feature importance — вклад каждого признака в прогноз.",
    ])

    try:
        model, scaler, feats, metrics, y_cv, df_pred = get_prediction_model(_data_hash)
        from src.prediction import (plot_actual_vs_predicted,
                                    plot_feature_importance,
                                    plot_next_season_forecast)

        # Метрики модели
        m1, m2, m3 = st.columns(3)
        m1.metric("📉 MAE (ошибка)", f"{metrics['mae']:.2f} pts")
        m2.metric("📈 R² (качество)", f"{metrics['r2']:.3f}")
        m3.metric("🔢 Признаков", len(metrics["features"]))

        # Интерпретируемость качества
        r2 = metrics["r2"]
        quality_txt = "Отличная" if r2 >= 0.85 else "Хорошая" if r2 >= 0.70 else "Средняя"
        quality_color = "#27ae60" if r2 >= 0.85 else "#f39c12" if r2 >= 0.70 else "#e74c3c"
        st.html(
            f"<div style='background:{quality_color}22;border-left:4px solid {quality_color};"
            f"padding:10px 16px;border-radius:6px;margin:8px 0'>"
            f"<b>Качество модели: {quality_txt}</b> — R²={r2:.3f} означает, что модель объясняет "
            f"{r2*100:.1f}% дисперсии очков. MAE = {metrics['mae']:.2f} очка в среднем.</div>"
        )

        tab_cv, tab_imp, tab_forecast, tab_table = st.tabs(
            ["📊 Кросс-валидация", "🎯 Важность признаков",
             "📈 Прогноз сезона", "📋 Таблица прогнозов"]
        )

        with tab_cv:
            section_title("Фактические vs Предсказанные (CV 5-fold)", "📊")
            y_true = df["PTS"].fillna(0).values
            names  = df[name_col].values
            fig_cv = plot_actual_vs_predicted(y_true, y_cv, names)
            st.plotly_chart(fig_cv, use_container_width=True, key="cv_scatter")
            st.caption(
                "Каждая точка — игрок. Серая линия — идеальное предсказание. "
                "Цвет: зелёный = точное, красный = ошибка. "
                "Оценка проводится вне обучающих данных (кросс-валидация)."
            )

        with tab_imp:
            section_title("Важность признаков для прогноза очков", "🎯")
            fig_imp = plot_feature_importance(model, feats)
            if fig_imp:
                st.plotly_chart(fig_imp, use_container_width=True, key="feat_imp")
                st.caption(
                    "Feature Importance (XGBoost/GB). Высокое значение = "
                    "признак сильно влияет на прогноз очков."
                )
            else:
                st.info("График важности признаков недоступен для данного типа модели.")

        with tab_forecast:
            section_title("Прогноз результативности на следующий сезон", "📈")
            top_n = st.slider("Топ-N игроков", 10, 40, 20, key="forecast_top_n")
            fig_f = plot_next_season_forecast(df_pred, top_n=top_n)
            st.plotly_chart(fig_f, use_container_width=True, key="forecast_bar")

            # Рост / спад
            if "pred_delta" in df_pred.columns:
                col_grow, col_drop = st.columns(2)
                with col_grow:
                    st.markdown("**🚀 Наибольший рост**")
                    g = df_pred.nlargest(8, "pred_delta")[[name_col, "PTS", "pred_pts", "pred_delta"]]
                    st.dataframe(g.reset_index(drop=True).style
                                   .background_gradient(subset=["pred_delta"], cmap="Greens"),
                                 use_container_width=True)
                with col_drop:
                    st.markdown("**📉 Наибольший спад**")
                    d = df_pred.nsmallest(8, "pred_delta")[[name_col, "PTS", "pred_pts", "pred_delta"]]
                    st.dataframe(d.reset_index(drop=True).style
                                   .background_gradient(subset=["pred_delta"], cmap="Reds_r"),
                                 use_container_width=True)

        with tab_table:
            section_title("Полная таблица прогнозов", "📋")
            show_cols = [c for c in [name_col, "team", "cluster_name",
                                      "PTS", "pred_pts", "pred_low",
                                      "pred_high", "pred_delta"] if c in df_pred.columns]
            tbl = df_pred[show_cols].sort_values("pred_pts", ascending=False).reset_index(drop=True)
            tbl.index += 1
            st.dataframe(
                tbl.style
                   .background_gradient(subset=["pred_pts"], cmap="Blues")
                   .background_gradient(subset=["pred_delta"], cmap="RdYlGn", vmin=-5, vmax=5)
                   .format({col: "{:.1f}" for col in ["PTS","pred_pts","pred_low",
                                                       "pred_high","pred_delta"]
                             if col in tbl.columns}),
                use_container_width=True,
            )
            # Поиск игрока
            st.divider()
            pred_player = st.selectbox("🔍 Найти прогноз игрока",
                                        ["—"] + sorted(df_pred[name_col].dropna().tolist()),
                                        key="pred_player_search")
            if pred_player != "—":
                pr = df_pred[df_pred[name_col] == pred_player].iloc[0]
                pc1, pc2, pc3, pc4 = st.columns(4)
                pc1.metric("Текущий сезон", f"{pr.get('PTS',0):.1f} pts")
                pc2.metric("Прогноз", f"{pr.get('pred_pts',0):.1f} pts",
                           f"{pr.get('pred_delta',0):+.1f}")
                pc3.metric("Мин. прогноз", f"{pr.get('pred_low',0):.1f} pts")
                pc4.metric("Макс. прогноз", f"{pr.get('pred_high',0):.1f} pts")

    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        st.info("Убедись, что данные загружены: `python src/fetch_data.py && python src/clustering.py`")


# ════════════════════════════════════════════════════════════════════════════════
# TRADE VALUE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "💰 Trade Value":
    page_header(
        "💰", "TRADE VALUE",
        "Составная формула: результативность · защита · эффективность · возраст/потенциал · надёжность",
        badge="Трансферный рынок", badge_color="#E67E22",
        accent="#E67E22",
    )
    page_hint([
        "**Что это?** Индекс рыночной ценности игрока для трейда (0–100) на основе взвешенной формулы из пяти компонентов.",
        "**Как работает?** TV = Результативность (35%) + Защита (20%) + Эффективность (25%) + Возраст/потенциал (15%) + Надёжность (5%). Каждый компонент нормализован к диапазону своего веса.",
        "**Зачем?** Объективно оценить «рыночный курс» игрока перед трейдом. TV помогает понять, кто в обмене получает больше ценности.",
        "**Ключевые термины:** TV ≥ 70 — элитный · TV 50–70 — стартовый 5 · TV < 30 — ролевой/ротационный · ΔTV = TV отдаёшь − TV получаешь (отрицательный = ты в выигрыше).",
    ])

    from src.trade_value import (plot_trade_leaderboard, plot_value_vs_age,
                                  plot_trade_tiers)

    df_tv = get_trade_values(_data_hash)

    # Метрики
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("💎 Франчайз", int((df_tv["tv_tier"] == "💎 Франчайз").sum()))
    t2.metric("⭐ Топ-15",   int((df_tv["tv_tier"] == "⭐ Топ-15").sum()))
    t3.metric("🔵 Стартер",  int((df_tv["tv_tier"] == "🔵 Стартер").sum()))
    t4.metric("🟡 Ролевой",  int((df_tv["tv_tier"] == "🟡 Ролевой").sum()))
    t5.metric("⚪ Резерв",   int((df_tv["tv_tier"] == "⚪ Резерв").sum()))

    tab_board, tab_age, tab_tiers, tab_all = st.tabs(
        ["🏆 Лидерборд", "📈 Возраст vs Value", "🥧 Тиры", "📋 Все игроки"]
    )

    with tab_board:
        top_n_tv = st.slider("Показать топ-N", 10, 50, 25, key="tv_top_n")
        # Применяем фильтр кластера если нужно
        df_tv_f = df_tv.copy()
        if sel_cluster != "Все":
            df_tv_f = df_tv_f[df_tv_f["cluster_name"] == sel_cluster] if "cluster_name" in df_tv_f.columns else df_tv_f
        if "games_played" in df_tv_f.columns:
            df_tv_f = df_tv_f[df_tv_f["games_played"] >= min_games]

        fig_lb = plot_trade_leaderboard(df_tv_f, top_n=top_n_tv)
        st.plotly_chart(fig_lb, use_container_width=True, key="tv_leaderboard")

    with tab_age:
        fig_age = plot_value_vs_age(df_tv)
        if fig_age:
            st.plotly_chart(fig_age, use_container_width=True, key="tv_age_scatter")
            st.caption(
                "Размер точки — очки за игру. "
                "Синяя зона — пиковый возраст (25–30 лет). "
                "Ищи молодых игроков с высоким Trade Value — потенциальные звёзды."
            )
        else:
            st.warning("Колонка возраста не найдена.")

    with tab_tiers:
        col_pie, col_info = st.columns([1, 1])
        with col_pie:
            fig_tiers = plot_trade_tiers(df_tv)
            st.plotly_chart(fig_tiers, use_container_width=True, key="tv_tiers_pie")
        with col_info:
            st.markdown("#### Легенда тиров")
            tiers_info = [
                ("💎 Франчайз", "≥ 82", "#1d428a",
                 "Лицо лиги. Максимальный контракт. Непродаваем."),
                ("⭐ Топ-15",   "68–81", "#c8102e",
                 "Звезда All-Star уровня. Очень высокая ценность."),
                ("🔵 Стартер",  "54–67", "#2980b9",
                 "Надёжный стартер. Хорошая трейд-фишка."),
                ("🟡 Ролевой",  "38–53", "#f39c12",
                 "Ценный резервист. Играет определённую роль."),
                ("⚪ Резерв",   "< 38",  "#95a5a6",
                 "Контракт по минимуму. Заменяемый."),
            ]
            for tier, rng, color, desc in tiers_info:
                st.html(
                    f"<div style='border-left:4px solid {color};padding:8px 12px;margin:6px 0;'>"
                    f"<b>{tier}</b> <span style='color:#888;font-size:12px'>(TV {rng})</span><br>"
                    f"<span style='font-size:13px;color:#8b9ab5'>{desc}</span></div>"
                )

        # Топ по каждому тиру
        st.divider()
        section_title("Лучшие в каждом тире", "🏅")
        tier_order = ["💎 Франчайз","⭐ Топ-15","🔵 Стартер","🟡 Ролевой","⚪ Резерв"]
        t_cols = st.columns(len(tier_order))
        for tc, tier in zip(t_cols, tier_order):
            sub = df_tv[df_tv["tv_tier"] == tier]
            if len(sub) == 0:
                continue
            top3 = sub.nlargest(3, "trade_value")
            tc.markdown(f"**{tier}**")
            for _, tr in top3.iterrows():
                tc.markdown(f"`{tr['trade_value']:.0f}` {tr[name_col]}")

    with tab_all:
        section_title("Полная таблица Trade Value", "📋")
        show_cols_tv = [c for c in [
            name_col, "team", "cluster_name", "trade_value", "tv_tier",
            "PTS", "AST", "REB", "tv_scoring", "tv_defense", "tv_efficiency",
            "tv_age_prime"
        ] if c in df_tv.columns]

        tbl_tv = df_tv[show_cols_tv].sort_values("trade_value", ascending=False).reset_index(drop=True)
        tbl_tv.index += 1

        fmt_dict = {c: "{:.1f}" for c in ["trade_value","PTS","AST","REB",
                                            "tv_scoring","tv_defense",
                                            "tv_efficiency","tv_age_prime"]
                    if c in tbl_tv.columns}

        st.dataframe(
            tbl_tv.style
                  .background_gradient(subset=["trade_value"] if "trade_value" in tbl_tv.columns else [],
                                       cmap="YlOrRd")
                  .format(fmt_dict),
            use_container_width=True,
        )

        # Поиск
        st.divider()
        search_player = st.selectbox(
            "🔍 Найти игрока",
            ["—"] + sorted(df_tv[name_col].dropna().tolist()),
            key="tv_search_player"
        )
        if search_player != "—":
            from src.trade_value import plot_trade_breakdown
            sr = df_tv[df_tv[name_col] == search_player].iloc[0]
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("💰 Trade Value", f"{sr['trade_value']:.1f}")
            sc2.metric("Тир", sr["tv_tier"])
            rank_s = int((df_tv["trade_value"] > sr["trade_value"]).sum()) + 1
            sc3.metric("Ранг", f"#{rank_s}")
            fig_bd = plot_trade_breakdown(sr)
            st.plotly_chart(fig_bd, use_container_width=True, key=f"tv_bd_{search_player}")


# ════════════════════════════════════════════════════════════════════════════════
# КРИВАЯ РАЗВИТИЯ
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📈 Развитие":
    page_header(
        "📈", "КРИВАЯ РАЗВИТИЯ",
        "Полиномиальная регрессия 3-й степени · Фазы карьеры · Прогноз пика",
        badge="Карьера", badge_color="#2ECC71",
        accent="#2ECC71",
    )
    page_hint([
        "**Что это?** Кривая развития карьеры: исторический тренд и прогноз пика для выбранного игрока.",
        "**Как работает?** Полиномиальная регрессия 3-й степени по возрасту → определяет фазу карьеры (рост / пик / спад) и предсказывает возраст максимальной результативности.",
        "**Зачем?** Решить, стоит ли подписывать долгосрочный контракт: игрок идёт в гору или уже прошёл пик? Особенно важно для Max-контрактов (4–5 лет).",
        "**Ключевые термины:** Фаза «Рост» — <25 лет, улучшение год к году · «Пик» — 26–30, стабильность · «Спад» — >31, снижение эффективности.",
    ])

    from src.development_curve import (
        get_development_summary, plot_development_curve, plot_multi_stat_curves
    )

    dev_player = st.selectbox("Выбери игрока", sorted(df[name_col].dropna().unique()),
                               key="dev_player_sel")

    if dev_player:
        summary = get_development_summary(df, dev_player)

        # ── Шапка с фазой карьеры ─────────────────────────────────────────────
        if summary:
            phase_col = "#27ae60" if summary["phase"] in ("Прорыв","Рост") else \
                        "#8e44ad" if summary["phase"] == "Пик" else \
                        "#e67e22" if summary["phase"] == "Стабильность" else "#e74c3c"

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🎂 Возраст", f"{summary['age']:.0f} лет" if summary.get("age") else "—")
            c2.html(
                f"<div style='padding:6px'><div style='font-size:12px;color:#666'>Фаза карьеры</div>"
                f"<div style='background:{phase_col};color:white;border-radius:20px;"
                f"padding:4px 14px;display:inline-block;font-weight:700;font-size:15px'>"
                f"{summary['phase_emoji']} {summary['phase']}</div></div>"
            )
            c3.metric("⭐ Пик (avg)", f"{summary['peak_age']} лет" if summary.get("peak_age") else "—")
            c4.metric("📅 До пика",
                      f"{summary['years_to_peak']:+.1f} лет" if summary.get("years_to_peak") is not None else "—")

            st.caption(f"_{summary.get('phase_desc','')}_")

            if summary.get("stat_peaks"):
                st.markdown("**Пиковый возраст по показателям:**")
                sp_cols = st.columns(len(summary["stat_peaks"]))
                stat_labels = {"PTS":"Очки","AST":"Передачи","REB":"Подборы","efficiency":"Эффективность"}
                for ci, (stat, peak_a) in zip(sp_cols, summary["stat_peaks"].items()):
                    ci.metric(stat_labels.get(stat, stat), f"{peak_a} лет")

        st.divider()

        # ── Графики ───────────────────────────────────────────────────────────
        tab_single, tab_multi = st.tabs(["📊 По показателю", "📊 Все показатели"])

        with tab_single:
            stat_choice = st.selectbox(
                "Показатель",
                [s for s in ["PTS","AST","REB","efficiency"] if s in df.columns],
                key="dev_stat_sel"
            )
            fig_dc = plot_development_curve(df, dev_player, stat=stat_choice)
            st.plotly_chart(fig_dc, use_container_width=True, key=f"dev_curve_{dev_player}_{stat_choice}")
            st.caption(
                "Серая полоса — среднее ±σ по лиге для данного возраста. "
                "Красная звезда — текущее положение игрока. "
                "Пунктир — прогнозируемый пик."
            )

        with tab_multi:
            fig_multi = plot_multi_stat_curves(df, dev_player)
            st.plotly_chart(fig_multi, use_container_width=True, key=f"dev_multi_{dev_player}")

        # ── Лига: топ игроков у пика ──────────────────────────────────────────
        st.divider()
        section_title("Игроки на пике (25–29 лет)", "🏆", "#FFC72C")
        age_col_d = next((c for c in ["PLAYER_AGE","AGE","age"] if c in df.columns), None)
        if age_col_d:
            peak_players = df[(df[age_col_d] >= 25) & (df[age_col_d] <= 29)].copy()
            show_stat = "PTS" if "PTS" in peak_players.columns else peak_players.columns[0]
            cols_pk = [c for c in [name_col, age_col_d, "PTS", "AST", "REB", "efficiency"] if c in peak_players.columns]
            st.dataframe(
                peak_players[cols_pk].sort_values("PTS" if "PTS" in peak_players.columns else cols_pk[0],
                                                   ascending=False).head(15).reset_index(drop=True),
                use_container_width=True
            )


# ════════════════════════════════════════════════════════════════════════════════
# РИСК ТРАВМЫ
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🏥 Риск травмы":
    page_header(
        "🏥", "РИСК ТРАВМЫ",
        "Составной индекс (0-100) · Возраст 30% · Нагрузка 25% · Эффективность 20% · Стресс 15%",
        badge="Здоровье", badge_color="#E74C3C",
        accent="#E74C3C",
    )
    page_hint([
        "**Что это?** Индекс риска травмы (0–100) для каждого игрока на основе пяти факторов риска.",
        "**Как работает?** InjuryRisk = 0.25×возраст + 0.30×нагрузка + 0.20×потери + 0.15×надёжность_GP + 0.10×эффективность. Каждый фактор масштабирован к [0,1].",
        "**Зачем?** Долгосрочный контракт с игроком risk > 70% — финансовая бомба. Используйте этот индекс вместе с TV для оценки соотношения ценность/риск.",
        "**Ключевые термины:** Risk < 30 — низкий риск 🟢 · 30–60 — средний 🟡 · > 70 — высокий 🔴 · GP < 50 — красный флаг (история пропусков).",
    ])

    from src.injury_risk import (
        compute_injury_risk, plot_risk_breakdown,
        plot_risk_leaderboard, plot_risk_vs_value, plot_risk_distribution
    )

    df_ir = get_injury_risks(_data_hash)

    # ── Метрики ───────────────────────────────────────────────────────────────
    r1c, r2c, r3c, r4c, r5c = st.columns(5)
    r1c.metric("🔴 Высокий",    int((df_ir["risk_tier"] == "🔴 Высокий").sum()))
    r2c.metric("🟠 Повышенный", int((df_ir["risk_tier"] == "🟠 Повышенный").sum()))
    r3c.metric("🟡 Умеренный",  int((df_ir["risk_tier"] == "🟡 Умеренный").sum()))
    r4c.metric("🟢 Низкий",     int((df_ir["risk_tier"] == "🟢 Низкий").sum()))
    r5c.metric("⚪ Минимальный",int((df_ir["risk_tier"] == "⚪ Минимальный").sum()))

    st.divider()

    tab_lead, tab_dist, tab_scatter, tab_player = st.tabs(
        ["🏆 Лидерборд", "📊 Распределение", "🎯 Риск vs Ценность", "👤 Игрок"]
    )

    with tab_lead:
        top_n_ir = st.slider("Топ-N игроков", 10, 40, 20, key="ir_top_n")
        df_ir_f = df_ir.copy()
        if sel_cluster != "Все" and "cluster_name" in df_ir_f.columns:
            df_ir_f = df_ir_f[df_ir_f["cluster_name"] == sel_cluster]
        if "games_played" in df_ir_f.columns:
            df_ir_f = df_ir_f[df_ir_f["games_played"] >= min_games]
        fig_irl = plot_risk_leaderboard(df_ir_f, n=top_n_ir)
        st.plotly_chart(fig_irl, use_container_width=True, key="ir_leaderboard")

    with tab_dist:
        fig_ird = plot_risk_distribution(df_ir)
        st.plotly_chart(fig_ird, use_container_width=True, key="ir_distribution")
        st.caption(
            "Распределение индекса риска по всей лиге. "
            "Зоны: зелёная — безопасно, красная — высокий риск."
        )

    with tab_scatter:
        fig_irv = plot_risk_vs_value(df_ir)
        st.plotly_chart(fig_irv, use_container_width=True, key="ir_vs_value")
        st.caption(
            "Игроки в правом нижнем углу — высокая ценность при низком риске (идеальные). "
            "Левый верхний угол — высокий риск при низкой ценности (проблемные)."
        )

    with tab_player:
        ir_player = st.selectbox("Выбери игрока", sorted(df[name_col].dropna().unique()),
                                  key="ir_player_sel")
        if ir_player:
            rows_ir = df_ir[df_ir[name_col] == ir_player]
            if not rows_ir.empty:
                row_ir = rows_ir.iloc[0]
                risk_val = float(row_ir["injury_risk"])
                risk_tier = str(row_ir["risk_tier"])
                risk_color = str(row_ir["risk_color"])

                # Шапка
                h1, h2, h3 = st.columns(3)
                h1.metric("⚠️ Индекс риска", f"{risk_val:.0f} / 100")
                h2.html(
                    f"<div style='padding:8px'><div style='font-size:12px;color:#666'>Уровень</div>"
                    f"<div style='background:{risk_color};color:white;border-radius:20px;"
                    f"padding:4px 14px;display:inline-block;font-weight:700;font-size:15px'>"
                    f"{risk_tier}</div></div>"
                )
                rank_ir = int((df_ir["injury_risk"] > risk_val).sum()) + 1
                h3.metric("📊 Ранг риска", f"#{rank_ir} из {len(df_ir)}")

                # Breakdown
                fig_irb = plot_risk_breakdown(row_ir)
                st.plotly_chart(fig_irb, use_container_width=True, key=f"ir_breakdown_{ir_player}")

                # Рекомендации
                section_title("Рекомендации", "💡", "#FFC72C")
                recs = []
                if float(row_ir.get("risk_age", 0)) > 70:
                    recs.append("🔴 **Возраст**: Игроку требуется повышенный мониторинг нагрузки.")
                if float(row_ir.get("risk_workload", 0)) > 70:
                    recs.append("🟠 **Нагрузка**: Рассмотреть ротацию и снижение игровых минут.")
                if float(row_ir.get("risk_efficiency", 0)) > 70:
                    recs.append("🟡 **Эффективность**: Падение эффективности может указывать на усталость.")
                if float(row_ir.get("risk_body", 0)) > 70:
                    recs.append("🟠 **Стресс**: Высокий балл очков при малом количестве игр — перегруз.")
                if not recs:
                    recs.append("✅ Уровень риска в норме. Продолжать мониторинг.")
                for rec in recs:
                    st.markdown(rec)


# ════════════════════════════════════════════════════════════════════════════════
# ОПТИМАЛЬНЫЙ СОСТАВ
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🏗️ Состав":
    page_header(
        "🏗️", "ОПТИМАЛЬНЫЙ СОСТАВ",
        "ML-оптимизация · 5 позиций · 4 стратегии: Сбалансированный / Атака / Защита / Молодёжь",
        badge="Тактика", badge_color="#27AE60",
        accent="#27AE60",
    )
    page_hint([
        "**Что это?** Подбор оптимального состава из доступного пула игроков по выбранной стратегии.",
        "**Как работает?** Жадный алгоритм: ранжирует игроков по целевой функции (TV/риск для «Баланс», PTS для «Атака», STL+BLK для «Защита», возраст для «Молодёжь») → заполняет 5 позиций не нарушая бюджета риска.",
        "**Зачем?** Быстро проверить гипотезу: «если мы обменяем X на Y, как изменится оптимальный состав?» Меняйте доступный пул и сравнивайте результаты.",
        "**Ключевые термины:** Бюджет риска — максимальный средний InjuryRisk состава · TV/Risk ratio — аналог коэффициента Шарпа для игроков.",
    ])

    from src.team_builder import build_optimal_lineup, plot_lineup_court, plot_lineup_bars, plot_lineup_radar

    # ── Настройки ─────────────────────────────────────────────────────────────
    tb_col1, tb_col2 = st.columns([2, 1])
    with tb_col1:
        strategy = st.radio(
            "🎯 Стратегия",
            ["balanced", "offense", "defense", "young"],
            format_func=lambda x: {
                "balanced": "⚖️ Сбалансированный",
                "offense":  "🏀 Атака",
                "defense":  "🛡 Защита",
                "young":    "🌱 Молодёжь (≤26 лет)",
            }[x],
            horizontal=True,
            key="tb_strategy"
        )
    with tb_col2:
        df_tv_tb = get_trade_values(_data_hash)
        locked_players = st.multiselect(
            "🔒 Принудительно включить",
            sorted(df[name_col].dropna().unique()),
            max_selections=3,
            key="tb_locked"
        )

    # Строим состав
    try:
        lineup_result = build_optimal_lineup(df_tv_tb, strategy=strategy, locked=locked_players)
        lineup_list = lineup_result["lineup"]
        team_stats  = lineup_result["team_stats"]

        # ── Карточки игроков ──────────────────────────────────────────────────
        section_title("Стартовая пятёрка", "🏀")
        pos_labels = {"PG":"🔵","SG":"🟣","SF":"🟢","PF":"🟠","C":"🔴"}
        p_cols = st.columns(5)
        for i, player in enumerate(lineup_list):
            with p_cols[i]:
                color = player["color"]
                pos   = player["pos"]
                emoji = pos_labels.get(pos, "⚪")
                pid_tb = df[df[name_col] == player["name"]]["PLAYER_ID"].values
                if len(pid_tb) > 0 and not pd.isna(pid_tb[0]):
                    player_img(pid_tb[0], width=100, radius=8)
                st.html(
                    f"<div style='background:{color};color:white;border-radius:10px;"
                    f"padding:10px;text-align:center;margin-top:4px'>"
                    f"<div style='font-size:11px;opacity:0.85'>{emoji} {player['pos_ru']}</div>"
                    f"<div style='font-weight:700;font-size:14px'>{player['name'].split()[-1]}</div>"
                    f"<div style='font-size:12px;margin-top:4px'>"
                    f"Очки: {player['pts']} | Пас: {player['ast']}<br>"
                    f"Подб: {player['reb']} | Рейтинг: {player['score']}</div>"
                    f"</div>"
                )

        st.divider()

        # ── Командная статистика ──────────────────────────────────────────────
        section_title("Командная статистика (сумма)", "📊", "#1D428A")
        tm1, tm2, tm3, tm4, tm5 = st.columns(5)
        tm1.metric("🏀 Очки",       f"{team_stats.get('pts', 0):.1f}")
        tm2.metric("🎯 Передачи",   f"{team_stats.get('ast', 0):.1f}")
        tm3.metric("💪 Подборы",    f"{team_stats.get('reb', 0):.1f}")
        tm4.metric("⚡ Эффективность", f"{team_stats.get('efficiency', 0):.1f}")
        tm5.metric("🎂 Средний возраст",
                   f"{team_stats['avg_age']:.1f}" if team_stats.get("avg_age") else "—")

        # ── Графики ───────────────────────────────────────────────────────────
        tab_court, tab_bars, tab_radar = st.tabs(
            ["🏟 Схема площадки", "📊 Статистика", "🎯 Radar"]
        )

        with tab_court:
            fig_court = plot_lineup_court(lineup_result)
            st.plotly_chart(fig_court, use_container_width=True, key="tb_court")

        with tab_bars:
            fig_bars = plot_lineup_bars(lineup_result)
            st.plotly_chart(fig_bars, use_container_width=True, key="tb_bars")

        with tab_radar:
            fig_rad = plot_lineup_radar(lineup_result)
            st.plotly_chart(fig_rad, use_container_width=True, key="tb_radar")

    except Exception as e:
        st.error(f"Ошибка построения состава: {e}")
        import traceback
        st.code(traceback.format_exc())


# ════════════════════════════════════════════════════════════════════════════════
# CONTRACT VALUE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "💵 Контракт":
    page_header(
        "💵", "КОНТРАКТНАЯ СТОИМОСТЬ",
        "Синтетический рыночный анализ · Trade Value → Зарплата · Недооценённые vs Переоценённые",
        badge="Рынок", badge_color="#8E44AD",
        accent="#8E44AD",
    )
    page_hint([
        "**Что это?** Оценка справедливой рыночной стоимости контракта каждого игрока и выявление переплат / скрытых выгод.",
        "**Как работает?** TV → логарифмическая шкала → калибровка к диапазону реальных зарплат NBA ($1M–$50M+). Игроки выше справедливой линии — переоценены; ниже — недооценены.",
        "**Зачем?** Найти «недооценённых» — игроков, которых ещё не заметил рынок. Это основа стратегии Moneyball: покупать ценность дёшево.",
        "**Ключевые термины:** Fair Value — справедливая зарплата по модели · Surplus Value — разница (справедливая − реальная): >0 = выгодная сделка · Max Contract — порог ~$30M+.",
    ])

    from src.contract_value import (
        plot_salary_leaderboard, plot_value_vs_salary,
        plot_value_surplus, plot_contract_tiers_breakdown,
        get_contract_summary
    )

    df_cv = get_contract_values(_data_hash)

    # ── Метрики ───────────────────────────────────────────────────────────────
    cv1, cv2, cv3, cv4, cv5 = st.columns(5)
    cv1.metric("💰 Макс-контракт", int((df_cv["contract_tier"] == "💰 Макс-контракт").sum()))
    cv2.metric("⭐ Всезвёздный",   int((df_cv["contract_tier"] == "⭐ Всезвёздный").sum()))
    cv3.metric("🔵 Стартер",       int((df_cv["contract_tier"] == "🔵 Стартер").sum()))
    cv4.metric("🟢 Недооценён",    int((df_cv["value_label"]   == "Недооценён").sum()))
    cv5.metric("🔴 Переоценён",    int((df_cv["value_label"]   == "Переоценён").sum()))

    st.divider()

    tab_lead_c, tab_surplus, tab_scatter_c, tab_tiers_c, tab_player_c = st.tabs([
        "💰 Зарплатный рейтинг", "📊 Профицит", "🎯 Value vs Зарплата",
        "🥧 Тиры", "👤 Игрок"
    ])

    with tab_lead_c:
        top_n_cv = st.slider("Топ-N", 10, 40, 20, key="cv_top_n")
        df_cv_f = df_cv.copy()
        if sel_cluster != "Все" and "cluster_name" in df_cv_f.columns:
            df_cv_f = df_cv_f[df_cv_f["cluster_name"] == sel_cluster]
        if "games_played" in df_cv_f.columns:
            df_cv_f = df_cv_f[df_cv_f["games_played"] >= min_games]
        fig_csl = plot_salary_leaderboard(df_cv_f, n=top_n_cv)
        st.plotly_chart(fig_csl, use_container_width=True, key="cv_salary_lead")

    with tab_surplus:
        fig_surp = plot_value_surplus(df_cv)
        st.plotly_chart(fig_surp, use_container_width=True, key="cv_surplus")
        st.caption(
            "**Зелёный** — игрок недооценён (Trade Value выше рыночной зарплаты). "
            "**Красный** — переоценён (зарплата выше реальной ценности)."
        )

    with tab_scatter_c:
        fig_cvs = plot_value_vs_salary(df_cv)
        st.plotly_chart(fig_cvs, use_container_width=True, key="cv_scatter")
        st.caption(
            "Пунктирная линия — «справедливая цена». "
            "Выше линии — ценность выше зарплаты (недооценён). "
            "Ниже — переоценён."
        )

    with tab_tiers_c:
        fig_ct = plot_contract_tiers_breakdown(df_cv)
        st.plotly_chart(fig_ct, use_container_width=True, key="cv_tiers")

    with tab_player_c:
        cv_player = st.selectbox("Выбери игрока", sorted(df[name_col].dropna().unique()),
                                  key="cv_player_sel")
        if cv_player:
            summary_cv = get_contract_summary(df_cv, cv_player)
            if summary_cv:
                p1c, p2c, p3c, p4c = st.columns(4)
                p1c.metric("💵 Рыночная зарплата",
                           f"${summary_cv['market_salary_m']:.1f}M")
                p2c.html(
                    f"<div style='padding:8px'><div style='font-size:12px;color:#666'>Тир</div>"
                    f"<div style='background:{summary_cv['contract_color']};color:white;"
                    f"border-radius:20px;padding:4px 14px;display:inline-block;"
                    f"font-weight:700;font-size:14px'>{summary_cv['contract_tier']}</div></div>"
                )
                p3c.metric("💰 Trade Value", f"{summary_cv['trade_value']:.1f}")

                surplus = summary_cv["value_surplus"]
                surplus_color = "#27ae60" if surplus > 8 else "#e74c3c" if surplus < -8 else "#3498db"
                p4c.html(
                    f"<div style='padding:8px'><div style='font-size:12px;color:#666'>Оценка</div>"
                    f"<div style='background:{surplus_color};color:white;border-radius:20px;"
                    f"padding:4px 14px;display:inline-block;font-weight:700;font-size:14px'>"
                    f"{summary_cv['value_label']} ({surplus:+.1f})</div></div>"
                )

                # Визуальная шкала
                st.divider()
                st.markdown("#### 📊 Зарплатная шкала лиги")
                salary_val = summary_cv["market_salary_m"]
                all_sal = df_cv["market_salary_m"].dropna()
                pct_sal = float((all_sal < salary_val).mean() * 100)
                col_bar_w = int(pct_sal * 3.5)
                sal_color = "#27ae60" if pct_sal < 50 else "#f39c12" if pct_sal < 80 else "#e74c3c"
                st.html(
                    f"<div style='margin:10px 0'>"
                    f"<span style='font-size:13px;color:#8b9ab5'>${salary_val:.1f}M</span> — "
                    f"выше, чем у <b style='color:#e8eaf6'>{pct_sal:.0f}%</b> игроков лиги<br>"
                    f"<div style='background:rgba(255,255,255,0.07);border-radius:8px;height:10px;margin-top:8px'>"
                    f"<div style='background:{sal_color};width:{min(col_bar_w, 350)}px;"
                    f"height:10px;border-radius:8px'></div></div></div>"
                )
            else:
                st.warning(f"Данные для {cv_player} не найдены.")


# ════════════════════════════════════════════════════════════════════════════════
# ПРОДВИНУТАЯ АНАЛИТИКА — Win Shares / VORP / On-Off / Momentum / Similarity
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Продвинутая аналитика":
    page_header(
        "🔬", "ПРОДВИНУТАЯ АНАЛИТИКА",
        "Win Shares · VORP · BPM · On/Off Split · Momentum Score · Player Similarity",
        badge="Advanced Stats", badge_color="#3498DB",
        accent="#3498DB",
    )
    page_hint([
        "**Что это?** Продвинутые метрики, которые не видны в box-score: Win Shares, BPM, VORP, On/Off, Momentum, граф схожих игроков.",
        "**Как работает?** Win Shares ≈ вклад в победы команды · BPM — Box Plus/Minus (эффективность относительно среднего игрока лиги) · VORP — Value Over Replacement Player · On/Off — разница ±, когда игрок на/вне площадки · Momentum — тренд последних 10 игр.",
        "**Зачем?** Стандартная статистика (PTS/AST/REB) не показывает *реальный вклад* игрока. Эти метрики используют все топ-клубы NBA для оценки «невидимого» вклада.",
        "**Ключевые термины:** BPM > +3 — All-Star уровень · VORP > 3.0 — претендент на MVP · On/Off > +5 — ключевой игрок · Momentum > 0 — игрок в форме.",
    ])

    from src.advanced_stats import (
        plot_ws_leaderboard, plot_vorp_chart,
        plot_on_off_scatter, plot_on_off_player,
        plot_momentum_leaderboard, plot_momentum_sparklines,
        get_similar_players, plot_similarity_radar, plot_similarity_heatmap,
    )

    df_adv = get_advanced_stats(_data_hash)

    tab_ws, tab_onoff, tab_momentum, tab_sim = st.tabs([
        "🏆 Win Shares / VORP",
        "📶 On/Off Split",
        "🔥 Momentum",
        "🔍 Player Similarity",
    ])

    # ── Win Shares / VORP ────────────────────────────────────────────────────
    with tab_ws:
        section_title("Win Shares & VORP", "🏆")

        wa1, wa2, wa3, wa4 = st.columns(4)
        wa1.metric("Лидер WS",
                   df_adv.loc[df_adv["win_shares"].idxmax(), name_col]
                   if "win_shares" in df_adv.columns else "—",
                   f"{df_adv['win_shares'].max():.1f} WS" if "win_shares" in df_adv.columns else "")
        wa2.metric("Лидер VORP",
                   df_adv.loc[df_adv["vorp"].idxmax(), name_col]
                   if "vorp" in df_adv.columns else "—",
                   f"{df_adv['vorp'].max():.1f}" if "vorp" in df_adv.columns else "")
        wa3.metric("Лидер BPM",
                   df_adv.loc[df_adv["bpm"].idxmax(), name_col]
                   if "bpm" in df_adv.columns else "—",
                   f"{df_adv['bpm'].max():.2f}" if "bpm" in df_adv.columns else "")
        wa4.metric("Лидер WS/48",
                   df_adv.loc[df_adv["ws_per_48"].idxmax(), name_col]
                   if "ws_per_48" in df_adv.columns else "—",
                   f"{df_adv['ws_per_48'].max():.3f}" if "ws_per_48" in df_adv.columns else "")

        ws_n = st.slider("Топ-N игроков", 10, 40, 20, key="ws_top_n")
        df_adv_f = df_adv.copy()
        if sel_cluster != "Все" and "cluster_name" in df_adv_f.columns:
            df_adv_f = df_adv_f[df_adv_f["cluster_name"] == sel_cluster]
        if "games_played" in df_adv_f.columns:
            df_adv_f = df_adv_f[df_adv_f["games_played"] >= min_games]

        col_ws1, col_ws2 = st.columns(2)
        with col_ws1:
            fig_ws = plot_ws_leaderboard(df_adv_f, n=ws_n)
            st.plotly_chart(fig_ws, use_container_width=True, key="ws_leaderboard")
        with col_ws2:
            fig_vorp = plot_vorp_chart(df_adv_f, n=ws_n)
            st.plotly_chart(fig_vorp, use_container_width=True, key="vorp_scatter")

        st.caption(
            "**BPM** (Box Plus/Minus) — на сколько пунктов лучше команда при игроке на площадке. "
            "**VORP** — дополнительные победы над игроком уровня замены. "
            "**WS** (Win Shares) — доля командных побед, приписываемая игроку."
        )

        # Поиск игрока
        st.divider()
        ws_player = st.selectbox("🔍 Показатели игрока",
                                  ["—"] + sorted(df_adv[name_col].dropna().tolist()),
                                  key="ws_player_search")
        if ws_player != "—":
            wr = df_adv[df_adv[name_col] == ws_player].iloc[0]
            wc1, wc2, wc3, wc4, wc5 = st.columns(5)
            wc1.metric("BPM",      f"{wr.get('bpm', 0):.2f}")
            wc2.metric("VORP",     f"{wr.get('vorp', 0):.2f}")
            wc3.metric("WS",       f"{wr.get('win_shares', 0):.2f}")
            wc4.metric("WS/48",    f"{wr.get('ws_per_48', 0):.3f}")
            wc5.metric("WS Off",   f"{wr.get('ws_off', 0):.2f} / Def {wr.get('ws_def', 0):.2f}")
            rank_ws = int((df_adv["win_shares"] > float(wr.get("win_shares", 0))).sum()) + 1
            st.caption(f"Ранг по Win Shares: **#{rank_ws}** из {len(df_adv)}")

    # ── On/Off Split ─────────────────────────────────────────────────────────
    with tab_onoff:
        section_title("On / Off Split", "📶")

        fig_onoff = plot_on_off_scatter(df_adv)
        st.plotly_chart(fig_onoff, use_container_width=True, key="onoff_scatter")
        st.caption(
            "**On NET** — рейтинг команды пока игрок на площадке. "
            "**On-Off diff** — насколько команда лучше/хуже с игроком (+= незаменим). "
            "Правый верхний угол — двусторонний импакт."
        )

        st.divider()
        oo_player = st.selectbox("👤 On/Off для игрока",
                                  sorted(df_adv[name_col].dropna().unique()),
                                  key="oo_player_sel")
        if oo_player:
            oo_row = df_adv[df_adv[name_col] == oo_player].iloc[0]
            fig_oop = plot_on_off_player(oo_row, oo_player)
            st.plotly_chart(fig_oop, use_container_width=True, key=f"oo_player_{oo_player}")

            oo1, oo2, oo3 = st.columns(3)
            oo1.metric("На площадке", f"{float(oo_row.get('on_net', 0)):+.1f}")
            oo2.metric("Вне площадки", f"{float(oo_row.get('off_net', 0)):+.1f}")
            diff_oo = float(oo_row.get("on_off_diff", 0))
            diff_col = "#27ae60" if diff_oo > 0 else "#e74c3c"
            oo3.metric("Дифференциал", f"{diff_oo:+.1f}")

    # ── Momentum ─────────────────────────────────────────────────────────────
    with tab_momentum:
        section_title("Momentum Score", "🔥", "#E67E22")

        mom_n = st.slider("Топ-N", 10, 40, 20, key="mom_top_n")
        df_adv_fm = df_adv.copy()
        if sel_cluster != "Все" and "cluster_name" in df_adv_fm.columns:
            df_adv_fm = df_adv_fm[df_adv_fm["cluster_name"] == sel_cluster]
        if "games_played" in df_adv_fm.columns:
            df_adv_fm = df_adv_fm[df_adv_fm["games_played"] >= min_games]

        fig_mom = plot_momentum_leaderboard(df_adv_fm, n=mom_n)
        st.plotly_chart(fig_mom, use_container_width=True, key="mom_leaderboard")
        st.caption(
            "Momentum учитывает: PIE (30%), Net Rating (25%), Эффективность (20%), "
            "Возраст/пик (15%), USG × Efficiency (10%)."
        )

        st.divider()
        mom_player = st.selectbox("📈 Тренд игрока",
                                   sorted(df_adv[name_col].dropna().unique()),
                                   key="mom_player_sel")
        if mom_player:
            mom_row = df_adv[df_adv[name_col] == mom_player].iloc[0]
            mc1, mc2 = st.columns([1, 2])
            with mc1:
                m_val   = float(mom_row.get("momentum", 50))
                m_label = str(mom_row.get("momentum_label", "—"))
                m_color = str(mom_row.get("momentum_color", "#3498db"))
                st.html(
                    f"<div style='background:{m_color};color:white;border-radius:14px;"
                    f"padding:20px;text-align:center;margin-top:10px'>"
                    f"<div style='font-size:13px;opacity:0.85'>Momentum Score</div>"
                    f"<div style='font-size:42px;font-weight:900'>{m_val:.0f}</div>"
                    f"<div style='font-size:14px;margin-top:4px'>{m_label}</div>"
                    f"</div>"
                )
            with mc2:
                fig_spark = plot_momentum_sparklines(df_adv, mom_player)
                st.plotly_chart(fig_spark, use_container_width=True,
                                key=f"mom_spark_{mom_player}")

    # ── Player Similarity ────────────────────────────────────────────────────
    with tab_sim:
        section_title("Player Similarity", "🔍", "#3498DB")
        st.caption("Косинусное сходство по 11 статам: очки, передачи, подборы, "
                   "перехваты, блоки, FG%, 3P%, USG%, TS%, +/−, эффективность")

        sim_player = st.selectbox("Выбери игрока",
                                   sorted(df_adv[name_col].dropna().unique()),
                                   key="sim_player_sel")
        n_sim = st.slider("Количество похожих", 3, 10, 5, key="sim_n")

        if sim_player:
            similar = get_similar_players(df_adv, sim_player, n=n_sim)

            if not similar.empty:
                # Карточки похожих
                st.markdown("#### Наиболее похожие игроки")
                sim_cols = st.columns(min(n_sim, 5))
                for i, (_, sr) in enumerate(similar.head(5).iterrows()):
                    with sim_cols[i % 5]:
                        pid_s = df[df[name_col] == sr[name_col]]["PLAYER_ID"].values
                        if len(pid_s) > 0 and not pd.isna(pid_s[0]):
                            player_img(pid_s[0], width=90, radius=8)
                        sim_pct = float(sr.get("similarity", 0))
                        bar_w   = int(sim_pct * 0.9)
                        bar_col = "#27ae60" if sim_pct > 85 else "#3498db" if sim_pct > 70 else "#e67e22"
                        st.html(
                            f"<div style='background:#f8f9fa;border-radius:10px;padding:8px;"
                            f"border-left:4px solid {bar_col};margin-top:4px'>"
                            f"<b style='font-size:12px'>{sr[name_col]}</b><br>"
                            f"<span style='font-size:11px;color:#888'>"
                            f"Сходство: {sim_pct:.0f}%</span><br>"
                            f"<div style='background:#e0e0e0;border-radius:4px;height:6px;margin-top:4px'>"
                            f"<div style='background:{bar_col};width:{bar_w}%;height:6px;"
                            f"border-radius:4px'></div></div>"
                            f"</div>"
                        )

                st.divider()

                tab_radar_sim, tab_heat_sim = st.tabs(["🎯 Radar", "🌡 Heatmap"])

                with tab_radar_sim:
                    n_rad = st.slider("Игроков на радаре", 2, 4, 3, key="sim_rad_n")
                    fig_rad_s = plot_similarity_radar(df_adv, sim_player, n_similar=n_rad)
                    st.plotly_chart(fig_rad_s, use_container_width=True,
                                    key=f"sim_radar_{sim_player}")

                with tab_heat_sim:
                    fig_heat_s = plot_similarity_heatmap(df_adv, sim_player, n=min(n_sim, 10))
                    st.plotly_chart(fig_heat_s, use_container_width=True,
                                    key=f"sim_heat_{sim_player}")
            else:
                st.warning("Недостаточно данных для расчёта сходства.")


# ════════════════════════════════════════════════════════════════════════════════
# MONTE CARLO
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🎲 Monte Carlo":
    page_header(
        "🎲", "MONTE CARLO СИМУЛЯЦИЯ",
        "8 000 симуляций на игрока · Вероятностные прогнозы · Интервалы неопределённости",
        badge="Probabilistic ML", badge_color="#9B59B6", accent="#9B59B6",
    )
    page_hint([
        "**Что это?** Вероятностные прогнозы: не «игрок наберёт 18.5 очков», а «с вероятностью 62% он наберёт ≥20 очков».",
        "**Как работает?** 8 000 симуляций сезона из Normal(μ, σ): μ = исторический средний, σ = μ × CV. CV зависит от числа игр (GP < 30 → выше неопределённость) и возраста (<22 или >34 → нестабильнее).",
        "**Зачем?** GM принимает контрактные решения в условиях неопределённости. P(All-Star) > 50% — разумное основание для Max-контракта. P10–P90 показывает диапазон возможных исходов.",
        "**Ключевые термины:** P(All-Star) = P(PTS≥20 И AST≥5 И REB≥5) · P10/P90 — нижняя/верхняя граница 80% предсказательного интервала · CV — коэффициент вариации (нестабильность).",
    ])

    df_mc = get_monte_carlo(_data_hash)
    name_col_mc = next(
        (c for c in ("PLAYER_NAME", "player_name", "name", "Name", "name_col")
         if c in df_mc.columns),
        df_mc.columns[0],
    )

    kpi_row([
        {"icon": "👥", "label": "Игроков",              "value": str(len(df_mc))},
        {"icon": "🎯", "label": "All-Star шанс ≥50%",
         "value": str((df_mc["p_allstar"] >= 50).sum())},
        {"icon": "🔥", "label": "P(≥25 очков) > 30%",
         "value": str((df_mc["p_25plus"] >= 30).sum())},
        {"icon": "📊", "label": "Медиана E[Очки]",
         "value": f"{df_mc['PTS_mean'].median():.1f}"},
        {"icon": "🎲", "label": "Симуляций/игрок",      "value": "8 000"},
    ])

    tab_uncert, tab_prob, tab_table, tab_player_mc = st.tabs([
        "🗺 Карта неопределённости", "🏆 Рейтинг вероятностей",
        "📋 Таблица", "🔍 Анализ игрока",
    ])

    with tab_uncert:
        from src.monte_carlo import plot_uncertainty_scatter
        st.plotly_chart(plot_uncertainty_scatter(df_mc),
                        use_container_width=True, key="mc_uncert")
        st.caption(
            "Размер точки — P(All-Star). Ось Y — неопределённость (CV%). "
            "Идеал: правый нижний угол (высокий E[PTS], низкий CV)."
        )

    with tab_prob:
        from src.monte_carlo import plot_probability_leaderboard
        _mc_opts = {
            "p_allstar": "All-Star (20pts / 5ast / 5reb)",
            "p_25plus":  "≥ 25 очков за игру",
            "p_20plus":  "≥ 20 очков за игру",
            "p_15plus":  "≥ 15 очков за игру",
        }
        _mc_opts_avail = {k: v for k, v in _mc_opts.items() if k in df_mc.columns}
        mc_metric = st.selectbox(
            "Показатель", list(_mc_opts_avail.keys()),
            format_func=lambda x: _mc_opts_avail.get(x, x),
            key="mc_metric_sel",
        )
        top_n_mc = st.slider("Топ-N", 10, 50, 25, key="mc_topn")
        st.plotly_chart(
            plot_probability_leaderboard(df_mc, metric=mc_metric, n=top_n_mc),
            use_container_width=True, key="mc_prob_board",
        )

    with tab_table:
        section_title("Вероятности по всем игрокам", "📋")
        _show_mc = [c for c in [
            name_col_mc, "team", "cluster_name",
            "PTS_mean", "PTS_p10", "PTS_p90",
            "p_20plus", "p_25plus", "p_allstar", "sim_cv",
        ] if c in df_mc.columns]
        _tbl_mc = (df_mc[_show_mc]
                   .sort_values("p_allstar", ascending=False)
                   .reset_index(drop=True))
        _tbl_mc.index += 1
        _fmt_mc = {c: "{:.1f}" for c in _show_mc
                   if c not in [name_col_mc, "team", "cluster_name"]}
        st.dataframe(
            _tbl_mc.style
                .background_gradient(
                    subset=["p_allstar"] if "p_allstar" in _tbl_mc.columns else [],
                    cmap="RdYlGn", vmin=0, vmax=100)
                .format(_fmt_mc),
            use_container_width=True,
        )

    with tab_player_mc:
        from src.monte_carlo import plot_player_simulation
        mc_player = st.selectbox(
            "Игрок", sorted(df[name_col].dropna().unique()), key="mc_player_sel"
        )
        mc_stat = st.radio(
            "Статистика", ["PTS", "AST", "REB"],
            horizontal=True, key="mc_stat_sel",
        )
        if mc_player:
            row_mc = df[df[name_col] == mc_player].iloc[0]
            st.plotly_chart(
                plot_player_simulation(row_mc, stat=mc_stat),
                use_container_width=True, key=f"mc_sim_{mc_player}_{mc_stat}",
            )
            _rp = df_mc[df_mc[name_col_mc] == mc_player]
            if not _rp.empty:
                rp = _rp.iloc[0]
                pc1, pc2, pc3, pc4 = st.columns(4)
                pc1.metric("E[Очки]",       f"{rp.get('PTS_mean', 0):.1f}")
                pc2.metric("P(All-Star)",   f"{rp.get('p_allstar', 0):.0f}%")
                pc3.metric("P(≥20 очков)",  f"{rp.get('p_20plus', 0):.0f}%")
                pc4.metric("Неопределённость", f"{rp.get('sim_cv', 0):.1f}%")


# ════════════════════════════════════════════════════════════════════════════════
# MARKOWITZ PORTFOLIO
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📐 Марковиц":
    page_header(
        "📐", "ПОРТФЕЛЬНАЯ ОПТИМИЗАЦИЯ",
        "Марковиц Mean-Variance · Эффективная граница · Максимальный TV при заданном риске",
        badge="Portfolio Theory", badge_color="#E67E22", accent="#E67E22",
    )
    page_hint([
        "**Что это?** Оптимальный ростер как финансовый портфель: максимальный суммарный Trade Value при заданном уровне риска травм.",
        "**Как работает?** Mean-Variance Optimization Марковица (Нобелевская премия 1990): каждый игрок = «акция» с доходностью TV и риском InjuryRisk. Алгоритм строит эффективную границу Парето — 20 оптимальных составов при разных бюджетах риска.",
        "**Зачем?** Диверсификация работает и в NBA: состав только из «звёзд» = высокий риск травм. Правильный микс звёзд и надёжных ролевиков максимизирует TV при приемлемом риске.",
        "**Ключевые термины:** Граница Парето — множество оптимальных решений · Бюджет риска (%) — допустимый √mean(risk²) ростера · Sharpe Ratio аналог — TV / InjuryRisk.",
    ])

    df_pf = get_portfolio_data(_data_hash)
    from src.portfolio import (optimize_team, compute_efficient_frontier,
                                plot_efficient_frontier, plot_risk_return_scatter,
                                plot_team_composition)

    tab_frontier, tab_team, tab_scatter_pf = st.tabs([
        "📈 Эффективная граница", "🏆 Оптимальный состав", "💹 Risk-Return",
    ])

    with tab_frontier:
        section_title("Граница Парето оптимальных составов", "📈", color="#E67E22")
        st.markdown(
            "_Каждая точка красной линии — состав с максимальным суммарным Trade Value "
            "при данном уровне риска травм. Фон — случайные составы._"
        )
        col_pf1, col_pf2 = st.columns([3, 1])
        with col_pf2:
            risk_tol = st.slider("Допустимый риск", 10, 90, 50, 5, key="pf_risk_tol")
            n_pf     = st.slider("Игроков в составе", 3, 8, 5, key="pf_n")
        with col_pf1:
            with st.spinner("Вычисляем границу..."):
                _frontier = compute_efficient_frontier(df_pf, n_points=20, n_players=n_pf)
                _opt_team = optimize_team(df_pf, n_players=n_pf, risk_budget=float(risk_tol))
            st.plotly_chart(
                plot_efficient_frontier(_frontier, selected_team=_opt_team,
                                        all_players_df=df_pf),
                use_container_width=True, key="pf_frontier",
            )

    with tab_team:
        section_title("Оптимальный состав", "🏆", color="#E67E22")
        col_t1, col_t2 = st.columns([1, 2])
        with col_t1:
            risk_tol2 = st.slider("Риск-бюджет", 10, 90, 50, 5, key="pf_risk2")
            n_pf2     = st.slider("Размер состава", 3, 8, 5, key="pf_n2")
        _opt2 = optimize_team(df_pf, n_players=n_pf2, risk_budget=float(risk_tol2))
        if _opt2 is not None and len(_opt2) > 0:
            kpi_row([
                {"icon": "💰", "label": "Суммарный Trade Value",
                 "value": f"{_opt2['trade_value'].sum():.0f}"},
                {"icon": "⚠️", "label": "Риск портфеля",
                 "value": f"{float(np.sqrt((_opt2['injury_risk']**2).mean())):.1f}"},
                {"icon": "📊", "label": "Ср. Trade Value",
                 "value": f"{_opt2['trade_value'].mean():.1f}"},
                {"icon": "👥", "label": "Игроков", "value": str(len(_opt2))},
            ])
            st.plotly_chart(plot_team_composition(_opt2),
                            use_container_width=True, key="pf_team_comp")
            _nc_pf = "name" if "name" in _opt2.columns else "PLAYER_NAME"
            _show_pf = [c for c in [_nc_pf, "team", "cluster_name",
                                     "trade_value", "tv_tier", "injury_risk",
                                     "PTS", "AST", "REB"] if c in _opt2.columns]
            st.dataframe(_opt2[_show_pf].reset_index(drop=True),
                         use_container_width=True)

    with tab_scatter_pf:
        st.plotly_chart(plot_risk_return_scatter(df_pf),
                        use_container_width=True, key="pf_scatter")
        st.caption(
            "Размер точки — очки за игру. "
            "Ищи правый нижний квадрант: высокий Trade Value + низкий риск травмы."
        )


# ════════════════════════════════════════════════════════════════════════════════
# BAYESIAN STATISTICS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🧮 Байес":
    page_header(
        "🧮", "БАЙЕСОВСКАЯ СТАТИСТИКА",
        "Beta-Binomial · Normal-Normal · Регуляризация малых выборок · Credible Intervals",
        badge="Bayesian Inference", badge_color="#00D4AA", accent="#00D4AA",
    )
    page_hint([
        "**Что это?** Байесовская коррекция статистики игроков с малым числом матчей — «притягивание» к лиговому среднему.",
        "**Как работает?** 2 модели: Beta-Binomial (FG%, 3P%, FT%, TS%) и Normal-Normal (PTS, AST, REB). Приор = лиговое среднее. После GP матчей апостериорное среднее = κ×лига + (1-κ)×наблюдения, где κ = сила приора / (сила + GP).",
        "**Зачем?** Игрок с FG%=80% за 5 матчей — не снайпер, а статистический шум. Байес снижает его оценку до реалистичных ~47%. Это защищает от переплаты за маленькие выборки.",
        "**Ключевые термины:** Приор — убеждения до данных (лига) · Постериор — обновлённая оценка после GP матчей · Credible Interval — байесовский аналог доверительного интервала · Shrinkage — «стягивание» к среднему.",
    ])

    df_bayes = get_bayesian_stats(_data_hash)
    from src.bayes_stats import (plot_shrinkage_dotplot, plot_shrinkage_scatter,
                                  plot_posterior_distribution, SHOOTING_PRIORS)

    st.html("""
<div style="background:rgba(0,212,170,0.07);border-left:4px solid #00D4AA;
     border-radius:8px;padding:12px 16px;margin-bottom:12px;">
  <b>Байесовская регуляризация</b>: игрок с 5 играми и FG%=60% — действительно ли
  он так хорош? Байес «притягивает» оценки к лиговому среднему (приору).
  Чем меньше матчей — тем сильнее <em>shrinkage</em>.
  Мы используем <b>Beta-Binomial</b> для процентов бросков
  и <b>Normal-Normal conjugate</b> для подсчётных статистик.
</div>""")

    _bayes_opts = {
        "FG_PCT":  "FG% (броски с игры)",
        "FG3_PCT": "3P% (трёхочковые)",
        "FT_PCT":  "FT% (штрафные)",
        "PTS":     "Очки/игра",
        "AST":     "Передачи/игра",
        "REB":     "Подборы/игра",
    }
    _stat_opts = [s for s in _bayes_opts if f"{s}_bayes" in df_bayes.columns]

    tab_dot_b, tab_scat_b, tab_post_b, tab_pl_b = st.tabs([
        "📍 Dotplot регуляризации", "⚖️ Scatter сжатия",
        "📐 Апостериорное распределение", "🔍 Анализ игрока",
    ])

    with tab_dot_b:
        _b1 = st.selectbox("Статистика", _stat_opts,
                           format_func=lambda x: _bayes_opts.get(x, x), key="b_stat1")
        _n_dot = st.slider("Игроков", 10, 50, 25, key="b_n_dot")
        if _b1:
            st.plotly_chart(plot_shrinkage_dotplot(df_bayes, stat=_b1, n=_n_dot),
                            use_container_width=True, key=f"b_dot_{_b1}")
            st.caption("🔴 ромб = байесовская оценка  |  ⚪ кружок = наблюдаемое  "
                       "|  синяя линия = 95% credible interval  |  жёлтая = приор")

    with tab_scat_b:
        _b2 = st.selectbox("Статистика", _stat_opts,
                           format_func=lambda x: _bayes_opts.get(x, x), key="b_stat2")
        if _b2:
            st.plotly_chart(plot_shrinkage_scatter(df_bayes, stat=_b2),
                            use_container_width=True, key=f"b_scat_{_b2}")
            st.caption("Точки ниже диагонали = регуляризованы вниз. "
                       "Цвет = степень сжатия к приору.")

    with tab_post_b:
        _b3_opts = [s for s in _stat_opts if s in SHOOTING_PRIORS]
        if _b3_opts:
            _b3 = st.selectbox("Статистика (проценты)", _b3_opts,
                               format_func=lambda x: _bayes_opts.get(x, x), key="b_stat3")
            _b3_pl = st.selectbox("Игрок", sorted(df[name_col].dropna().unique()),
                                  key="b_pl3")
            if _b3 and _b3_pl:
                _row_b3 = df[df[name_col] == _b3_pl].iloc[0]
                st.plotly_chart(
                    plot_posterior_distribution(_row_b3, stat=_b3),
                    use_container_width=True, key=f"b_post_{_b3_pl}_{_b3}",
                )
        else:
            st.info("Нет доступных процентных статистик.")

    with tab_pl_b:
        section_title("Детальный анализ игрока", "🔍", color="#00D4AA")
        _b4_pl = st.selectbox("Игрок", sorted(df[name_col].dropna().unique()),
                              key="b_pl4")
        if _b4_pl:
            _row_b4  = df[df[name_col] == _b4_pl].iloc[0]
            _row_bay = df_bayes[df_bayes[name_col] == _b4_pl]
            _gp4 = int(_row_b4.get("games_played", _row_b4.get("GP", 30)) or 30)
            st.markdown(f"**Сыграно игр: {_gp4}** — чем меньше, тем сильнее сжатие к приору")
            if not _row_bay.empty:
                _rb4 = _row_bay.iloc[0]
                _items = [(s, _bayes_opts[s]) for s in _stat_opts]
                _bc = st.columns(3)
                for i, (s, lbl) in enumerate(_items[:6]):
                    _mult = 100 if s.endswith("_PCT") else 1
                    _obs  = float(_row_b4.get(s, 0) or 0) * _mult
                    _est  = float(_rb4.get(f"{s}_bayes", _obs)) * _mult
                    _shr  = float(_rb4.get(f"{s}_shrink", 0)) * 100
                    _lo   = float(_rb4.get(f"{s}_ci_low",  _est * 0.95)) * _mult
                    _hi   = float(_rb4.get(f"{s}_ci_high", _est * 1.05)) * _mult
                    with _bc[i % 3]:
                        st.html(f"""
<div style="background:rgba(0,212,170,0.06);border:1px solid rgba(0,212,170,0.2);
     border-radius:10px;padding:12px;margin:4px 0;">
  <div style="font-size:10px;color:#00D4AA;font-weight:800;
              text-transform:uppercase;letter-spacing:1px;">{lbl}</div>
  <div style="font-size:24px;font-weight:900;color:#fff;margin:4px 0;">{_est:.2f}</div>
  <div style="font-size:11px;color:#8b9ab5;">Наблюдаемое: {_obs:.2f}</div>
  <div style="font-size:11px;color:#8b9ab5;">95% CI: [{_lo:.2f}, {_hi:.2f}]</div>
  <div style="font-size:11px;color:#00D4AA;">Сжатие к приору: {_shr:.0f}%</div>
</div>""")


# ════════════════════════════════════════════════════════════════════════════════
# NETWORK ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🕸 Сеть игроков":
    page_header(
        "🕸", "СЕТЕВОЙ АНАЛИЗ СТИЛЕЙ",
        "Граф косинусного сходства · PageRank · Сообщества · Betweenness Centrality",
        badge="Graph Theory", badge_color="#3498DB", accent="#3498DB",
    )
    page_hint([
        "**Что это?** Граф игроков: рёбра = статистическое сходство, узлы = игроки. Позволяет найти «клонов», влиятельные архетипы и изолированных уникумов.",
        "**Как работает?** cosine_similarity(X_scaled) → 456×456 матрица → ребро если sim > 0.92 → ~1 400 рёбер. PageRank реализован с нуля на NumPy (d=0.85). Сообщества найдены жадным поиском модульности (6 групп).",
        "**Зачем?** Найти замену игроку на трейде: «кто статистически похож на LeBron?» Centrality показывает «типичных представителей» архетипа — идеальных таргетов для замены.",
        "**Ключевые термины:** Косинусная схожесть ∈ [0,1] — 1 = идентичный стиль · PageRank — мера «типичности» для своего архетипа · Betweenness — игроки-«мосты» между разными стилями.",
    ])

    _sim_mat, _net_names, _net_metrics = get_network_data(_data_hash)
    from src.network import (plot_network_graph, plot_centrality_leaderboard,
                              plot_similarity_clusters, SIM_THRESHOLD)

    kpi_row([
        {"icon": "👥", "label": "Узлов в сети",    "value": str(len(_net_names))},
        {"icon": "🔗", "label": "Порог сходства",  "value": f"{SIM_THRESHOLD:.0%}"},
        {"icon": "⭐", "label": "Макс. PageRank",
         "value": f"{_net_metrics['pagerank'].max():.2f}"},
        {"icon": "🌐", "label": "Сообществ",
         "value": str(_net_metrics['net_community'].nunique())},
        {"icon": "🔢", "label": "Макс. связей",
         "value": str(int(_net_metrics['n_similar'].max()))},
    ])

    tab_graph_n, tab_central_n, tab_heat_n = st.tabs([
        "🌐 Граф", "🏆 Центральность", "🔥 Матрица сходства",
    ])

    with tab_graph_n:
        col_g1, col_g2 = st.columns([4, 1])
        with col_g2:
            _max_nodes = st.slider("Узлов", 20, 80, 60, 10, key="net_n")
        with col_g1:
            with st.spinner("Строим граф..."):
                _fig_net = plot_network_graph(
                    _sim_mat, _net_names, _net_metrics, max_nodes=_max_nodes
                )
            st.plotly_chart(_fig_net, use_container_width=True, key="net_graph")
        st.caption(
            f"Размер узла — PageRank (влияние). Цвет — тип игрока. "
            f"Рёбра — сходство ≥ {SIM_THRESHOLD:.0%}."
        )

    with tab_central_n:
        _nm_sel = st.selectbox(
            "Метрика",
            ["pagerank", "degree", "n_similar"],
            format_func=lambda x: {
                "pagerank":  "PageRank — стилистическое влияние",
                "degree":    "Степень — доля похожих игроков",
                "n_similar": "Кол-во прямых связей",
            }.get(x, x),
            key="net_metric",
        )
        _top_n_net = st.slider("Топ-N", 10, 40, 20, key="net_topn")
        st.plotly_chart(
            plot_centrality_leaderboard(_net_metrics, metric=_nm_sel, n=_top_n_net),
            use_container_width=True, key="net_central",
        )
        st.caption(
            "PageRank: высокое значение → игрок находится в центре большого кластера "
            "похожих игроков — его стиль наиболее типичен для лиги."
        )

    with tab_heat_n:
        _n_heat_n = st.slider("Игроков в матрице", 20, 60, 35, 5, key="net_heat_n")
        with st.spinner("Строим матрицу..."):
            _fig_hnet = plot_similarity_clusters(
                _sim_mat, _net_names, _net_metrics, n=_n_heat_n
            )
        st.plotly_chart(_fig_hnet, use_container_width=True, key="net_heatmap")
        st.caption("Блоки на диагонали = сообщества игроков схожего стиля.")


# ════════════════════════════════════════════════════════════════════════════════
# SURVIVAL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "⚰️ Выживаемость":
    page_header(
        "⚰️", "АНАЛИЗ ВЫЖИВАЕМОСТИ КАРЬЕРЫ",
        "Kaplan-Meier · Cox Proportional Hazards · Медиана пика · Факторы риска спада",
        badge="Survival Analysis", badge_color="#E74C3C", accent="#E74C3C",
    )
    page_hint([
        "**Что это?** Анализ «выживаемости» карьеры: с какой вероятностью игрок сохранит пиковую эффективность через 1, 2, 3 сезона?",
        "**Как работает?** «Событие» = снижение efficiency ≥15% от пика. Kaplan-Meier (реализован с нуля) оценивает S(t) — вероятность дожить до момента t без спада. Cox PH модель (аппрокс. LogisticReg + 200 bootstrap) даёт Hazard Ratio факторов.",
        "**Зачем?** Продлевать контракт имеет смысл только если S(3 сезона) > 50%. HR > 1.1 означает, что фактор (возраст, нагрузка) ускоряет спад — учитывайте при переговорах.",
        "**Ключевые термины:** S(t) — вероятность остаться на пике · KM — непараметрическая кривая выживаемости · HR — Hazard Ratio: >1 ускоряет спад, <1 защищает · Медиана выживаемости — сезон, когда S(t) = 50%.",
    ])

    _df_surv, _cox_df = get_survival_data(_data_hash)
    from src.survival import plot_km_by_group, plot_cox_forest, plot_survival_heatmap

    st.html("""
<div style="background:rgba(231,76,60,0.07);border-left:4px solid #E74C3C;
     border-radius:8px;padding:12px 16px;margin-bottom:12px;">
  <b>Kaplan-Meier</b> оценивает вероятность того, что игрок сохранит пиковую
  производительность через время <em>t</em>. Цензурированные наблюдения (игроки
  ещё не достигли спада) учитываются корректно.
  <b>Cox PH</b> показывает, какие факторы значимо увеличивают или снижают риск
  карьерного снижения через Hazard Ratio.
</div>""")

    kpi_row([
        {"icon": "👥", "label": "Игроков в анализе",  "value": str(len(_df_surv))},
        {"icon": "⚠️", "label": "Событий (спад)",
         "value": str(int(_df_surv["observed"].sum()))},
        {"icon": "🛡",  "label": "Цензурированных",
         "value": str(int((_df_surv["observed"] == 0).sum()))},
        {"icon": "📊", "label": "Медиана времени",
         "value": f"{_df_surv['time'].median():.2f}"},
        {"icon": "🎂", "label": "Средний возраст",
         "value": f"{_df_surv['age'].mean():.1f}" if "age" in _df_surv.columns else "—"},
    ])

    tab_risk_km, tab_age_km, tab_arch_km, tab_cox_km, tab_heat_km = st.tabs([
        "🔴 По уровню риска", "📅 По возрасту",
        "🗂 По архетипу", "🌲 Cox Forest Plot", "🔥 Тепловая карта",
    ])

    with tab_risk_km:
        st.plotly_chart(
            plot_km_by_group(_df_surv, group_col="risk_group"),
            use_container_width=True, key="km_risk",
        )
        st.caption(
            "T₅₀ = медианное время выживаемости. "
            "Чем дольше кривая остаётся высокой — тем дольше пик карьеры."
        )

    with tab_age_km:
        st.plotly_chart(
            plot_km_by_group(_df_surv, group_col="age_group"),
            use_container_width=True, key="km_age",
        )

    with tab_arch_km:
        if "cluster_name" in _df_surv.columns:
            st.plotly_chart(
                plot_km_by_group(_df_surv, group_col="cluster_name"),
                use_container_width=True, key="km_cluster",
            )
        else:
            st.info("Данные кластеров недоступны.")

    with tab_cox_km:
        if _cox_df is not None and not _cox_df.empty:
            col_cx1, col_cx2 = st.columns([2, 1])
            with col_cx1:
                st.plotly_chart(
                    plot_cox_forest(_cox_df),
                    use_container_width=True, key="cox_forest",
                )
            with col_cx2:
                section_title("Hazard Ratios", "📋", color="#E74C3C")
                st.dataframe(_cox_df, use_container_width=True)
                st.caption(
                    "HR > 1.1 → увеличивает риск спада.  \n"
                    "HR < 0.9 → снижает риск.  \n"
                    "95% CI не должен включать HR=1 для значимости."
                )
        else:
            st.info("Недостаточно данных для Cox PH модели.")

    with tab_heat_km:
        st.plotly_chart(
            plot_survival_heatmap(_df_surv),
            use_container_width=True, key="surv_heatmap",
        )
        st.caption(
            "Медиана симулированного времени до снижения эффективности "
            "по группам возраста и риска."
        )


# ════════════════════════════════════════════════════════════════════════════════
# AI SCOUT REPORT
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Скаут":
    from src.ai_analyst import (REPORT_TYPES, build_scout_prompt,
                                 build_trade_prompt, extract_verdict,
                                 trade_balance_html)

    page_header(
        "🤖", "AI СКАУТ-РЕПОРТ",
        "Gemini 2.5 Flash · 4 типа отчётов · Профессиональный скаутинговый анализ",
        badge="Powered by Gemini", badge_color="#9B59B6", accent="#9B59B6",
    )
    page_hint([
        "**Что это?** AI-скаутинговый отчёт: Gemini 2.5 Flash генерирует профессиональный анализ игрока на основе его статпрофиля, кластера, Trade Value и Injury Risk.",
        "**Как работает?** Промпт содержит полный статпрофиль + средние по кластеру + TV + IR + системную роль «ведущий NBA аналитик». Gemini возвращает структурированный русскоязычный отчёт за 3–5 секунд.",
        "**Зачем?** Ручной анализ одного игрока занимает 30–60 минут эксперта. AI делает это мгновенно и без субъективности. 4 типа отчётов покрывают все задачи GM: скаутинг, контракт, прогноз карьеры, сравнение.",
        "**Типы отчётов:** 🔍 Полный скаутинг — все стороны игры · 💰 Контрактная оценка — справедливая зарплата · 📈 Прогноз карьеры — потолок и исторические аналоги · ⚔️ Сравнение — head-to-head с AI-вердиктом.",
    ])

    df_tv_ai = get_trade_values(_data_hash)
    df_ir_ai = get_injury_risks(_data_hash)
    # Merge TV + injury_risk into one enriched df
    _ai_df = df_tv_ai.copy()
    if "injury_risk" not in _ai_df.columns and "injury_risk" in df_ir_ai.columns:
        _ai_df["injury_risk"] = df_ir_ai["injury_risk"].values

    all_players_ai = sorted(_ai_df[name_col].dropna().unique())

    # ── controls ──────────────────────────────────────────────────────────────
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1])
    with col_ctrl1:
        ai_player = st.selectbox("Игрок", all_players_ai, key="ai_scout_player")
    with col_ctrl2:
        ai_report_type_label = st.selectbox(
            "Тип отчёта", list(REPORT_TYPES.keys()), key="ai_scout_type"
        )
        ai_report_type = REPORT_TYPES[ai_report_type_label]
    with col_ctrl3:
        st.html("<div style='height:28px'></div>")
        ai_generate = st.button(
            "✨ Сгенерировать", type="primary", key="ai_scout_btn",
            use_container_width=True,
        )

    # Second player selector (only for compare type)
    ai_compare_player = None
    if ai_report_type == "compare":
        others = [p for p in all_players_ai if p != ai_player]
        ai_compare_player = st.selectbox(
            "Сравнить с", others, key="ai_scout_compare_player"
        )

    ai_extra_ctx = st.text_input(
        "Дополнительный контекст (необязательно)",
        placeholder="Например: 'Рассматриваем под роль 3-and-D на ветеранский минимум'",
        key="ai_scout_ctx",
    )

    # ── player snapshot card ──────────────────────────────────────────────────
    if ai_player:
        _ai_row = _ai_df[_ai_df[name_col] == ai_player].iloc[0]
        snap_c1, snap_c2, snap_c3, snap_c4, snap_c5 = st.columns(5)
        snap_c1.metric("⚡ Trade Value",  f"{_ai_row.get('trade_value', 0):.1f}")
        snap_c2.metric("🏀 Очки",         f"{_ai_row.get('PTS', 0):.1f}")
        snap_c3.metric("🎯 TS%",           f"{_ai_row.get('TS_PCT', 0)*100:.1f}%")
        snap_c4.metric("🔥 USG%",          f"{_ai_row.get('USG_PCT', 0)*100:.1f}%")
        snap_c5.metric("⚠️ Риск травмы",  f"{_ai_row.get('injury_risk', 0):.1f}%")

    # ── generate ──────────────────────────────────────────────────────────────
    if ai_generate:
        try:
            _ai_row = _ai_df[_ai_df[name_col] == ai_player].iloc[0]
            _cluster_avg = _ai_df[
                _ai_df["cluster_name"] == _ai_row.get("cluster_name", "")
            ].mean(numeric_only=True)

            _compare_row = None
            if ai_report_type == "compare" and ai_compare_player:
                _compare_matches = _ai_df[_ai_df[name_col] == ai_compare_player]
                if not _compare_matches.empty:
                    _compare_row = _compare_matches.iloc[0]

            prompt = build_scout_prompt(
                _ai_row, _cluster_avg,
                report_type=ai_report_type,
                compare_row=_compare_row,
                extra_context=ai_extra_ctx,
            )

            with st.spinner("🤖 Gemini анализирует..."):
                _client = google_genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                _resp = _client.models.generate_content(
                    model="gemini-2.5-flash", contents=prompt
                )
                report_text = _resp.text

            # Save to session history
            if "ai_scout_history" not in st.session_state:
                st.session_state.ai_scout_history = []
            st.session_state.ai_scout_history.insert(0, {
                "player": ai_player,
                "type":   ai_report_type_label,
                "report": report_text,
            })
            st.session_state.ai_scout_history = st.session_state.ai_scout_history[:5]

        except KeyError:
            st.error("🔑 GEMINI_API_KEY не найден в `.streamlit/secrets.toml`")
            st.code("[secrets]\nGEMINI_API_KEY = 'your-key-here'")
        except Exception as _e:
            st.error(f"Ошибка Gemini: {_e}")

    # ── render history ────────────────────────────────────────────────────────
    history = st.session_state.get("ai_scout_history", [])
    if history:
        for _idx, _entry in enumerate(history):
            _is_latest = (_idx == 0)
            _expander_label = (
                f"{'📄' if _is_latest else '🗂'} "
                f"{_entry['player']} — {_entry['type']}"
                + (" (последний)" if _is_latest else "")
            )
            with st.expander(_expander_label, expanded=_is_latest):
                st.html(
                    f"""<div style="
                        background:linear-gradient(135deg,#0f1520,#0c1220);
                        border:1px solid rgba(155,89,182,0.25);
                        border-radius:12px; padding:24px 28px;
                        font-family:'Inter',sans-serif; line-height:1.7;
                        color:#e8eaf6; font-size:14px;
                    ">{_entry['report']}</div>"""
                )
                st.download_button(
                    "📥 Скачать", _entry["report"],
                    file_name=f"scout_{_entry['player'].replace(' ','_')}_{_idx}.txt",
                    mime="text/plain",
                    key=f"ai_dl_{_idx}",
                )
    else:
        st.html(
            """<div style="text-align:center; padding:60px 20px;
                           color:#4a566e; font-size:15px;">
                🤖 Выберите игрока и тип отчёта, затем нажмите <b>Сгенерировать</b>
            </div>"""
        )

    # ── sidebar: report type descriptions ────────────────────────────────────
    with st.sidebar.expander("📖 Типы отчётов", expanded=False):
        st.markdown("""
**🔍 Полный скаутинг** — сильные/слабые стороны, роль, TV оценка

**💰 Контрактная оценка** — справедливая зарплата, риски, рекомендация GM

**📈 Прогноз карьеры** — фаза карьеры, потолок/дно, исторические аналоги

**⚔️ Сравнение** — прямое сравнение двух игроков с вердиктом
        """)


# ════════════════════════════════════════════════════════════════════════════════
# AI TRADE ANALYST
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔄 AI Трейд":
    from src.ai_analyst import (build_trade_prompt, extract_verdict,
                                 trade_balance_html)

    page_header(
        "🔄", "AI ТРЕЙД-АНАЛИТИК",
        "Trade Machine · Баланс ценностей · Gemini-вердикт · Стратегический анализ",
        badge="Trade Machine", badge_color="#E67E22", accent="#E67E22",
    )
    page_hint([
        "**Что это?** Симулятор трейда: выберите до 4 игроков с каждой стороны → получите TV-баланс и AI-анализ выгодности обмена.",
        "**Как работает?** ΔTV = TV_отдаёшь − TV_получаешь → визуальный прогресс-бар. Gemini 2.5 Flash анализирует обмен в 5 секциях: ценностный баланс, командная логика, риски, альтернативы, вердикт.",
        "**Зачем?** Быстро проверить любую трейд-идею: кто выигрывает, какие риски, есть ли лучшие варианты. Вердикт автоматически: ✅ Team A получает больше / 🔄 Нейтрально / ⚠️ Осторожно.",
        "**Ключевые термины:** ΔTV — разница Trade Value: отрицательный = вы в выигрыше · Вердикт: ✅ выгодно · 🔄 нейтрально · ⚠️ осторожно (красные флаги в анализе).",
    ])

    df_tv_tr = get_trade_values(_data_hash)
    df_ir_tr = get_injury_risks(_data_hash)
    _tr_df = df_tv_tr.copy()
    if "injury_risk" not in _tr_df.columns and "injury_risk" in df_ir_tr.columns:
        _tr_df["injury_risk"] = df_ir_tr["injury_risk"].values

    all_tr_players = sorted(_tr_df[name_col].dropna().unique())

    # ── Team name inputs ───────────────────────────────────────────────────────
    tn_col1, tn_col2 = st.columns(2)
    with tn_col1:
        team_a_name = st.text_input("Название команды A", value="Команда A", key="tr_team_a")
    with tn_col2:
        team_b_name = st.text_input("Название команды B", value="Команда B", key="tr_team_b")

    # ── Player selectors ───────────────────────────────────────────────────────
    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        st.html(
            f"""<div style="background:rgba(200,16,46,0.08); border:1px solid rgba(200,16,46,0.25);
                border-radius:10px; padding:12px 16px; margin-bottom:8px;
                font-size:13px; color:#e8eaf6; font-weight:600">
                🔴 {team_a_name} — отдаёт
            </div>"""
        )
        players_out_names = st.multiselect(
            "Выберите игроков (макс. 4)", all_tr_players,
            max_selections=4, key="tr_players_out",
        )
    with sel_col2:
        st.html(
            f"""<div style="background:rgba(29,66,138,0.12); border:1px solid rgba(29,66,138,0.3);
                border-radius:10px; padding:12px 16px; margin-bottom:8px;
                font-size:13px; color:#e8eaf6; font-weight:600">
                🔵 {team_b_name} — получает
            </div>"""
        )
        players_in_names = st.multiselect(
            "Выберите игроков (макс. 4)",
            [p for p in all_tr_players if p not in players_out_names],
            max_selections=4, key="tr_players_in",
        )

    # ── Trade Value balance ────────────────────────────────────────────────────
    _rows_out = [_tr_df[_tr_df[name_col] == p].iloc[0]
                 for p in players_out_names if p in _tr_df[name_col].values]
    _rows_in  = [_tr_df[_tr_df[name_col] == p].iloc[0]
                 for p in players_in_names  if p in _tr_df[name_col].values]

    tv_out = sum(float(r.get("trade_value", 0)) for r in _rows_out)
    tv_in  = sum(float(r.get("trade_value", 0)) for r in _rows_in)
    tv_bal = tv_in - tv_out

    if players_out_names or players_in_names:
        st.html(trade_balance_html(tv_out, tv_in))

        # KPI row
        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric(
            f"TV отдаём ({team_a_name})", f"{tv_out:.1f}",
            delta=None,
        )
        kc2.metric(
            f"TV получаем ({team_b_name})", f"{tv_in:.1f}",
            delta=f"{tv_bal:+.1f}",
            delta_color="normal",
        )
        _avg_age_out = np.mean([r.get("AGE", 27) for r in _rows_out]) if _rows_out else 0
        _avg_age_in  = np.mean([r.get("AGE", 27) for r in _rows_in])  if _rows_in  else 0
        kc3.metric("Ср. возраст → отдаём", f"{_avg_age_out:.1f}" if _rows_out else "—")
        kc4.metric("Ср. возраст → получаем", f"{_avg_age_in:.1f}" if _rows_in else "—",
                   delta=f"{_avg_age_in - _avg_age_out:+.1f}" if (_rows_out and _rows_in) else None)

        # Player mini-cards side by side
        if _rows_out or _rows_in:
            section_title("Детали обмена", "📋", color="#E67E22")
            card_cols = st.columns(2)

            def _mini_card(row: pd.Series, accent: str) -> str:
                pname = str(row.get("PLAYER_NAME") or row.get("name") or "?")
                tm    = str(row.get("TEAM_ABBREVIATION") or row.get("team") or "—")
                age   = f"{row.get('AGE', '?'):.0f}" if isinstance(row.get("AGE"), float) else "?"
                tv    = f"{row.get('trade_value', 0):.1f}"
                tier  = str(row.get("tv_tier", ""))
                pts   = f"{row.get('PTS', 0):.1f}"
                ast   = f"{row.get('AST', 0):.1f}"
                reb   = f"{row.get('REB', 0):.1f}"
                risk  = f"{row.get('injury_risk', 0):.0f}"
                cls   = str(row.get("cluster_name", ""))
                return (
                    f"<div style='background:#0f1520; border:1px solid {accent}33; "
                    f"border-radius:10px; padding:12px 14px; margin-bottom:8px;'>"
                    f"<div style='font-weight:700; color:#e8eaf6; font-size:14px'>{pname}</div>"
                    f"<div style='font-size:11px; color:#6b7a99; margin:2px 0 6px'>"
                    f"{tm} · {age} лет · {cls}</div>"
                    f"<div style='font-size:13px; color:#8b9ab5'>"
                    f"TV: <b style='color:{accent}'>{tv}/100</b> {tier} &nbsp;|&nbsp; "
                    f"⚠️ Risk: {risk}%</div>"
                    f"<div style='font-size:12px; color:#6b7a99; margin-top:4px'>"
                    f"{pts} pts · {ast} ast · {reb} reb</div>"
                    f"</div>"
                )

            with card_cols[0]:
                for r in _rows_out:
                    st.html(_mini_card(r, "#C8102E"))
            with card_cols[1]:
                for r in _rows_in:
                    st.html(_mini_card(r, "#1D428A"))

    # ── Analyse button ─────────────────────────────────────────────────────────
    st.markdown("---")
    can_analyse = len(_rows_out) > 0 and len(_rows_in) > 0
    if not can_analyse:
        st.info("Выберите хотя бы одного игрока с каждой стороны для анализа.")

    tr_btn = st.button(
        "🤖 Анализировать сделку", type="primary",
        key="tr_analyse_btn", disabled=not can_analyse,
        use_container_width=False,
    )

    if tr_btn and can_analyse:
        try:
            _tr_prompt = build_trade_prompt(
                team_a_name, _rows_out,
                team_b_name, _rows_in,
                tv_balance=tv_bal,
            )
            with st.spinner("🤖 Gemini анализирует сделку..."):
                _client_tr = google_genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                _resp_tr = _client_tr.models.generate_content(
                    model="gemini-2.5-flash", contents=_tr_prompt
                )
                _tr_report = _resp_tr.text

            _verdict_emoji, _verdict_label = extract_verdict(_tr_report)

            # Save to session state
            if "ai_trade_history" not in st.session_state:
                st.session_state.ai_trade_history = []
            st.session_state.ai_trade_history.insert(0, {
                "label":   f"{', '.join(players_out_names)} ↔ {', '.join(players_in_names)}",
                "report":  _tr_report,
                "verdict": (_verdict_emoji, _verdict_label),
                "tv_out":  tv_out,
                "tv_in":   tv_in,
            })
            st.session_state.ai_trade_history = st.session_state.ai_trade_history[:4]

        except KeyError:
            st.error("🔑 GEMINI_API_KEY не найден в `.streamlit/secrets.toml`")
        except Exception as _e:
            st.error(f"Ошибка Gemini: {_e}")

    # ── Trade history ──────────────────────────────────────────────────────────
    _tr_history = st.session_state.get("ai_trade_history", [])
    for _ti, _te in enumerate(_tr_history):
        _ve, _vl = _te["verdict"]
        _exp_label = (
            f"{_ve} {_te['label']}  |  "
            f"TV: {_te['tv_out']:.1f} → {_te['tv_in']:.1f}  ({_vl})"
            + ("  ← последний" if _ti == 0 else "")
        )
        with st.expander(_exp_label, expanded=(_ti == 0)):
            # Verdict badge
            _badge_color = (
                "#27AE60" if _ve == "✅" else
                "#E67E22" if _ve in ("⚠️", "🔄") else
                "#6b7a99"
            )
            st.html(
                f"""<div style="display:inline-block; background:{_badge_color}22;
                    border:1px solid {_badge_color}55; border-radius:20px;
                    padding:4px 16px; margin-bottom:12px;
                    font-size:14px; font-weight:700; color:{_badge_color}">
                    {_ve} {_vl}
                </div>"""
            )
            st.html(
                f"""<div style="
                    background:linear-gradient(135deg,#0f1520,#0c1220);
                    border:1px solid rgba(230,126,34,0.2);
                    border-radius:12px; padding:24px 28px;
                    font-family:'Inter',sans-serif; line-height:1.7;
                    color:#e8eaf6; font-size:14px;
                ">{_te['report']}</div>"""
            )
            st.download_button(
                "📥 Скачать анализ", _te["report"],
                file_name=f"trade_analysis_{_ti}.txt",
                mime="text/plain",
                key=f"tr_dl_{_ti}",
            )
