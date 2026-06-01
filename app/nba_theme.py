"""
NBA Analytics — World-Class Dark Theme
Plotly template + Streamlit CSS + HTML component builders
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

# ── Palette ───────────────────────────────────────────────────────────────────
NBA_RED    = "#C8102E"
NBA_BLUE   = "#1D428A"
NBA_GOLD   = "#FFC72C"
BG_APP     = "#080c14"
BG_CARD    = "#0f1520"
BG_CHART   = "#0c1220"
BG_HOVER   = "#162035"
TEXT_MAIN  = "#E8EAF6"
TEXT_SUB   = "#8B9AB5"
TEXT_MUTED = "#4A566E"
GRID_LINE  = "#141E30"
BORDER_DIM = "rgba(255,255,255,0.06)"
BORDER_RED = "rgba(200,16,46,0.22)"

COLOR_SEQ = [
    "#C8102E","#1D428A","#FFC72C","#00D4AA",
    "#9B59B6","#E67E22","#2ECC71","#3498DB",
    "#E74C3C","#F0A500",
]

# ── Plotly template ───────────────────────────────────────────────────────────
def register_nba_template():
    from src.theme_utils import ensure_template as _ensure
    _ensure()


def _register_nba_template_full():
    """Full registration (kept for reference, delegates to src.theme_utils)."""
    tpl = go.layout.Template()
    tpl.layout.update(
        plot_bgcolor  = BG_CHART,
        paper_bgcolor = BG_CARD,
        colorway      = COLOR_SEQ,
        font=dict(family="'Inter','Segoe UI',Arial,sans-serif", color=TEXT_MAIN, size=12),
        title=dict(font=dict(size=16, color=TEXT_MAIN,
                             family="'Inter','Segoe UI',Arial,sans-serif"), x=0.02),
        xaxis=dict(gridcolor=GRID_LINE, gridwidth=1, linecolor=GRID_LINE,
                   zerolinecolor=GRID_LINE,
                   tickfont=dict(color=TEXT_MUTED, size=11),
                   title_font=dict(color=TEXT_SUB, size=12),
                   showspikes=True, spikecolor=NBA_RED, spikethickness=1, spikedash="dot"),
        yaxis=dict(gridcolor=GRID_LINE, gridwidth=1, linecolor=GRID_LINE,
                   zerolinecolor=GRID_LINE,
                   tickfont=dict(color=TEXT_MUTED, size=11),
                   title_font=dict(color=TEXT_SUB, size=12)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_MAIN, size=11),
                    bordercolor=BORDER_DIM, borderwidth=1),
        hoverlabel=dict(bgcolor=BG_HOVER, bordercolor=NBA_RED,
                        font=dict(color="white", size=12,
                                  family="'Segoe UI',sans-serif")),
        polar=dict(bgcolor=BG_CHART,
                   radialaxis=dict(gridcolor=GRID_LINE, tickfont=dict(color=TEXT_MUTED)),
                   angularaxis=dict(gridcolor=GRID_LINE, tickfont=dict(color=TEXT_MUTED))),
        margin=dict(t=55, b=40, l=50, r=20),
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=TEXT_MUTED, activecolor=NBA_RED),
        coloraxis=dict(
            colorbar=dict(
                bgcolor="rgba(0,0,0,0)",
                bordercolor=BORDER_DIM,
                tickfont=dict(color=TEXT_MUTED, size=10),
                title=dict(font=dict(color=TEXT_SUB, size=11)),
                thickness=10, len=0.75,
            )
        ),
        annotationdefaults=dict(
            font=dict(color=TEXT_MUTED, size=10),
            arrowcolor=BORDER_DIM,
        ),
    )
    pio.templates["nba_dark"] = tpl
    pio.templates.default = "nba_dark"


# ── HTML Components ───────────────────────────────────────────────────────────
def page_header(icon: str, title: str, subtitle: str,
                badge: str = None, badge_color: str = "#C8102E",
                accent: str = "#C8102E") -> None:
    badge_html = ""
    if badge:
        badge_html = (
            f'<span style="display:inline-block;background:{badge_color};'
            f'color:#fff;border-radius:20px;padding:3px 14px;font-size:11px;'
            f'font-weight:700;letter-spacing:0.8px;margin-top:10px;">{badge}</span>'
        )
    st.html(f"""
<div style="
    background:linear-gradient(135deg,{BG_CARD} 0%,#0d1825 60%,{BG_CARD} 100%);
    border:1px solid {BORDER_RED};
    border-left:4px solid {accent};
    border-radius:16px;
    padding:24px 32px;
    margin-bottom:20px;
    position:relative;
    overflow:hidden;
">
  <div style="position:absolute;top:-40px;right:-40px;width:200px;height:200px;
              background:radial-gradient(circle,rgba(200,16,46,0.08) 0%,transparent 70%);
              pointer-events:none;"></div>
  <div style="display:flex;align-items:center;gap:18px;">
    <div style="font-size:40px;line-height:1;filter:drop-shadow(0 2px 8px rgba(200,16,46,0.4));">
      {icon}
    </div>
    <div>
      <div style="font-size:10px;font-weight:800;color:{accent};
                  text-transform:uppercase;letter-spacing:2px;margin-bottom:4px;">
        NBA Analytics · Сезон 2025-26
      </div>
      <div style="font-family:'Barlow Condensed','Arial Black',sans-serif;
                  font-size:28px;font-weight:900;color:#fff;
                  letter-spacing:-0.5px;line-height:1.1;">
        {title}
      </div>
      <div style="font-size:13px;color:{TEXT_SUB};margin-top:5px;">{subtitle}</div>
      {badge_html}
    </div>
  </div>
</div>
""")


def kpi_row(items: list[dict]) -> None:
    """
    items = [{"label":"...", "value":"...", "delta":"...", "icon":"..."}]
    """
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        icon  = item.get("icon", "")
        label = item.get("label", "")
        value = item.get("value", "—")
        delta = item.get("delta", "")
        color = item.get("color", NBA_RED)
        delta_color = "#27ae60" if str(delta).startswith("+") else \
                      "#e74c3c" if str(delta).startswith("-") else TEXT_MUTED
        delta_html = (f'<div style="font-size:11px;color:{delta_color};'
                      f'font-weight:700;margin-top:2px;">{delta}</div>') if delta else ""
        col.html(f"""
<div style="background:linear-gradient(135deg,rgba(29,66,138,0.15),rgba(200,16,46,0.08));
            border:1px solid {BORDER_RED};border-radius:14px;
            padding:16px 20px;transition:transform 0.15s,box-shadow 0.15s;
            box-shadow:0 2px 12px rgba(0,0,0,0.3);">
  <div style="font-size:11px;color:{TEXT_MUTED};font-weight:700;
              text-transform:uppercase;letter-spacing:0.9px;margin-bottom:8px;">
    {icon} {label}
  </div>
  <div style="font-family:'Barlow Condensed','Arial Black',sans-serif;
              font-size:2rem;font-weight:900;color:#fff;line-height:1;">
    {value}
  </div>
  {delta_html}
</div>
""")


def section_title(text: str, icon: str = "", color: str = NBA_RED) -> None:
    st.html(f"""
<div style="display:flex;align-items:center;gap:10px;margin:20px 0 10px;">
  <div style="width:4px;height:24px;background:{color};border-radius:2px;flex-shrink:0;"></div>
  <span style="font-family:'Barlow Condensed','Arial Black',sans-serif;
               font-size:16px;font-weight:900;color:#e8eaf6;
               text-transform:uppercase;letter-spacing:1px;">{icon} {text}</span>
</div>
""")


def player_hero_card(name: str, team: str, cluster: str,
                     photo_url: str = None) -> None:
    photo_html = ""
    if photo_url:
        photo_html = f"""
<img src="{photo_url}" style="width:110px;height:auto;border-radius:12px;
     border:2px solid rgba(200,16,46,0.4);
     box-shadow:0 4px 20px rgba(0,0,0,0.6);object-fit:cover;">"""
    st.html(f"""
<div style="background:linear-gradient(135deg,#091523 0%,#1a0512 100%);
            border:1px solid rgba(200,16,46,0.25);border-radius:20px;
            padding:24px;text-align:center;
            box-shadow:0 8px 32px rgba(0,0,0,0.5);">
  {photo_html}
  <div style="font-family:'Barlow Condensed',sans-serif;font-size:22px;
              font-weight:900;color:#fff;margin-top:10px;">{name}</div>
  <div style="color:#8b9ab5;font-size:13px;margin-top:3px;">{team}</div>
  <div style="display:inline-block;
              background:linear-gradient(135deg,rgba(200,16,46,0.25),rgba(29,66,138,0.25));
              border:1px solid rgba(200,16,46,0.3);border-radius:20px;
              padding:4px 14px;font-size:11px;font-weight:700;
              color:#e8eaf6;margin-top:8px;letter-spacing:0.5px;">{cluster}</div>
</div>
""")


# ── CSS ───────────────────────────────────────────────────────────────────────
NBA_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Barlow+Condensed:wght@600;700;900&display=swap');

/* ═══ HIDE STREAMLIT CHROME ══════════════════════════════════════════════════ */
[data-testid="stHeader"],
[data-testid="stDecoration"],
[data-testid="stToolbar"],
[data-testid="stStatusWidget"],
#MainMenu,
header { display: none !important; height: 0 !important; }
footer { display: none !important; }

/* ═══ PAGE FADE-IN ANIMATION ═════════════════════════════════════════════════ */
@keyframes nbaFadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ═══ BASE ═══════════════════════════════════════════════════════════════════ */
html, body, [class*="css"], .stApp {
    background-color: #080c14 !important;
    font-family: 'Inter','Segoe UI',Arial,sans-serif !important;
    color: #E8EAF6 !important;
}
.main .block-container {
    padding: 1rem 2rem 3rem !important;
    max-width: 1480px !important;
    animation: nbaFadeUp 0.32s ease-out both !important;
}
/* Subtle dot-grid background on main area */
.main {
    background-image: radial-gradient(rgba(200,16,46,0.04) 1px, transparent 1px) !important;
    background-size: 28px 28px !important;
}

/* ═══ SIDEBAR ════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #040810 !important;
    border-right: 1px solid rgba(200,16,46,0.18) !important;
    min-width: 260px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0.5rem 0.75rem 1rem !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not([class*="badge"]),
[data-testid="stSidebar"] small {
    color: #8b9ab5 !important;
}
[data-testid="stSidebar"] hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg,rgba(200,16,46,0.4),rgba(29,66,138,0.3),transparent) !important;
    margin: 8px 0 !important;
}

/* ── Sidebar selectbox / slider ── */
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.08) !important;
    border-radius: 9px !important;
    color: #c8d0e0 !important;
    font-size: 13px !important;
}
[data-testid="stSidebar"] .stSlider [role="slider"] {
    background: #C8102E !important;
    border: 2px solid #fff !important;
}
[data-testid="stSidebar"] .stSlider > label {
    font-size: 12px !important;
    color: #6b7a99 !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

/* ═══ SIDEBAR CATEGORY NAV (EXPANDERS) ═══════════════════════════════════════ */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: transparent !important;
    border: none !important;
    margin: 2px 0 !important;
    box-shadow: none !important;
}
/* Category header button */
[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stExpander"] > details > summary {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    font-family: 'Barlow Condensed','Arial Black',sans-serif !important;
    font-size: 10px !important;
    font-weight: 900 !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    color: #3a4d65 !important;
    transition: all 0.18s ease !important;
    cursor: pointer !important;
    user-select: none !important;
    list-style: none !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    background: rgba(200,16,46,0.07) !important;
    border-color: rgba(200,16,46,0.2) !important;
    color: #8b9ab5 !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] details[open] summary,
[data-testid="stSidebar"] details[open] > summary {
    background: rgba(200,16,46,0.05) !important;
    border-color: rgba(200,16,46,0.22) !important;
    color: #C8102E !important;
}
/* Arrow icon */
[data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
    width: 12px !important;
    height: 12px !important;
    opacity: 0.4 !important;
    color: inherit !important;
}
[data-testid="stSidebar"] details[open] summary svg {
    opacity: 0.9 !important;
}
/* Expander content indent */
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    border-left: 2px solid rgba(200,16,46,0.12) !important;
    margin-left: 6px !important;
    padding: 2px 0 4px 4px !important;
    border-top: none !important;
}

/* ── Sidebar nav buttons ─────────────────────────────────────────────────────
   Use 3 attribute selectors → specificity 0,3,0 which beats
   global .stButton > button (0,2,0) so overrides work reliably.          */

/* Remove wrapper margin */
[data-testid="stSidebar"] [data-testid="stExpander"] .stButton {
    margin: 1px 0 !important;
    padding: 0 !important;
}

/* ALL nav buttons — inactive (secondary) */
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stBaseButton-secondary"] {
    background: transparent !important;
    border: none !important;
    border-left: 2px solid transparent !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    color: #4f6280 !important;
    font-family: 'Inter','Segoe UI',sans-serif !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    letter-spacing: 0.1px !important;
    padding: 7px 10px 7px 14px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    width: 100% !important;
    transition: background 0.15s, border-color 0.15s, color 0.15s !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stBaseButton-secondary"]:hover {
    background: rgba(200,16,46,0.07) !important;
    border-left-color: rgba(200,16,46,0.35) !important;
    color: #9fb0c8 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Active nav button (primary) */
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stBaseButton-primary"] {
    background: rgba(200,16,46,0.10) !important;
    border: none !important;
    border-left: 2px solid #C8102E !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    color: #ffffff !important;
    font-family: 'Inter','Segoe UI',sans-serif !important;
    font-size: 12.5px !important;
    font-weight: 600 !important;
    letter-spacing: 0.1px !important;
    padding: 7px 10px 7px 14px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    width: 100% !important;
    transition: background 0.15s !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stBaseButton-primary"]:hover {
    background: rgba(200,16,46,0.16) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ═══ TYPOGRAPHY ══════════════════════════════════════════════════════════════ */
h1 {
    font-family: 'Barlow Condensed','Arial Black',sans-serif !important;
    font-weight: 900 !important;
    font-size: 2rem !important;
    color: #ffffff !important;
    letter-spacing: -0.5px !important;
    line-height: 1.15 !important;
}
h2 {
    font-family: 'Barlow Condensed','Arial Black',sans-serif !important;
    font-weight: 800 !important;
    font-size: 1.35rem !important;
    color: #dde2f0 !important;
    letter-spacing: 0.2px !important;
    text-transform: uppercase !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid rgba(200,16,46,0.18) !important;
    margin-bottom: 16px !important;
}
h3 {
    font-family: 'Inter','Segoe UI',sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: #b0bccc !important;
    letter-spacing: 0.1px !important;
}
h4 {
    font-family: 'Inter','Segoe UI',sans-serif !important;
    font-weight: 600 !important;
    color: #8b9ab5 !important;
    font-size: 0.9rem !important;
}
.stMarkdown p { color: #8b9ab5 !important; line-height: 1.65; }

/* ═══ METRICS ════════════════════════════════════════════════════════════════ */
[data-testid="stMetric"] {
    background: linear-gradient(135deg,
        rgba(29,66,138,0.16) 0%,
        rgba(200,16,46,0.08) 100%) !important;
    border: 1px solid rgba(200,16,46,0.2) !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(200,16,46,0.14) !important;
    border-color: rgba(200,16,46,0.35) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Barlow Condensed','Arial Black',sans-serif !important;
    font-size: 2rem !important;
    font-weight: 900 !important;
    color: #ffffff !important;
    line-height: 1.1 !important;
}
[data-testid="stMetricLabel"] {
    color: #4a566e !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.1px !important;
    font-weight: 700 !important;
}

/* ═══ TABS ════════════════════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.025) !important;
    border-radius: 12px !important;
    padding: 5px !important;
    gap: 3px !important;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #4a566e !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 0.3px !important;
    padding: 8px 18px !important;
    transition: all 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255,255,255,0.05) !important;
    color: #c8d0e0 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#C8102E,#1D428A) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 16px rgba(200,16,46,0.35) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 16px !important; }

/* ═══ BUTTONS ════════════════════════════════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg,#C8102E,#9B0822) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 800 !important;
    font-size: 14px !important;
    letter-spacing: 0.5px !important;
    padding: 10px 26px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 16px rgba(200,16,46,0.28) !important;
    font-family: 'Inter','Segoe UI',sans-serif !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(200,16,46,0.40) !important;
}
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.06) !important;
    box-shadow: none !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}
.stDownloadButton > button {
    background: rgba(29,66,138,0.2) !important;
    border: 1px solid rgba(29,66,138,0.4) !important;
    box-shadow: none !important;
}

/* ═══ INPUTS ═════════════════════════════════════════════════════════════════ */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: #e0e4f0 !important;
    font-size: 13px !important;
}
.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
    border-color: #C8102E !important;
    box-shadow: 0 0 0 2px rgba(200,16,46,0.18) !important;
}
[data-baseweb="popover"] > div,
[data-baseweb="menu"] {
    background: #0f1520 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
}
[data-baseweb="menu"] li { color: #c0c8dc !important; }
[data-baseweb="menu"] li:hover { background: rgba(200,16,46,0.14) !important; }
.stRadio [data-baseweb="radio"] span:first-child {
    border-color: rgba(200,16,46,0.45) !important;
}
.stRadio [aria-checked="true"] span:first-child {
    background: #C8102E !important;
    border-color: #C8102E !important;
}
/* Slider thumb */
.stSlider [role="slider"] {
    background: #C8102E !important;
    border: 2px solid #fff !important;
    box-shadow: 0 2px 10px rgba(200,16,46,0.5) !important;
}
/* Slider filled track */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="progressbar"],
[data-testid="stSlider"] [class*="Track"] > div:first-child {
    background: linear-gradient(90deg,#C8102E,#9B0822) !important;
}
/* Slider empty track */
[data-testid="stSlider"] [data-baseweb="slider"] [class*="Track"] {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 4px !important;
}

/* ═══ MULTISELECT TAGS ═══════════════════════════════════════════════════════ */
[data-baseweb="tag"] {
    background: rgba(200,16,46,0.18) !important;
    border: 1px solid rgba(200,16,46,0.3) !important;
    border-radius: 6px !important;
    height: 24px !important;
}
[data-baseweb="tag"] span { color: #f0d0d5 !important; font-size: 12px !important; }
[data-baseweb="tag"] button { color: rgba(240,208,213,0.7) !important; }
[data-baseweb="tag"] button:hover { color: #fff !important; }

/* ═══ DATA TABLES ════════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"], [data-testid="stDataFrameResizable"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
}

/* ═══ PLOTLY CHART CONTAINER ═════════════════════════════════════════════════ */
[data-testid="stPlotlyChart"] {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    box-shadow: 0 4px 28px rgba(0,0,0,0.45) !important;
    background: #0c1220 !important;
    transition: box-shadow 0.2s !important;
}
[data-testid="stPlotlyChart"]:hover {
    box-shadow: 0 8px 36px rgba(0,0,0,0.55),
                0 0 0 1px rgba(200,16,46,0.12) !important;
}

/* ═══ IMAGES ═════════════════════════════════════════════════════════════════ */
[data-testid="stImage"] img {
    border-radius: 12px !important;
    border: 2px solid rgba(200,16,46,0.3) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.6) !important;
}

/* ═══ MAIN CONTENT EXPANDERS ═════════════════════════════════════════════════ */
.main [data-testid="stExpander"] {
    background: rgba(255,255,255,0.018) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    margin: 6px 0 !important;
}
.main [data-testid="stExpander"] summary {
    padding: 12px 16px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    color: #6b7d96 !important;
    font-family: 'Inter','Segoe UI',sans-serif !important;
}
.main [data-testid="stExpander"] summary:hover {
    background: rgba(200,16,46,0.04) !important;
    color: #a0b0c8 !important;
}
.main details[open] > summary {
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
    color: #c0cce0 !important;
}

/* ═══ ALERTS ══════════════════════════════════════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
    backdrop-filter: blur(4px) !important;
}
[data-testid="stAlert"][data-baseweb="notification"] {
    background: rgba(52,152,219,0.09) !important;
    border-left-color: #3498db !important;
}
/* info */
div[data-testid="stAlert"] > div[role="alert"] {
    color: #8fb8d8 !important;
    font-size: 13px !important;
}
.stAlert p { color: inherit !important; }

/* ═══ DIVIDER / HR ═══════════════════════════════════════════════════════════ */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg,rgba(200,16,46,0.4),rgba(29,66,138,0.3),transparent) !important;
    margin: 20px 0 !important;
}

/* ═══ CAPTIONS ═══════════════════════════════════════════════════════════════ */
.stCaption, [data-testid="stCaptionContainer"] p,
small { color: #3d4f65 !important; font-size: 11.5px !important; line-height: 1.5 !important; }

/* ═══ SCROLLBAR ══════════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #080c14; }
::-webkit-scrollbar-thumb { background: rgba(200,16,46,0.32); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #C8102E; }

/* ═══ SPINNER ════════════════════════════════════════════════════════════════ */
.stSpinner > div { border-top-color: #C8102E !important; }

/* ═══ COLUMNS ════════════════════════════════════════════════════════════════ */
[data-testid="column"] { padding: 0 5px !important; }

/* ═══ STAT BAR (custom) ══════════════════════════════════════════════════════ */
.stat-row { display:flex; align-items:center; margin:5px 0; font-size:13px; }
.stat-label { width:170px; font-weight:500; color:#4a566e; }
.stat-value { width:55px; text-align:right; font-weight:900; color:#ffffff;
              margin-right:12px; font-size:14px;
              font-family:'Barlow Condensed','Arial Black',sans-serif; }
.bar-bg { background:rgba(255,255,255,0.06); border-radius:5px; width:180px;
          height:7px; display:inline-block; vertical-align:middle; }
.bar-fill { height:7px; border-radius:5px; display:block; }
.pct-label { margin-left:10px; font-size:11px; font-weight:700; }

/* ═══ CARDS (custom) ═════════════════════════════════════════════════════════ */
.player-card {
    background:linear-gradient(135deg,#091523,#130810);
    border:1px solid rgba(200,16,46,0.28);
    border-radius:20px; padding:22px; color:white; text-align:center;
    box-shadow:0 8px 32px rgba(0,0,0,0.55),inset 0 1px 0 rgba(255,255,255,0.04);
}
.cluster-badge {
    display:inline-block;
    background:linear-gradient(135deg,rgba(200,16,46,0.22),rgba(29,66,138,0.22));
    border:1px solid rgba(200,16,46,0.28);
    border-radius:20px; padding:5px 16px; font-size:12px; margin-top:8px;
    letter-spacing:0.5px; font-weight:600;
}
.similar-card {
    background:rgba(255,255,255,0.025);
    border-radius:12px; padding:12px 16px; margin:5px 0;
    border-left:4px solid #1d428a;
    transition:all 0.15s ease;
}
.similar-card:hover {
    background:rgba(29,66,138,0.1);
    border-left-color:#c8102e;
}
.gem-card {
    background:linear-gradient(135deg,#0f1f40,#1a0f30);
    border:1px solid rgba(255,199,44,0.18);
    border-radius:14px; padding:16px; color:white; margin:8px 0;
    box-shadow:0 4px 18px rgba(0,0,0,0.45);
}
.tv-badge {
    display:inline-block; border-radius:20px; padding:5px 16px;
    font-weight:800; font-size:13px; letter-spacing:0.5px;
}

/* ═══ FADE-IN ANIMATION ══════════════════════════════════════════════════════ */
@keyframes fadeInUp {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0); }
}
.main .block-container > div > div {
    animation: fadeInUp 0.25s ease forwards;
}

/* ═══ METRIC DELTA ═══════════════════════════════════════════════════════════ */
[data-testid="stMetricDelta"] {
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    font-family: 'Inter','Segoe UI',sans-serif !important;
    margin-top: 2px !important;
}
[data-testid="stMetricDelta"] svg { display: none !important; }

/* ═══ DARK ALERTS ════════════════════════════════════════════════════════════ */
.stAlert[data-type="info"],
div[data-testid="stAlert"]:has([aria-label*="info"]),
.element-container .stInfo {
    background: rgba(52,152,219,0.08) !important;
    border-left: 4px solid #3498DB !important;
    border-radius: 10px !important;
    color: #7db8d8 !important;
}
.stAlert[data-type="success"],
.element-container .stSuccess {
    background: rgba(39,174,96,0.08) !important;
    border-left: 4px solid #27AE60 !important;
    border-radius: 10px !important;
}
.stAlert[data-type="warning"],
.element-container .stWarning {
    background: rgba(243,156,18,0.08) !important;
    border-left: 4px solid #F39C12 !important;
    border-radius: 10px !important;
}
.stAlert[data-type="error"],
.element-container .stError {
    background: rgba(200,16,46,0.08) !important;
    border-left: 4px solid #C8102E !important;
    border-radius: 10px !important;
}

/* ═══ NUMBER INPUT ════════════════════════════════════════════════════════════ */
[data-testid="stNumberInput"] input,
.stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: #e0e4f0 !important;
    font-size: 13px !important;
    font-family: 'Inter','Segoe UI',sans-serif !important;
}
[data-testid="stNumberInput"] input:focus,
.stTextInput input:focus {
    border-color: #C8102E !important;
    box-shadow: 0 0 0 2px rgba(200,16,46,0.15) !important;
}

/* ═══ CHECKBOX ═══════════════════════════════════════════════════════════════ */
[data-testid="stCheckbox"] label {
    font-size: 13px !important;
    color: #6b7d96 !important;
    font-family: 'Inter','Segoe UI',sans-serif !important;
}
[data-testid="stCheckbox"] [data-baseweb="checkbox"] div:first-child {
    border-color: rgba(200,16,46,0.4) !important;
    border-radius: 4px !important;
}
[data-testid="stCheckbox"] [aria-checked="true"] div:first-child {
    background: #C8102E !important;
    border-color: #C8102E !important;
}

/* ═══ SIDEBAR LABELS ═════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label {
    font-size: 10px !important;
    font-weight: 800 !important;
    color: #2e3f55 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
}

/* ═══ DATAFRAME DARK HEADER ══════════════════════════════════════════════════ */
[data-testid="stDataFrame"] th,
[data-testid="stDataFrameResizable"] th {
    background: rgba(29,66,138,0.25) !important;
    color: #8b9ab5 !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    border-bottom: 1px solid rgba(200,16,46,0.15) !important;
}
[data-testid="stDataFrame"] td,
[data-testid="stDataFrameResizable"] td {
    background: rgba(255,255,255,0.015) !important;
    color: #c0cce0 !important;
    font-size: 13px !important;
    border-bottom: 1px solid rgba(255,255,255,0.04) !important;
}
[data-testid="stDataFrame"] tr:hover td,
[data-testid="stDataFrameResizable"] tr:hover td {
    background: rgba(200,16,46,0.06) !important;
}

/* ═══ TAB PANEL SPACING ══════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 20px !important;
    padding-bottom: 8px !important;
}

/* ═══ SECTION TITLE SPACING ══════════════════════════════════════════════════ */
.main [data-testid="stMarkdownContainer"] h2 {
    margin-top: 24px !important;
}

/* ═══ SIMILAR CARD DARK ══════════════════════════════════════════════════════ */
.similar-card {
    background: rgba(255,255,255,0.025) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin: 5px 0 !important;
    border-left: 4px solid rgba(29,66,138,0.5) !important;
    color: #b0bccc !important;
}
.similar-card:hover {
    background: rgba(29,66,138,0.08) !important;
    border-left-color: #C8102E !important;
}

/* ═══ PROGRESS / STATUS DOTS ═════════════════════════════════════════════════ */
.stProgress > div > div > div {
    background: linear-gradient(90deg,#C8102E,#1D428A) !important;
    border-radius: 4px !important;
}
.stProgress > div > div {
    background: rgba(255,255,255,0.07) !important;
    border-radius: 4px !important;
}

/* ── Multiselect tags ──────────────────────────────────────────────────────── */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background-color: #1D428A !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}
[data-testid="stMultiSelect"] span[data-baseweb="tag"] span {
    color: #ffffff !important;
}
[data-testid="stMultiSelect"] span[data-baseweb="tag"] [role="presentation"] svg {
    fill: rgba(255,255,255,0.7) !important;
}
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    background-color: #0d1825 !important;
    border-color: rgba(200,16,46,0.4) !important;
}
[data-testid="stMultiSelect"] label {
    color: #8b9ab5 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}
</style>
"""
