"""
Lightweight NBA dark theme registration — no Streamlit dependency.
Used by src/*.py modules so charts work standalone and inside the app.
"""
import plotly.graph_objects as go
import plotly.io as pio

BG_CHART   = "#0c1220"
BG_CARD    = "#0f1520"
BG_HOVER   = "#162035"
TEXT_MAIN  = "#E8EAF6"
TEXT_SUB   = "#8B9AB5"
TEXT_MUTED = "#4A566E"
GRID_LINE  = "#141E30"
NBA_RED    = "#C8102E"
NBA_BLUE   = "#1D428A"
BORDER_DIM = "rgba(255,255,255,0.06)"
COLOR_SEQ  = [
    "#C8102E","#1D428A","#FFC72C","#00D4AA",
    "#9B59B6","#E67E22","#2ECC71","#3498DB",
    "#E74C3C","#F0A500",
]


def _register():
    tpl = go.layout.Template()
    tpl.layout.update(
        plot_bgcolor  = BG_CHART,
        paper_bgcolor = BG_CARD,
        colorway      = COLOR_SEQ,
        font=dict(family="'Inter','Segoe UI',Arial,sans-serif",
                  color=TEXT_MAIN, size=12),
        title=dict(font=dict(size=16, color=TEXT_MAIN,
                             family="'Inter','Segoe UI',Arial,sans-serif"), x=0.02),
        xaxis=dict(gridcolor=GRID_LINE, gridwidth=1, linecolor=GRID_LINE,
                   zerolinecolor=GRID_LINE,
                   tickfont=dict(color=TEXT_MUTED, size=11),
                   title_font=dict(color=TEXT_SUB, size=12),
                   showspikes=True, spikecolor=NBA_RED,
                   spikethickness=1, spikedash="dot"),
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
                   radialaxis=dict(gridcolor=GRID_LINE,
                                   tickfont=dict(color=TEXT_MUTED)),
                   angularaxis=dict(gridcolor=GRID_LINE,
                                    tickfont=dict(color=TEXT_MUTED))),
        margin=dict(t=55, b=40, l=50, r=20),
        modebar=dict(bgcolor="rgba(0,0,0,0)",
                     color=TEXT_MUTED, activecolor=NBA_RED),
        coloraxis=dict(
            colorbar=dict(
                bgcolor="rgba(0,0,0,0)", bordercolor=BORDER_DIM,
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
    pio.templates.default    = "nba_dark"


def ensure_template() -> str:
    """Register nba_dark if needed and return its name."""
    if "nba_dark" not in pio.templates:
        _register()
    return "nba_dark"
