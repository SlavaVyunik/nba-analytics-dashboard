"""
AI Analyst — prompt builders for Gemini-powered scouting & trade analysis.

No LLM calls here — pure prompt construction + result formatting so the
module stays testable without an API key.
"""

from __future__ import annotations

import textwrap
from typing import Any

import numpy as np
import pandas as pd

# ── helpers ──────────────────────────────────────────────────────────────────

_NAME_COLS = ("PLAYER_NAME", "player_name", "name", "Name")
_TEAM_COLS = ("TEAM_ABBREVIATION", "team", "Team")


def _nc(df: pd.DataFrame) -> str:
    return next((c for c in _NAME_COLS if c in df.columns), df.columns[0])


def _pct(val: Any, factor: float = 100.0) -> str:
    try:
        return f"{float(val) * factor:.1f}%"
    except Exception:
        return "—"


def _f1(val: Any) -> str:
    try:
        return f"{float(val):.1f}"
    except Exception:
        return "—"


def _fi(val: Any) -> str:
    try:
        return str(int(round(float(val))))
    except Exception:
        return "—"


def _signed(val: Any) -> str:
    try:
        v = float(val)
        return f"+{v:.1f}" if v >= 0 else f"{v:.1f}"
    except Exception:
        return "—"


def _tier_label(tier: str) -> str:
    mapping = {
        "💎 Франчайз": "Franchise (top 5 league)",
        "⭐ Топ-15":    "Elite (top 15 league)",
        "🔵 Стартер":  "Starter",
        "🟡 Ролевой":  "Role player",
        "⚪ Резерв":   "Reserve / G-league bubble",
    }
    return mapping.get(tier, tier)


def _cluster_label(cluster: str) -> str:
    mapping = {
        "Плеймейкер":      "Playmaker / point-of-attack creator",
        "Снайпер":         "3-and-D / spot-up shooter",
        "Форвард-универсал":"Versatile forward / Swiss-army wing",
        "Большой":         "Big man / frontcourt anchor",
        "Ролевой игрок":   "Role player / complementary piece",
        "Защитный игрок":  "Defensive specialist",
        "Атакующий":       "High-usage scorer",
    }
    return mapping.get(cluster, cluster)


def _row_stats_block(row: pd.Series, label: str = "") -> str:
    header = f"Stats ({label}):" if label else "Stats (per game):"
    pct_factor = 100 if row.get("FG_PCT", 0) <= 1 else 1
    return textwrap.dedent(f"""
        {header}
        • Points: {_f1(row.get('PTS'))}  Assists: {_f1(row.get('AST'))}  Rebounds: {_f1(row.get('REB'))}
        • Steals: {_f1(row.get('STL'))}  Blocks: {_f1(row.get('BLK'))}  Turnovers: {_f1(row.get('TOV'))}
        • FG%: {_pct(row.get('FG_PCT'))}  3P%: {_pct(row.get('FG3_PCT'))}  FT%: {_pct(row.get('FT_PCT'))}
        • USG%: {_pct(row.get('USG_PCT'))}  TS%: {_pct(row.get('TS_PCT'))}  +/-: {_signed(row.get('PLUS_MINUS'))}
        • Minutes: {_f1(row.get('MIN'))}  Games: {_fi(row.get('games_played'))}
        • Trade Value: {_f1(row.get('trade_value'))}/100  Tier: {_tier_label(row.get('tv_tier',''))}
        • Injury Risk: {_f1(row.get('injury_risk'))}%  Efficiency: {_f1(row.get('efficiency'))}
    """).strip()


# ── Scout prompt builders ─────────────────────────────────────────────────────

REPORT_TYPES = {
    "🔍 Полный скаутинг":     "full_scout",
    "💰 Контрактная оценка":  "contract",
    "📈 Прогноз карьеры":     "career",
    "⚔️ Сравнение игроков":   "compare",
}


def build_scout_prompt(
    row: pd.Series,
    cluster_avg: pd.Series,
    report_type: str = "full_scout",
    compare_row: pd.Series | None = None,
    extra_context: str = "",
) -> str:
    """
    Build a Gemini prompt for a player scouting report.

    Parameters
    ----------
    row          : full enriched player row (with trade_value, injury_risk, etc.)
    cluster_avg  : per-stat mean for the player's cluster
    report_type  : one of REPORT_TYPES values
    compare_row  : second player row (only used when report_type == "compare")
    extra_context: free-text appended before the instructions
    """
    player  = str(row.get("PLAYER_NAME") or row.get("name") or "Unknown")
    team    = str(row.get("TEAM_ABBREVIATION") or row.get("team") or "—")
    age     = _fi(row.get("AGE"))
    cluster = str(row.get("cluster_name", "—"))

    stats_block = _row_stats_block(row)
    cluster_block = (
        f"Cluster average for '{cluster}' ({_cluster_label(cluster)}):\n"
        f"• PTS: {_f1(cluster_avg.get('PTS'))}  AST: {_f1(cluster_avg.get('AST'))}"
        f"  REB: {_f1(cluster_avg.get('REB'))}"
        f"  USG%: {_pct(cluster_avg.get('USG_PCT'))}"
        f"  TS%: {_pct(cluster_avg.get('TS_PCT'))}"
    )

    base = (
        "You are a senior NBA analyst and scout. "
        "Write your analysis IN RUSSIAN, using NBA terminology naturally. "
        "Be direct, specific and analytical — no filler phrases, no greeting.\n\n"
        f"Player: {player} | Team: {team} | Age: {age} | "
        f"Archetype: {cluster} ({_cluster_label(cluster)})\n\n"
        f"{stats_block}\n\n"
        f"{cluster_block}\n"
    )
    if extra_context:
        base += f"\nAdditional context: {extra_context}\n"

    if report_type == "full_scout":
        instructions = textwrap.dedent("""
            Write a professional scouting report with these FOUR sections in bold markdown:

            **1. Сильные стороны**
            What makes this player valuable. Be specific: name actual skills, tendencies, situations.

            **2. Зоны роста**
            Concrete weaknesses (mention specific stats that are below cluster average).
            What must improve and how.

            **3. Роль в команде**
            Ideal role (starter / sixth-man / specialist). What type of system fits.
            Which partner archetypes complement him best.

            **4. Трансферная оценка**
            Fair trade value, salary tier (max / mid / vet min), contract risk.
            Would you buy or sell right now? Give a one-sentence verdict.

            Total length: 300-420 words.
        """).strip()

    elif report_type == "contract":
        instructions = textwrap.dedent("""
            Write a CONTRACT VALUATION report with these sections:

            **1. Рыночная стоимость**
            Estimated annual salary ($ millions) and why. Compare to similar players.

            **2. Риски контракта**
            Age curve, injury risk score, consistency flags (games played, variation).
            Would you give a long-term deal? Max years?

            **3. Командная ценность**
            Cost-efficiency: is this player over/under/fairly paid at market value?
            What does he bring beyond raw stats (intangibles, leadership, stretch the floor)?

            **4. Рекомендация GM**
            If you were the GM: extend, trade, let walk, or buy out?
            One-line verdict.

            Total length: 280-380 words.
        """).strip()

    elif report_type == "career":
        instructions = textwrap.dedent("""
            Write a CAREER TRAJECTORY forecast with these sections:

            **1. Текущая фаза карьеры**
            Based on age and stats: breakthrough / peak / stable / declining.
            How close to his statistical ceiling?

            **2. Прогноз на 2–3 сезона**
            Which stats will grow, which will plateau or decline and why.
            Injury risk factor.

            **3. Потолок и дно**
            Best-case scenario (everything goes right).
            Worst-case scenario (injuries, regression).

            **4. Историческое сравнение**
            Name 2-3 real NBA players with similar profiles at this age.
            What happened to their careers?

            **5. Инвестиционный рейтинг**
            Score 1–10 for: peak value, floor value, longevity.
            One-sentence summary.

            Total length: 300-400 words.
        """).strip()

    elif report_type == "compare" and compare_row is not None:
        p2      = str(compare_row.get("PLAYER_NAME") or compare_row.get("name") or "Player 2")
        team2   = str(compare_row.get("TEAM_ABBREVIATION") or compare_row.get("team") or "—")
        age2    = _fi(compare_row.get("AGE"))
        cluster2 = str(compare_row.get("cluster_name", "—"))
        stats2_block = _row_stats_block(compare_row, label=p2)
        base += (
            f"\n--- PLAYER 2 ---\n"
            f"Player: {p2} | Team: {team2} | Age: {age2} | "
            f"Archetype: {cluster2} ({_cluster_label(cluster2)})\n\n"
            f"{stats2_block}\n"
        )
        instructions = textwrap.dedent(f"""
            Write a head-to-head COMPARISON of {player} vs {p2} with these sections:

            **1. Статистическое сравнение**
            Side-by-side breakdown of key categories. Who leads and by how much?
            Highlight surprising gaps or unexpected advantages.

            **2. Стиль и роль**
            How do their archetypes differ or overlap?
            In which system does each thrive?

            **3. Ценность в обмене**
            If a team had to choose one — who and why?
            Trade value delta and justification.

            **4. Вердикт**
            One clear winner with a one-sentence explanation.
            Score: {player} X — Y {p2}.

            Total length: 280-380 words.
        """).strip()
    else:
        # fallback to full scout if compare without second player
        instructions = "Write a brief 200-word scouting overview."

    return base + "\n\n" + instructions


# ── Trade prompt builder ──────────────────────────────────────────────────────

def build_trade_prompt(
    team_a_name: str,
    players_out: list[pd.Series],
    team_b_name: str,
    players_in: list[pd.Series],
    tv_balance: float,
) -> str:
    """
    Build a Gemini prompt for a trade analysis.

    tv_balance = sum(players_in TV) - sum(players_out TV)
    Positive = Team A gains value.
    """

    def _player_summary(row: pd.Series) -> str:
        name    = str(row.get("PLAYER_NAME") or row.get("name") or "?")
        age     = _fi(row.get("AGE"))
        cluster = _cluster_label(str(row.get("cluster_name", "")))
        tv      = _f1(row.get("trade_value"))
        tier    = _tier_label(str(row.get("tv_tier", "")))
        pts     = _f1(row.get("PTS"))
        ast     = _f1(row.get("AST"))
        reb     = _f1(row.get("REB"))
        risk    = _f1(row.get("injury_risk"))
        return (f"  • {name} (age {age}, {cluster}) — "
                f"TV: {tv}/100 [{tier}], "
                f"{pts}pts/{ast}ast/{reb}reb, injury risk {risk}%")

    out_block = "\n".join(_player_summary(r) for r in players_out) or "  (none)"
    in_block  = "\n".join(_player_summary(r) for r in players_in) or "  (none)"

    tv_out = sum(float(r.get("trade_value", 0)) for r in players_out)
    tv_in  = sum(float(r.get("trade_value", 0)) for r in players_in)

    direction = (
        f"Team A gains +{abs(tv_balance):.1f} TV points"
        if tv_balance > 1 else
        f"Team B gains +{abs(tv_balance):.1f} TV points"
        if tv_balance < -1 else
        "Trade is roughly equal in TV"
    )

    prompt = textwrap.dedent(f"""
        You are a senior NBA trade analyst. Analyze this proposed trade IN RUSSIAN.
        Be direct, precise, opinionated. No filler phrases.

        ══ TRADE PROPOSAL ══
        Team A ({team_a_name}) SENDS:
{out_block}
          Total TV sent: {tv_out:.1f}

        Team B ({team_b_name}) RECEIVES:
{in_block}
          Total TV received: {tv_in:.1f}

        Trade Value balance: {direction} (delta = {tv_balance:+.1f})

        ══ YOUR ANALYSIS ══

        **1. Ценностный баланс**
        Who wins on raw trade value? Is the gap justified by age, role, or upside?
        Call out any overpay or undervaluation.

        **2. Командная логика**
        What does each team gain and lose strategically?
        Does this trade make both teams better — or just one?

        **3. Риски**
        Injury risks, age cliffs, fit issues, contract concerns.
        Who takes on the bigger gamble?

        **4. Альтернативы**
        Briefly: is there a better trade either team could make instead?

        **5. Вердикт**
        Winner: Team A / Team B / Neutral
        Confidence: 🔴 High / 🟡 Medium / 🟢 Low
        One-sentence summary of why.

        Total length: 320-450 words.
    """).strip()

    return prompt


# ── Result formatting ─────────────────────────────────────────────────────────

def extract_verdict(report_text: str) -> tuple[str, str]:
    """
    Parse the AI report text and extract a verdict emoji + label.
    Returns (emoji, label) e.g. ("✅", "Team A wins").
    Falls back to ("💬", "See report") if verdict not found.
    """
    text_lower = report_text.lower()

    # ── Explicit trade winner lines (highest priority) ────────────────────────
    if any(w in text_lower for w in (
        "winner: team a", "побеждает команда a", "выигрывает команда a",
        "команда a побеждает", "команда a выигрывает",
    )):
        return "✅", "Team A wins"
    if any(w in text_lower for w in (
        "winner: team b", "побеждает команда b", "выигрывает команда b",
        "команда b побеждает", "команда b выигрывает",
    )):
        return "🔄", "Team B wins"

    # ── Negative signals (check BEFORE positive to avoid "не стоит" ↔ "стоит") ─
    if any(w in text_lower for w in (
        "невыгодно", "переплата", "не стоит", "не рекомендую",
        "отказаться", "провальная", "плохая сделка",
    )):
        return "⚠️", "Осторожно"

    # ── Neutral ───────────────────────────────────────────────────────────────
    if any(w in text_lower for w in ("нейтрально", "neutral", "ничья", "поровну")):
        return "⚖️", "Нейтрально"

    # ── Positive ──────────────────────────────────────────────────────────────
    if any(w in text_lower for w in (
        "выгодно", "отличный", "рекомендую", "стоит принять",
        "хорошая сделка", "выгодная", "разумная сделка",
    )):
        return "✅", "Выгодно"

    return "💬", "Анализ готов"


def trade_balance_html(tv_out: float, tv_in: float) -> str:
    """Return an HTML string showing the TV balance bar."""
    total = max(tv_out + tv_in, 1)
    pct_out = tv_out / total * 100
    pct_in  = tv_in  / total * 100
    delta   = tv_in - tv_out
    sign    = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
    color   = "#27AE60" if delta >= 0 else "#C8102E"
    return f"""
<div style="margin:12px 0 8px; font-family:'Inter',sans-serif">
  <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px">
    <span style="color:#8b9ab5; font-size:12px">Trade Value баланс</span>
    <span style="background:{color}22; color:{color}; border:1px solid {color}55;
                 border-radius:20px; padding:2px 10px; font-size:13px; font-weight:700">
      {sign} TV
    </span>
  </div>
  <div style="display:flex; height:12px; border-radius:6px; overflow:hidden; background:#0f1520">
    <div style="width:{pct_out:.1f}%; background:#C8102E; transition:width .4s"></div>
    <div style="width:{pct_in:.1f}%;  background:#1D428A; transition:width .4s"></div>
  </div>
  <div style="display:flex; justify-content:space-between; margin-top:4px;
              font-size:11px; color:#6b7a99">
    <span>🔴 Отдаём: {tv_out:.1f}</span>
    <span>🔵 Получаем: {tv_in:.1f}</span>
  </div>
</div>""".strip()
