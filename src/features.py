"""
Feature engineering для NBA.
Данные уже в формате per-game (PerGame из nba_api).
"""
import pandas as pd
import numpy as np

FEATURE_COLS = [
    "PTS", "AST", "REB", "STL", "BLK",
    "FG_PCT", "FG3_PCT", "FT_PCT",
    "TOV", "PLUS_MINUS",
    "USG_PCT", "TS_PCT",
]


def clean_players(df):
    """Фильтрация: минимум 20 игр, убрать незначимые строки."""
    df = df.copy()

    # Стандартизируем имя
    if "PLAYER_NAME" in df.columns and "name" not in df.columns:
        df["name"] = df["PLAYER_NAME"]
    if "TEAM_ABBREVIATION" in df.columns and "team" not in df.columns:
        df["team"] = df["TEAM_ABBREVIATION"]

    # Минимум игр
    gp_col = "GP" if "GP" in df.columns else "GAME_COUNT"
    if gp_col in df.columns:
        df[gp_col] = pd.to_numeric(df[gp_col], errors="coerce")
        df = df[df[gp_col] >= 20].copy()
        df = df.rename(columns={gp_col: "games_played"})

    # Убрать строки без ключевых данных
    key = [c for c in ["PTS", "AST", "REB"] if c in df.columns]
    if key:
        df = df.dropna(subset=key)

    df = df.reset_index(drop=True)
    print(f"После очистки: {len(df)} игроков")
    return df


def add_features(df):
    """Добавить составные признаки."""
    df = df.copy()

    # Числовой тип всех признаков
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Комплексный индекс эффективности
    pts = df.get("PTS", 0)
    ast = df.get("AST", 0)
    reb = df.get("REB", 0)
    stl = df.get("STL", 0)
    blk = df.get("BLK", 0)
    tov = df.get("TOV", 0)
    df["efficiency"] = pts + ast + reb + stl + blk - tov

    # Скоринг + плеймейкинг
    df["scoring_creation"] = pts + 0.5 * ast
    df["defensive_impact"] = stl + blk

    # Заполнить USG_PCT / TS_PCT если отсутствуют
    if "USG_PCT" not in df.columns:
        df["USG_PCT"] = 0.0
    if "TS_PCT" not in df.columns:
        df["TS_PCT"] = df.get("FG_PCT", 0)

    return df


FINAL_FEATURES = [
    "PTS", "AST", "REB", "STL", "BLK",
    "FG_PCT", "FG3_PCT", "TOV", "PLUS_MINUS",
    "USG_PCT", "TS_PCT", "efficiency",
]

if __name__ == "__main__":
    df = pd.read_csv("data/raw/players_raw.csv")
    df = clean_players(df)
    df = add_features(df)
    df.to_csv("data/processed/players_features.csv", index=False)
    print("Сохранено -> data/processed/players_features.csv")
    available = [c for c in FINAL_FEATURES if c in df.columns]
    print(df[available].describe().round(3))
