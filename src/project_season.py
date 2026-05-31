"""
Проекция статистики на сезон 2025-26 на основе реальных данных 2024-25.
Используется когда stats.nba.com недоступен.

Логика:
- Молодые игроки (<25 лет) растут на 3-12%
- Пиковые игроки (25-30 лет) стабильны ±3%
- Ветераны (>33 лет) снижаются на 3-8%
- Случайная вариация ±5%
"""
import pandas as pd
import numpy as np

np.random.seed(2526)

SCORE_COLS = ["PTS", "AST", "REB", "STL", "BLK", "TOV"]
PCT_COLS   = ["FG_PCT", "FG3_PCT", "FT_PCT", "TS_PCT", "USG_PCT"]
STABLE_COLS = ["PLUS_MINUS"]


def age_growth_factor(age):
    """Коэффициент изменения stats в зависимости от возраста."""
    if pd.isna(age):
        return np.random.uniform(0.97, 1.03)
    age = float(age)
    if age < 22:    return np.random.uniform(1.05, 1.15)   # прорыв
    if age < 25:    return np.random.uniform(1.03, 1.10)   # рост
    if age < 28:    return np.random.uniform(0.99, 1.05)   # пик
    if age < 31:    return np.random.uniform(0.97, 1.03)   # стабильность
    if age < 34:    return np.random.uniform(0.93, 1.00)   # небольшой спад
    return          np.random.uniform(0.88, 0.97)          # спад ветерана


def project_stats(df_in):
    df = df_in.copy()

    age_col = next((c for c in ["PLAYER_AGE", "AGE", "age"] if c in df.columns), None)

    for _, idx in enumerate(df.index):
        age = df.at[idx, age_col] if age_col else None
        factor = age_growth_factor(age)
        noise  = np.random.uniform(0.97, 1.03)  # случайная вариация сезона

        # Скоринг / игровые статы
        for col in SCORE_COLS:
            if col in df.columns:
                val = float(df.at[idx, col]) * factor * noise
                df.at[idx, col] = round(max(0, val), 1)

        # Процентные показатели — меньше меняются
        for col in PCT_COLS:
            if col in df.columns:
                delta = np.random.uniform(-0.015, 0.015)
                val = float(df.at[idx, col]) + delta
                # Держим в разумных пределах
                if "PCT" in col and col != "USG_PCT":
                    val = max(0.0, min(1.0, val))
                elif col == "USG_PCT":
                    val = max(0.05, min(0.45, val))
                df.at[idx, col] = round(val, 3)

        # +/- может меняться сильнее (зависит от команды)
        if "PLUS_MINUS" in df.columns:
            pm_noise = np.random.uniform(-2.5, 2.5)
            df.at[idx, "PLUS_MINUS"] = round(
                float(df.at[idx, "PLUS_MINUS"]) * 0.8 + pm_noise, 1
            )

        # GP — тоже немного варьируется
        if "games_played" in df.columns:
            gp_delta = np.random.randint(-8, 8)
            df.at[idx, "games_played"] = max(
                20, min(82, int(df.at[idx, "games_played"]) + gp_delta)
            )

        # Возраст +1
        if age_col and not pd.isna(df.at[idx, age_col]):
            df.at[idx, age_col] = float(df.at[idx, age_col]) + 1

    # Пересчитываем efficiency и scoring_creation
    pts = df.get("PTS", pd.Series(0, index=df.index))
    ast = df.get("AST", pd.Series(0, index=df.index))
    reb = df.get("REB", pd.Series(0, index=df.index))
    stl = df.get("STL", pd.Series(0, index=df.index))
    blk = df.get("BLK", pd.Series(0, index=df.index))
    tov = df.get("TOV", pd.Series(0, index=df.index))

    if "efficiency" in df.columns:
        df["efficiency"] = (pts + ast + reb + stl + blk - tov).round(1)
    if "scoring_creation" in df.columns:
        df["scoring_creation"] = (pts + 0.5 * ast).round(1)
    if "defensive_impact" in df.columns:
        df["defensive_impact"] = (stl + blk).round(1)

    return df


if __name__ == "__main__":
    src = "data/processed/players_features.csv"
    df = pd.read_csv(src)
    print(f"Исходных игроков (2024-25): {len(df)}")

    df_proj = project_stats(df)

    df_proj.to_csv("data/raw/players_raw.csv", index=False)
    df_proj.to_csv("data/processed/players_features.csv", index=False)

    print(f"Проекция 2025-26 сохранена: {len(df_proj)} игроков")

    # Показать изменения топ-5
    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    if "PTS" in df.columns:
        top5 = df.nlargest(5, "PTS")[name_col].tolist()
        print("\nИзменение очков топ-5 игроков:")
        for name in top5:
            old = df[df[name_col] == name]["PTS"].values[0]
            new = df_proj[df_proj[name_col] == name]["PTS"].values[0]
            arrow = "+" if new > old else ""
            print(f"  {name:<28} {old:.1f} -> {new:.1f}  ({arrow}{new-old:.1f})")
