"""
Anomaly Detection — поиск недооценённых и переоценённых игроков.
IsolationForest находит игроков, чьи статы выбиваются из своего кластера.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "PTS", "AST", "REB", "STL", "BLK",
    "FG_PCT", "FG3_PCT", "TOV", "PLUS_MINUS",
    "USG_PCT", "TS_PCT", "efficiency",
]


def detect_anomalies(df, contamination=0.08):
    """
    Найти игроков-аномалий внутри каждого кластера.
    Возвращает DataFrame с колонками:
      anomaly_score  — чем выше, тем более аномален (0..100)
      anomaly_label  — 'underrated' / 'overrated' / 'normal'
    """
    df = df.copy()
    available = [c for c in FEATURE_COLS if c in df.columns]
    name_col  = "name" if "name" in df.columns else "PLAYER_NAME"

    df["anomaly_score"] = 0.0
    df["anomaly_label"] = "normal"

    for cluster in df["cluster_name"].dropna().unique():
        mask = df["cluster_name"] == cluster
        sub  = df[mask][available].fillna(0)
        if len(sub) < 6:
            continue

        scaler = StandardScaler()
        X = scaler.fit_transform(sub)

        iso = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
        )
        iso.fit(X)

        # score_samples: чем ниже (отрицательнее) — тем больше аномалия
        raw_scores = iso.score_samples(X)
        # Нормируем в 0..100 (100 = сильная аномалия)
        mn, mx = raw_scores.min(), raw_scores.max()
        norm = 100 * (1 - (raw_scores - mn) / (mx - mn + 1e-9))

        df.loc[mask, "anomaly_score"] = norm

        # Определяем тип аномалии: underrated или overrated
        predictions = iso.predict(X)   # -1 = аномалия, 1 = норма
        idx_list = df[mask].index.tolist()

        for i, (pred, idx) in enumerate(zip(predictions, idx_list)):
            if pred == -1:
                # Сравниваем суммарную результативность с медианой кластера
                eff = df.at[idx, "efficiency"] if "efficiency" in df.columns else df.at[idx, "PTS"]
                median_eff = df[mask]["efficiency"].median() if "efficiency" in df.columns else df[mask]["PTS"].median()
                df.at[idx, "anomaly_label"] = "underrated" if eff > median_eff else "overrated"

    return df


def get_hidden_gems(df, top_n=15):
    """Топ-N недооценённых игроков (высокая эффективность, низкий USG%)."""
    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    df = detect_anomalies(df)

    gems = df[df["anomaly_label"] == "underrated"].copy()
    if gems.empty:
        return gems

    # Сортируем по anomaly_score + efficiency
    gems["gem_score"] = gems["anomaly_score"] * 0.5 + gems.get("efficiency", gems.get("PTS", 0)) * 2
    cols = [name_col, "team", "cluster_name", "PTS", "AST", "REB",
            "USG_PCT", "TS_PCT", "efficiency", "anomaly_score"]
    cols = [c for c in cols if c in gems.columns]
    return gems.nlargest(top_n, "gem_score")[cols]


def get_overrated(df, top_n=10):
    """Топ переоценённых (высокий USG%, низкая эффективность)."""
    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    df = detect_anomalies(df)

    over = df[df["anomaly_label"] == "overrated"].copy()
    if over.empty:
        return over

    cols = [name_col, "team", "cluster_name", "PTS", "AST", "REB",
            "USG_PCT", "TS_PCT", "efficiency", "anomaly_score"]
    cols = [c for c in cols if c in over.columns]
    return over.nlargest(top_n, "anomaly_score")[cols]


if __name__ == "__main__":
    df = pd.read_csv("data/processed/players_clustered.csv")
    df = detect_anomalies(df)
    print("Распределение меток:")
    print(df["anomaly_label"].value_counts())
    print("\nТоп недооценённых:")
    print(get_hidden_gems(df, 10).to_string(index=False))
