"""
Загрузка статистики игроков NBA за сезон 2025-26.
Использует nba_api - бесплатно, без ключей.
"""
import pandas as pd
import time
from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashplayerbiostats
from nba_api.stats.library.http import NBAStatsHTTP

SEASON = "2025-26"

# NBA требует browser-like заголовки иначе блокирует
NBAStatsHTTP.HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Host": "stats.nba.com",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


def _fetch_with_retry(fn, retries=3, delay=5):
    """Повторить запрос при таймауте."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt < retries - 1:
                print(f"  Попытка {attempt+1} не удалась ({type(e).__name__}), ждём {delay}с...")
                time.sleep(delay)
            else:
                raise


def get_player_stats(season=SEASON):
    print(f"Загружаем базовую статистику NBA {season}...")
    time.sleep(1)
    def _fetch():
        return leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Base",
            timeout=60,
        ).get_data_frames()[0]
    df = _fetch_with_retry(_fetch)
    print(f"  Базовая статистика: {len(df)} игроков")
    return df


def get_advanced_stats(season=SEASON):
    print("Загружаем продвинутую статистику...")
    time.sleep(1)
    def _fetch():
        return leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Advanced",
            timeout=60,
        ).get_data_frames()[0]
    df = _fetch_with_retry(_fetch)
    print(f"  Продвинутая статистика: {len(df)} игроков")
    return df


def get_bio_stats(season=SEASON):
    print("Загружаем биометрию...")
    time.sleep(1)
    def _fetch():
        return leaguedashplayerbiostats.LeagueDashPlayerBioStats(
            season=season, timeout=60,
        ).get_data_frames()[0]
    df = _fetch_with_retry(_fetch)
    print(f"  Биометрия: {len(df)} игроков")
    return df


if __name__ == "__main__":
    base = get_player_stats(SEASON)
    adv = get_advanced_stats(SEASON)
    bio = get_bio_stats(SEASON)

    # Объединяем по PLAYER_ID
    adv_cols = ["PLAYER_ID", "USG_PCT", "NET_RATING", "PIE", "TS_PCT",
                "AST_PCT", "REB_PCT", "PACE"]
    adv_cols = [c for c in adv_cols if c in adv.columns]

    bio_cols = ["PLAYER_ID", "PLAYER_AGE", "PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT"]
    bio_cols = [c for c in bio_cols if c in bio.columns]

    df = base.merge(adv[adv_cols], on="PLAYER_ID", how="left")
    df = df.merge(bio[bio_cols], on="PLAYER_ID", how="left")

    df.to_csv("data/raw/players_raw.csv", index=False)
    print(f"\nСохранено {len(df)} игроков -> data/raw/players_raw.csv")
    print("Колонки:", df.columns.tolist())
    print("\nПример:")
    print(df[["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "PTS", "AST", "REB"]].head(5).to_string())
