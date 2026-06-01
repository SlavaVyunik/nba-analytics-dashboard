# NBA Analytics Dashboard

Аналитическая платформа для анализа игроков NBA сезона 2025-26.

**Стек:** Streamlit · XGBoost · UMAP · K-Means · Monte Carlo · Google Gemini AI

---

## Быстрый старт

### 1. Клонировать репозиторий
```bash
git clone https://github.com/SlavaVyunik/nba-analytics-dashboard.git
cd nba-analytics-dashboard
```

### 2. Установить зависимости
```bash
pip install -r requirements.txt
```

### 3. Добавить API-ключ Google Gemini
Скопируйте пример и вставьте свой ключ:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Откройте `.streamlit/secrets.toml` и замените значение ключа.  
Получить ключ бесплатно: [aistudio.google.com](https://aistudio.google.com) → **Get API key**

### 4. Загрузить данные
```bash
python src/fetch_data.py
python src/features.py
python src/clustering.py
```

### 5. Запустить дашборд
```bash
streamlit run app/dashboard.py
```
Откройте браузер: [localhost:8501](http://localhost:8501)

---

## Модули

| Модуль | Описание |
|--------|----------|
| Обзор лиги | UMAP-карта, KPI, топ-5 по очкам/передачам/подборам |
| Профиль игрока | Radar-диаграмма, продвинутые метрики, сравнение |
| Кластеры | K-Means 5 архетипов, силуэтный анализ |
| Прогноз | XGBoost R²=0.977, MAE=0.69 |
| Монте-Карло | 8 000 симуляций на игрока, P(All-Star) |
| Trade Value | Оценка ценности обмена 0-100 |
| Контракты | Справедливая зарплата, Surplus Value |
| AI-скаутинг | Скаутские отчёты через Google Gemini |
| + 5 модулей | Аномалии, Байес, Survival, Network, Shot Chart |

---

## Структура проекта
```
app/
  dashboard.py        # Главный Streamlit-дашборд
  nba_theme.py        # NBA-тема и компоненты
src/
  fetch_data.py       # Загрузка данных через NBA API
  features.py         # Feature engineering
  clustering.py       # UMAP + K-Means
  prediction.py       # XGBoost модель
  monte_carlo.py      # Monte Carlo симуляции
  trade_value.py      # Trade Value модель
  ai_analyst.py       # Google Gemini интеграция
  ...                 # + 10 других модулей
data/                 # Генерируется скриптами (не в репозитории)
requirements.txt
```
