"""
Прогноз результативности на следующий сезон.
XGBoost + Ridge Regression с кросс-валидацией.
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

FEATURE_COLS = [
    "AST", "REB", "STL", "BLK", "TOV",
    "FG_PCT", "FG3_PCT", "FT_PCT",
    "USG_PCT", "TS_PCT", "PLUS_MINUS", "efficiency",
]
TARGET = "PTS"
AGE_COL_OPTIONS = ["PLAYER_AGE", "AGE", "age"]


def build_model(df):
    """
    Обучить модель предсказания очков на текущем сезоне.
    Возвращает (model, scaler, feature_cols, cv_metrics).
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    age_col   = next((c for c in AGE_COL_OPTIONS if c in df.columns), None)
    if age_col:
        available = available + [age_col]

    X = df[available].fillna(0).values
    y = df[TARGET].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Выбираем лучшую модель
    if HAS_XGB:
        model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, random_state=42,
        )

    # Кросс-валидация
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(model, X_scaled, y, cv=cv)

    mae = mean_absolute_error(y, y_pred_cv)
    r2  = r2_score(y, y_pred_cv)

    # Финальное обучение на всех данных
    model.fit(X_scaled, y)

    metrics = {"mae": mae, "r2": r2, "features": available}

    pickle.dump(
        {"model": model, "scaler": scaler, "features": available},
        open("data/processed/prediction_model.pkl", "wb"),
    )

    return model, scaler, available, metrics, y_pred_cv


def predict_next_season(df, model, scaler, feature_cols):
    """
    Предсказать очки следующего сезона с учётом возраста.
    Возвращает df с колонками pred_pts, pred_low, pred_high.
    """
    from src.project_season import age_growth_factor

    age_col = next((c for c in AGE_COL_OPTIONS if c in df.columns), None)

    X = df[feature_cols].fillna(0).values
    X_scaled = scaler.transform(X)
    base_pred = model.predict(X_scaled)

    # Применяем возрастной коэффициент
    growth = np.array([
        age_growth_factor(row[age_col] if age_col else None)
        for _, row in df.iterrows()
    ])

    pred = base_pred * growth
    noise = base_pred * 0.08  # 8% интервал неопределённости

    df = df.copy()
    df["pred_pts"]  = pred.round(1)
    df["pred_low"]  = (pred - noise).round(1)
    df["pred_high"] = (pred + noise).round(1)
    df["pred_delta"] = (df["pred_pts"] - df[TARGET]).round(1)

    return df


def plot_actual_vs_predicted(y_true, y_pred, name_col_vals):
    """Scatter actual vs predicted с именами."""
    fig = px.scatter(
        x=y_true, y=y_pred,
        hover_name=name_col_vals,
        labels={"x": "Фактические очки", "y": "Предсказанные очки"},
        title="Модель: Фактические vs Предсказанные (кросс-валидация 5-fold)",
        template="nba_dark",
        height=450,
        color=np.abs(np.array(y_true) - np.array(y_pred)),
        color_continuous_scale="RdYlGn_r",
    )
    # Диагональ идеального предсказания
    mn = min(min(y_true), min(y_pred))
    mx = max(max(y_true), max(y_pred))
    fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                  line=dict(color="grey", dash="dash", width=1))
    fig.update_coloraxes(showscale=False)
    return fig


def plot_feature_importance(model, feature_cols):
    """Bar chart важности признаков."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        return None

    labels = {
        "AST": "Передачи", "REB": "Подборы", "STL": "Перехваты",
        "BLK": "Блок-шоты", "TOV": "Потери", "FG_PCT": "FG%",
        "FG3_PCT": "3P%", "FT_PCT": "FT%", "USG_PCT": "USG%",
        "TS_PCT": "TS%", "PLUS_MINUS": "+/-", "efficiency": "Эффективность",
        "PLAYER_AGE": "Возраст", "AGE": "Возраст",
    }
    feat_df = pd.DataFrame({
        "Признак": [labels.get(f, f) for f in feature_cols],
        "Важность": imp,
    }).sort_values("Важность", ascending=True)

    fig = px.bar(feat_df, x="Важность", y="Признак", orientation="h",
                 title="Важность признаков для предсказания очков",
                 template="nba_dark", height=400,
                 color="Важность",
                 color_continuous_scale=[[0.0,"#0d1f3c"],[0.5,"#1D428A"],[1.0,"#C8102E"]])
    fig.update_coloraxes(showscale=False)
    fig.update_traces(
        texttemplate="%{x:.3f}", textposition="outside",
        textfont=dict(size=10, color="#8b9ab5"),
    )
    return fig


def plot_next_season_forecast(df, top_n=20):
    """График прогноза на следующий сезон — топ игроков."""
    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    top = df.nlargest(top_n, "pred_pts").sort_values("pred_pts")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top[TARGET], y=top[name_col],
        orientation="h", name="2025-26 факт",
        marker_color="#95a5a6", opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        x=top["pred_pts"], y=top[name_col],
        orientation="h", name="Прогноз 2026-27",
        marker_color="#1d428a", opacity=0.9,
    ))
    fig.add_trace(go.Scatter(
        x=top["pred_high"], y=top[name_col],
        mode="markers", name="Верхняя граница",
        marker=dict(symbol="line-ns", color="#c8102e", size=8, line_width=2),
    ))
    fig.update_layout(
        barmode="overlay",
        title="Прогноз результативности — следующий сезон",
        xaxis_title="Очки за игру",
        template="nba_dark",
        height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


if __name__ == "__main__":
    df = pd.read_csv("data/processed/players_clustered.csv")
    model, scaler, feats, metrics, y_cv = build_model(df)
    print(f"MAE: {metrics['mae']:.2f} pts  |  R2: {metrics['r2']:.3f}")
    df_pred = predict_next_season(df, model, scaler, feats)
    name_col = "name" if "name" in df.columns else "PLAYER_NAME"
    print("\nТоп-10 прогноз:")
    print(df_pred.nlargest(10, "pred_pts")[[name_col, "PTS", "pred_pts", "pred_delta"]].to_string(index=False))
