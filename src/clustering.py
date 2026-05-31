import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap

FEATURE_COLS = [
    "PTS", "AST", "REB", "STL", "BLK",
    "FG_PCT", "FG3_PCT", "TOV", "PLUS_MINUS",
    "USG_PCT", "TS_PCT", "efficiency",
]

CLUSTER_NAMES = {
    0: "Звезда лиги",
    1: "Плеймейкер",
    2: "Большой (Big Man)",
    3: "3-and-D игрок",
    4: "Ролевой игрок",
}


def run_clustering(df, n_clusters=5):
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    print(f"Признаки: {available_cols}")

    X = df[available_cols].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.80, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA: {X_pca.shape[1]} компонент, {pca.explained_variance_ratio_.sum():.1%} дисперсии")

    print("Строим UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_pca)

    print("Подбираем k:")
    scores = {}
    for k in range(3, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)
        scores[k] = silhouette_score(X_pca, labels)
        print(f"  k={k}  silhouette={scores[k]:.3f}")
    best_k = max(scores, key=scores.get)
    print(f"Лучший k = {best_k} (используем {n_clusters})")

    km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = km_final.fit_predict(X_pca)
    df["cluster_name"] = df["cluster"].map(CLUSTER_NAMES)
    df["umap_x"] = X_umap[:, 0]
    df["umap_y"] = X_umap[:, 1]

    pickle.dump(
        {"scaler": scaler, "pca": pca, "reducer": reducer, "kmeans": km_final},
        open("data/processed/models.pkl", "wb"),
    )

    df.to_csv("data/processed/players_clustered.csv", index=False)
    print("\nРаспределение:")
    print(df.groupby("cluster_name").size().to_string())
    print("\nСредние по кластерам:")
    show = [c for c in ["PTS", "AST", "REB", "STL", "BLK", "FG_PCT"] if c in df.columns]
    print(df.groupby("cluster_name")[show].mean().round(2).to_string())
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/players_features.csv")
    run_clustering(df)
