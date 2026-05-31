"""
Player style similarity network analysis.

Builds a cosine-similarity graph from standardised per-game statistics,
then runs custom graph algorithms implemented entirely in NumPy:

  - Force-directed spring layout (_spring_layout)
  - PageRank with dangling-node handling (_pagerank)
  - Greedy label-propagation community detection

No external graph library (networkx, igraph, etc.) is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from src.theme_utils import ensure_template

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TPL = ensure_template()   # register nba_dark if not yet registered

NET_FEATURES = [
    "PTS", "AST", "REB", "STL", "BLK",
    "FG_PCT", "FG3_PCT", "USG_PCT", "TS_PCT", "PLUS_MINUS", "efficiency",
]
SIM_THRESHOLD: float = 0.92

NBA_RED = "#C8102E"
NBA_BLUE = "#1D428A"
BG_CHART = "#0c1220"
TEXT_MUTED = "#8b9ab5"
GRID = "#141E30"
GOLD = "#FFC72C"

NBA_PALETTE = [
    "#C8102E", "#1D428A", "#FFC72C", "#27AE60",
    "#9B59B6", "#E67E22", "#1ABC9C", "#E74C3C",
    "#3498DB", "#F39C12",
]

# ---------------------------------------------------------------------------
# Internal: graph algorithms from scratch
# ---------------------------------------------------------------------------

def _spring_layout(adj: np.ndarray, iters: int = 60, seed: int = 42) -> np.ndarray:
    """
    Vectorised force-directed (Fruchterman-Reingold) layout using NumPy.

    Parameters
    ----------
    adj : np.ndarray, shape (n, n)
        Weighted adjacency matrix (symmetric, diagonal = 0).
    iters : int
        Number of iteration steps.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (n, 2)
        2-D node positions in the range roughly [-1, 1].
    """
    n = adj.shape[0]
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, (n, 2))

    k = np.sqrt(1.0 / max(n, 1))
    # Cooling schedule: linearly from t_start to t_end
    t_start, t_end = 0.15, 0.01
    temps = np.linspace(t_start, t_end, iters)

    for t in temps:
        # delta[i, j] = pos[j] - pos[i]  →  shape (n, n, 2)
        delta = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]     # (n, n, 2)
        dist = np.linalg.norm(delta, axis=2)                       # (n, n)
        np.fill_diagonal(dist, 1e6)                                # avoid self-force

        # Unit direction matrix
        with np.errstate(divide="ignore", invalid="ignore"):
            unit = delta / dist[:, :, np.newaxis]                  # (n, n, 2)

        # Repulsive force: k² / dist²  (pushes every pair apart)
        rep_mag = (k ** 2) / (dist ** 2)                           # (n, n)
        rep = -rep_mag[:, :, np.newaxis] * unit                    # (n, n, 2)

        # Attractive force: dist / k  (pulls connected pairs together)
        att_mag = (dist / k) * adj                                 # (n, n)
        att = att_mag[:, :, np.newaxis] * unit                     # (n, n, 2)

        # Net displacement per node
        disp = np.sum(rep + att, axis=1)                           # (n, 2)

        # Scale displacement by cooling temperature
        disp_norm = np.linalg.norm(disp, axis=1, keepdims=True)
        disp_norm = np.where(disp_norm < 1e-8, 1e-8, disp_norm)
        scale = np.minimum(disp_norm, t) / disp_norm
        pos = pos + disp * scale

    return pos


def _pagerank(
    adj: np.ndarray,
    d: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Standard PageRank algorithm with dangling-node handling.

    Parameters
    ----------
    adj : np.ndarray, shape (n, n)
        Binary (or weighted) adjacency matrix with zeros on the diagonal.
    d : float
        Damping factor (0.85 is standard).
    max_iter : int
        Maximum power-iteration steps.
    tol : float
        Convergence tolerance on the L1 norm of rank change.

    Returns
    -------
    np.ndarray, shape (n,)
        PageRank scores normalised to sum to 1, then scaled by *n* so that
        the average node has rank ≈ 1.0.
    """
    n = adj.shape[0]
    if n == 0:
        return np.array([])

    # Row-normalise to get transition matrix
    out_deg = adj.sum(axis=1, keepdims=True)
    dangling = (out_deg.ravel() == 0)
    out_deg_safe = np.where(out_deg == 0, 1.0, out_deg)
    M = adj / out_deg_safe                                         # (n, n)

    rank = np.full(n, 1.0 / n)
    teleport = np.full(n, 1.0 / n)

    for _ in range(max_iter):
        # Handle dangling nodes: spread their rank uniformly
        dangling_sum = rank[dangling].sum()
        new_rank = (
            d * (M.T @ rank + dangling_sum * teleport)
            + (1.0 - d) * teleport
        )
        if np.abs(new_rank - rank).sum() < tol:
            rank = new_rank
            break
        rank = new_rank

    # Normalise and scale so average = 1
    rank = rank / rank.sum()
    rank = rank * n
    return rank


# ---------------------------------------------------------------------------
# Public: compute network metrics
# ---------------------------------------------------------------------------

def compute_network(
    df: pd.DataFrame,
    threshold: float = SIM_THRESHOLD,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Build the player similarity network and compute graph metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Player DataFrame containing at least some of NET_FEATURES.
    threshold : float
        Cosine similarity threshold above which two players are connected.

    Returns
    -------
    sim_matrix : np.ndarray, shape (n, n)
        Pairwise cosine similarity matrix.
    names : list[str]
        Player names corresponding to matrix rows/columns.
    metrics_df : pd.DataFrame
        Columns: name_col, pagerank, degree, n_similar, net_community,
        and cluster_name if present in *df*.
    """
    # Select and clean features
    avail = [f for f in NET_FEATURES if f in df.columns]
    if not avail:
        raise ValueError(f"None of NET_FEATURES found in DataFrame columns: {df.columns.tolist()}")

    sub = df[avail].fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sub.values)

    sim_matrix = cosine_similarity(X_scaled).astype(float)   # (n, n)

    # Name column
    name_col = next(
        (c for c in ("PLAYER_NAME", "player_name", "name", "Name") if c in df.columns),
        None,
    )
    names: list[str] = (
        df[name_col].fillna("Unknown").tolist()
        if name_col else [f"Player_{i}" for i in range(len(df))]
    )

    n = len(names)

    # Binary adjacency: above threshold, no self-loops
    adj = (sim_matrix >= threshold).astype(float)
    np.fill_diagonal(adj, 0.0)

    # PageRank
    pr = _pagerank(adj) * 100.0   # scale to percentage-like values

    # Degree centrality (fraction of possible connections)
    degree = adj.sum(axis=1) / max(n - 1, 1) * 100.0

    # n_similar: raw count of connected neighbours
    n_similar = adj.sum(axis=1).astype(int)

    # Community detection via greedy label propagation
    # Sort edges by similarity descending; merge smaller community into larger
    community = np.arange(n, dtype=int)   # each node starts as its own community
    community_size = np.ones(n, dtype=int)

    # Get all edges sorted by similarity (descending)
    rows_idx, cols_idx = np.triu_indices(n, k=1)
    edge_sims = sim_matrix[rows_idx, cols_idx]
    order = np.argsort(edge_sims)[::-1]

    target_n_communities = 6
    for eidx in order:
        i, j = int(rows_idx[eidx]), int(cols_idx[eidx])
        ci, cj = community[i], community[j]
        if ci == cj:
            continue
        # Count distinct communities
        n_comm = len(np.unique(community))
        if n_comm <= target_n_communities:
            break
        # Merge smaller community into larger
        if community_size[ci] >= community_size[cj]:
            # Absorb cj into ci
            mask = community == cj
            community[mask] = ci
            community_size[ci] += community_size[cj]
        else:
            mask = community == ci
            community[mask] = cj
            community_size[cj] += community_size[ci]

    # Remap community ids to 0-based integers
    unique_comms = sorted(set(community.tolist()))
    comm_map = {old: new for new, old in enumerate(unique_comms)}
    community = np.array([comm_map[c] for c in community], dtype=int)

    # Build metrics DataFrame
    metrics_dict: dict[str, list] = {
        "name_col": names,
        "pagerank": pr.tolist(),
        "degree": degree.tolist(),
        "n_similar": n_similar.tolist(),
        "net_community": community.tolist(),
    }
    if "cluster_name" in df.columns:
        metrics_dict["cluster_name"] = df["cluster_name"].tolist()

    metrics_df = pd.DataFrame(metrics_dict)
    return sim_matrix, names, metrics_df


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_network_graph(
    sim_matrix: np.ndarray,
    names: list[str],
    metrics_df: pd.DataFrame,
    threshold: float = SIM_THRESHOLD,
    max_nodes: int = 70,
) -> go.Figure:
    """
    Interactive force-directed network graph of the top-*max_nodes* players.

    Nodes: sized by PageRank, colored by cluster/community.
    Edges: thin translucent lines for cosine similarity >= threshold.
    Labels: last name only in tiny muted text.
    """
    n_total = len(names)
    if n_total == 0:
        return go.Figure()

    # Select top nodes by pagerank
    pr_vals = metrics_df["pagerank"].values
    top_idx = np.argsort(pr_vals)[::-1][:max_nodes]
    top_idx_sorted = np.sort(top_idx)

    sub_names = [names[i] for i in top_idx_sorted]
    sub_sim = sim_matrix[np.ix_(top_idx_sorted, top_idx_sorted)]
    sub_adj = (sub_sim >= threshold).astype(float)
    np.fill_diagonal(sub_adj, 0.0)
    sub_metrics = metrics_df.iloc[top_idx_sorted].reset_index(drop=True)

    # Spring layout
    pos = _spring_layout(sub_adj, iters=60, seed=42)
    node_x = pos[:, 0].tolist()
    node_y = pos[:, 1].tolist()

    # Determine community colors
    comm_ids = sub_metrics["net_community"].values
    unique_comms = sorted(set(comm_ids.tolist()))
    comm_color_map = {c: NBA_PALETTE[i % len(NBA_PALETTE)] for i, c in enumerate(unique_comms)}

    # Edge traces — one trace per connected pair
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    n_sub = len(sub_names)
    for i in range(n_sub):
        for j in range(i + 1, n_sub):
            if sub_adj[i, j] > 0:
                edge_x += [node_x[i], node_x[j], None]
                edge_y += [node_y[i], node_y[j], None]

    fig = go.Figure()

    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.07)", width=0.7),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Node traces grouped by community for legend
    for comm_id in unique_comms:
        mask = comm_ids == comm_id
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue

        pr_sub = sub_metrics["pagerank"].values[idxs]
        sizes = 8.0 + (pr_sub / max(pr_sub.max(), 1.0)) * 14.0

        comm_names = [sub_names[i] for i in idxs]
        last_names = [n.split()[-1] if " " in n else n for n in comm_names]

        hover_texts = []
        for k, i in enumerate(idxs):
            row = sub_metrics.iloc[i]
            cluster = row.get("cluster_name", f"Группа {comm_id}")
            hover_texts.append(
                f"<b>{comm_names[k]}</b><br>"
                f"PageRank: {row['pagerank']:.2f}<br>"
                f"Связей: {int(row['n_similar'])}<br>"
                f"Тип: {cluster}"
            )

        color = comm_color_map[comm_id]
        cluster_label = (
            sub_metrics.loc[sub_metrics["net_community"] == comm_id, "cluster_name"].iloc[0]
            if "cluster_name" in sub_metrics.columns
            else f"Группа {comm_id + 1}"
        )

        fig.add_trace(go.Scatter(
            x=[node_x[i] for i in idxs],
            y=[node_y[i] for i in idxs],
            mode="markers+text",
            name=str(cluster_label),
            marker=dict(
                color=color,
                size=sizes.tolist(),
                opacity=0.90,
                line=dict(color="rgba(255,255,255,0.25)", width=0.8),
            ),
            text=last_names,
            textposition="top center",
            textfont=dict(size=7, color="rgba(255,255,255,0.50)"),
            hovertext=hover_texts,
            hoverinfo="text",
        ))

    fig.update_layout(
        title=dict(
            text=f"Граф стилевой схожести — топ-{max_nodes} игроков",
            font=dict(size=14, color="#e8eaf6"),
        ),
        height=600,
        template="nba_dark",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def plot_centrality_leaderboard(
    metrics_df: pd.DataFrame,
    metric: str = "pagerank",
    n: int = 20,
) -> go.Figure:
    """
    Horizontal bar chart ranking players by a centrality metric.

    Supported metrics: "pagerank", "degree", "n_similar".
    """
    metric_titles = {
        "pagerank":   "PageRank — влиятельность игроков в сети схожести",
        "degree":     "Степень централизации — % связей к максимально возможным",
        "n_similar":  "Количество схожих игроков (выше порогового значения)",
    }
    metric_labels = {
        "pagerank":  "PageRank (масштаб)",
        "degree":    "Степень централизации (%)",
        "n_similar": "Кол-во схожих игроков",
    }

    title = metric_titles.get(metric, f"Рейтинг по {metric}")
    x_label = metric_labels.get(metric, metric)

    if metric not in metrics_df.columns:
        raise ValueError(f"Metric '{metric}' not found in metrics_df columns.")

    top = (
        metrics_df[["name_col", metric]]
        .dropna()
        .sort_values(metric, ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

    # Color gradient: NBA_RED (top rank) → NBA_BLUE (lower rank)
    n_rows = len(top)
    colors = [
        f"rgba({int(200 - (200 - 29) * i / max(n_rows - 1, 1))},"
        f"{int(16 + (66 - 16) * i / max(n_rows - 1, 1))},"
        f"{int(46 + (138 - 46) * i / max(n_rows - 1, 1))},0.90)"
        for i in range(n_rows)
    ]

    fig = go.Figure(go.Bar(
        x=top[metric],
        y=top["name_col"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.2f}" for v in top[metric]],
        textposition="outside",
        textfont=dict(color=TEXT_MUTED, size=9),
        hovertemplate="%{y}: %{x:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#e8eaf6")),
        height=max(360, n * 22),
        template="nba_dark",
        xaxis=dict(title=x_label),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=160, r=70, t=60, b=40),
    )
    return fig


def plot_similarity_clusters(
    sim_matrix: np.ndarray,
    names: list[str],
    metrics_df: pd.DataFrame,
    n: int = 60,
) -> go.Figure:
    """
    Heatmap of pairwise similarity for the top-*n* players by PageRank.

    Players are sorted by community so that clusters appear as coherent blocks.
    Rectangle annotations highlight each community's bounding box.
    """
    n_total = len(names)
    if n_total == 0:
        return go.Figure()

    # Top n by pagerank
    pr_vals = metrics_df["pagerank"].values
    top_idx = np.argsort(pr_vals)[::-1][:n]

    # Sort by community within selection
    sub_metrics = metrics_df.iloc[top_idx].copy()
    sub_metrics = sub_metrics.sort_values("net_community").reset_index(drop=True)
    sorted_original_idx = top_idx[sub_metrics.index]

    sub_sim = sim_matrix[np.ix_(sorted_original_idx, sorted_original_idx)]
    sub_names = [names[i] for i in sorted_original_idx]
    comms = sub_metrics["net_community"].values

    # Shorten display names
    short_names = [nm.split()[-1] if " " in nm else nm for nm in sub_names]

    fig = go.Figure(go.Heatmap(
        z=sub_sim,
        x=short_names,
        y=short_names,
        colorscale=[[0, "#080c14"], [0.5, "#1D428A"], [1, "#C8102E"]],
        zmin=0.0,
        zmax=1.0,
        colorbar=dict(
            thickness=10,
            len=0.7,
            tickfont=dict(color="#6b7a99", size=10),
            title=dict(
                text="Схожесть",
                font=dict(color="#8b9ab5", size=11),
            ),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.06)",
        ),
        hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
    ))

    # Community boundary rectangles
    unique_comms = sorted(set(comms.tolist()))
    for comm_id in unique_comms:
        idxs = np.where(comms == comm_id)[0]
        if len(idxs) < 2:
            continue
        lo = int(idxs.min()) - 0.5
        hi = int(idxs.max()) + 0.5
        fig.add_shape(
            type="rect",
            x0=lo, x1=hi,
            y0=lo, y1=hi,
            line=dict(color="rgba(255,199,44,0.5)", width=1.5, dash="dot"),
            fillcolor="rgba(0,0,0,0)",
        )

    fig.update_layout(
        title=dict(
            text="Матрица схожести — блочная структура стилей",
            font=dict(size=14, color="#e8eaf6"),
        ),
        height=max(500, n * 8 + 100),
        template="nba_dark",
        xaxis=dict(tickfont=dict(size=7), tickangle=45),
        yaxis=dict(tickfont=dict(size=7), autorange="reversed"),
        margin=dict(l=100, r=80, t=70, b=100),
    )
    return fig
