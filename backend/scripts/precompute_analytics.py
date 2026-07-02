"""
Precompute analytics data for the dashboard endpoints.

Reads from saved_models/graph_data.npz and generates:
  - class_distribution.json
  - degree_distribution.npz
  - tsne_coords.npz
  - feature_importance.json
  - model_comparison.json

Usage:
    python scripts/precompute_analytics.py
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="saved_models")
    parser.add_argument("--output-dir", default="precomputed")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load graph data
    print("Loading graph data...")
    graph_data_path = models_dir / "graph_data.npz"
    if not graph_data_path.exists():
        raise FileNotFoundError(f"Graph data not found: {graph_data_path}")

    with np.load(graph_data_path, allow_pickle=True) as data:
        X_full = data["X_full"]
        y_full = data["y_full"]
        edge_index = data["edge_index"]
        train_mask = data["train_mask"]
        node_names = data["node_names"]
        class_names = data["class_names"].tolist()

    num_nodes = X_full.shape[0]
    num_edges_directed = edge_index.shape[1]  # both directions stored
    num_edges_undirected = num_edges_directed // 2

    # Load vectorizer for feature names
    vectorizer_path = models_dir / "tfidf_vectorizer.pkl"
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    # 1. Class distribution
    print("Computing class distribution...")
    unique, counts = np.unique(y_full, return_counts=True)
    class_dist = [
        {"class_name": class_names[int(c)], "count": int(cnt)}
        for c, cnt in zip(unique, counts)
    ]
    with open(output_dir / "class_distribution.json", "w") as f:
        json.dump(class_dist, f, indent=2)

    # 2. Degree distribution (vectorized — fast)
    print("Computing degree distribution...")
    degrees = np.bincount(edge_index[0], minlength=num_nodes)
    hist_counts, hist_bins = np.histogram(degrees, bins=50)
    np.savez(
        output_dir / "degree_distribution.npz",
        counts=hist_counts,
        bins=hist_bins,
    )

    # 3. t-SNE
    print("Computing t-SNE (sample 2000 nodes)...")
    sample_size = min(2000, num_nodes)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(num_nodes, sample_size, replace=False)
    X_sample = X_full[sample_idx]
    y_sample = y_full[sample_idx]
    asins_sample = node_names[sample_idx]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(X_sample)

    np.savez(
        output_dir / "tsne_coords.npz",
        x=coords[:, 0],
        y=coords[:, 1],
        labels=y_sample,
        asins=asins_sample,
        class_names=np.array(class_names),
    )

    # 4. Feature importance
    print("Computing feature importance (Random Forest)...")
    train_idx = np.where(train_mask.astype(bool))[0]
    X_train = X_full[train_idx]
    y_train = y_full[train_idx]

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    feature_imp = []
    with open(output_dir / "feature_importance.json", "w") as f:
        json.dump(feature_imp, f, indent=2)

    # 5. Model comparison (placeholder — fill from notebook metrics)
    print("Saving model comparison placeholder...")
    max_possible_edges = num_nodes * (num_nodes - 1) / 2
    model_comparison = {
        "graph_stats": {
            "num_nodes": int(num_nodes),
            "num_edges": int(num_edges_undirected),
            "num_classes": len(class_names),
            "avg_degree": round(num_edges_directed / num_nodes, 2),
            "density": round(num_edges_undirected / max_possible_edges, 6),
        },
        "nc_metrics": {},
        "lp_metrics": {},
    }
    with open(output_dir / "model_comparison.json", "w") as f:
        json.dump(model_comparison, f, indent=2)

    print(f"All analytics saved to {output_dir}/")


if __name__ == "__main__":
    main()
