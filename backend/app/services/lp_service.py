import logging
from typing import Dict, List, Set, Tuple

import numpy as np
import tensorflow as tf

from app.services.model_registry import ModelRegistry
from app.ml.inference_helpers import top_k_indices

logger = logging.getLogger(__name__)


class LPService:
    """Link Prediction inference for existing graph nodes."""

    def __init__(self, registry: ModelRegistry):
        self._registry = registry
        self._embeddings_cache: Dict[str, np.ndarray] = {}

    def _get_embeddings(self, model_name: str) -> np.ndarray:
        """Get node embeddings from LP model (cached)."""
        if model_name in self._embeddings_cache:
            return self._embeddings_cache[model_name]

        model = self._registry.lp_models.get(model_name)
        if model is None:
            raise ValueError(f"LP model '{model_name}' not loaded")

        embeddings = model.get_embeddings(training=False).numpy()
        self._embeddings_cache[model_name] = embeddings
        return embeddings

    def score_pairs(self, pairs: np.ndarray) -> Dict[str, np.ndarray]:
        """Score (M, 2) pairs of node indices across all LP models."""
        pairs_tf = tf.constant(pairs, dtype=tf.int32)
        results = {}
        for model_name, model in self._registry.lp_models.items():
            scores = model.predict(pairs_tf, batch_size=len(pairs), verbose=0)
            results[model_name] = scores.flatten()
        return results

    def top_k_for_node(
        self, asin: str, k: int = 10, exclude_real: bool = True
    ) -> Dict[str, List[dict]]:
        """For a given node, score against all candidates, return top-K per model."""
        idx = self._registry.asin_to_idx.get(asin)
        if idx is None:
            raise KeyError(f"ASIN '{asin}' not found in graph")

        # Find real neighbors to exclude
        exclude = set()
        if exclude_real:
            edge_index = self._registry.edge_index
            # Find all neighbors (both directions since undirected)
            mask_src = edge_index[0] == idx
            mask_dst = edge_index[1] == idx
            neighbors_from_src = set(edge_index[1][mask_src].tolist())
            neighbors_from_dst = set(edge_index[0][mask_dst].tolist())
            exclude = neighbors_from_src | neighbors_from_dst
            exclude.add(idx)  # exclude self

        thresholds = self._registry.lp_config.get("thresholds", {})

        COSINE_GATE = 0.40

        results = {}
        for model_name in self._registry.lp_models:
            embeddings = self._get_embeddings(model_name)
            node_emb = embeddings[idx]

            # Cosine gate: zero out embedding-distant nodes before scoring
            norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(node_emb) + 1e-9
            cos_sim = np.dot(embeddings, node_emb) / norms

            # Dot product with all nodes
            scores = np.dot(embeddings, node_emb)
            # Apply sigmoid
            scores = 1.0 / (1.0 + np.exp(-scores))

            # Gate: discard candidates below cosine threshold
            scores[cos_sim < COSINE_GATE] = 0.0

            top_indices = top_k_indices(scores, k, exclude)
            threshold = thresholds.get(model_name, 0.5)

            suggestions = []
            for target_idx in top_indices:
                target_asin = self._registry.node_names[target_idx]
                score = float(scores[target_idx])
                suggestions.append({
                    "asin": str(target_asin),
                    "score": score,
                    "predicted_link": score >= threshold,
                    "target_class": self._registry.get_class_name(
                        int(self._registry.y_full[target_idx])
                    ),
                })
            results[model_name] = suggestions

        return results

    def score_neighborhood_edges(
        self,
        model_name: str,
        center_idx: int,
        neighborhood: Set[int],
        real_edges: Set[Tuple[int, int]],
    ) -> dict:
        """Classify center-node edges as TP, FP, or FN for one model.
        """
        embeddings = self._get_embeddings(model_name)
        threshold = self._registry.lp_config.get("thresholds", {}).get(
            model_name, 0.5
        )

        def _score(src: int, dst: int) -> float:
            s = float(np.dot(embeddings[src], embeddings[dst]))
            return 1.0 / (1.0 + np.exp(-s))

        # Real neighbors of center within the neighborhood
        real_neighbors_of_center = set()
        for s, d in real_edges:
            if s == center_idx:
                real_neighbors_of_center.add(d)
            elif d == center_idx:
                real_neighbors_of_center.add(s)

        tp_edges = []
        fn_edges = []

        # Classify only real edges involving the center node
        for neighbor in real_neighbors_of_center:
            score = _score(center_idx, neighbor)
            entry = {"src_idx": center_idx, "dst_idx": neighbor, "score": score}
            if score >= threshold:
                tp_edges.append(entry)
            else:
                fn_edges.append(entry)

        # FP: center node scored against non-neighbors in neighborhood
        fp_edges = []
        for node_idx in neighborhood:
            if node_idx == center_idx or node_idx in real_neighbors_of_center:
                continue
            score = _score(center_idx, node_idx)
            if score >= threshold:
                fp_edges.append(
                    {"src_idx": center_idx, "dst_idx": node_idx, "score": score}
                )

        # Also check top FP candidates from outside the neighborhood
        center_emb = embeddings[center_idx]
        all_scores = np.dot(embeddings, center_emb)
        all_scores = 1.0 / (1.0 + np.exp(-all_scores))

        # Real neighbors of center (full graph, not just neighborhood)
        edge_index = self._registry.edge_index
        mask_src = edge_index[0] == center_idx
        mask_dst = edge_index[1] == center_idx
        all_real_neighbors = (
            set(edge_index[1][mask_src].tolist())
            | set(edge_index[0][mask_dst].tolist())
        )
        all_real_neighbors.add(center_idx)

        above_threshold = np.where(all_scores >= threshold)[0]
        for idx in above_threshold:
            idx_int = int(idx)
            if idx_int in all_real_neighbors or idx_int in neighborhood:
                continue
            fp_edges.append(
                {"src_idx": center_idx, "dst_idx": idx_int, "score": float(all_scores[idx_int])}
            )

        # Sort FP by score descending, cap at 10
        fp_edges.sort(key=lambda e: e["score"], reverse=True)
        fp_edges = fp_edges[:10]

        return {"tp_edges": tp_edges, "fp_edges": fp_edges, "fn_edges": fn_edges}
