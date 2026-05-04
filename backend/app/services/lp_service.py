import logging
from typing import Dict, List

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

        num_nodes = self._registry.num_nodes
        thresholds = self._registry.lp_config.get("thresholds", {})

        results = {}
        for model_name in self._registry.lp_models:
            embeddings = self._get_embeddings(model_name)
            node_emb = embeddings[idx]

            # Dot product with all nodes
            scores = np.dot(embeddings, node_emb)
            # Apply sigmoid
            scores = 1.0 / (1.0 + np.exp(-scores))

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
