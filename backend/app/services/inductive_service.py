import logging
from contextlib import contextmanager
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

from app.services.model_registry import ModelRegistry
from app.ml.inference_helpers import top_k_indices
from app.ml.custom_layers import (
    StaticGraphSAGE, MultiHeadStaticGAT, StaticGraphAttn, StrictGCNModel,
)

logger = logging.getLogger(__name__)

# Number of pseudo-neighbors to inject for inductive inference
K_NEIGHBORS = 10

# Domain detection thresholds
SBERT_DOMAIN_THRESHOLD = 0.30      # SBERT cosine sims are higher for in-domain
TFIDF_MIN_WORDS = 5                # minimum non-zero TF-IDF features
TFIDF_DOMAIN_THRESHOLD = 0.10      # cosine sim threshold for TF-IDF

# LP cosine similarity gate (minimum to predict a link as real)
SBERT_LP_COS_GATE = 0.30
TFIDF_LP_COS_GATE = 0.10


class InductiveService:
    """Cold-start inference using similarity-based neighbor injection.

    Industry-standard approach (used by PinSage, Uber Eats, etc.):
    1. New node arrives with features but no edges
    2. Find k nearest existing nodes via cosine similarity
    3. Inject bidirectional edges between new node and pseudo-neighbors
    4. Run full GNN forward pass with the augmented graph
    5. Remove injected edges (restore model state)
    """

    def __init__(self, registry: ModelRegistry):
        self._registry = registry
        self._lp_embeddings_cache: Dict[str, np.ndarray] = {}
        self._X_full_normalized: Optional[np.ndarray] = None

    # ── public ────────────────────────────────────────────────────────────────

    def vectorize_text(self, title: str, description: str) -> np.ndarray:
        """Return (1, num_features) float32 feature vector."""
        text = f"{title} {description}"
        if self._registry._vectorizer_type == "sbert":
            vec = self._registry.vectorizer.encode(
                [text], convert_to_numpy=True
            ).astype(np.float32)
            return vec  # shape (1, 384)
        return self._registry.vectorizer.transform([text]).toarray().astype(np.float32)

    def predict_full(self, title: str, description: str, top_k: int = 10) -> dict:
        """Combined NC + LP inductive prediction."""
        text_vec = self.vectorize_text(title, description)

        # Compute neighbors once
        neighbor_indices, all_similarities, max_sim = self._find_pseudo_neighbors(
            text_vec, k=K_NEIGHBORS
        )

        warning = self._check_domain(text_vec, max_sim)

        # Pre-cache LP embeddings BEFORE NC pass (NC patches shared graph_info)
        for model_name in self._registry.lp_models:
            self._get_lp_embeddings(model_name)

        nc_results = self._run_nc(text_vec, neighbor_indices)
        lp_results = self._run_lp(text_vec, neighbor_indices, all_similarities, top_k)
        result = {"nc_predictions": nc_results, "lp_predictions": lp_results}
        if warning:
            result["warning"] = warning
        return result

    def predict_nc_inductive(self, text_vec: np.ndarray) -> Dict[str, dict]:
        neighbor_indices, _, _ = self._find_pseudo_neighbors(text_vec, k=K_NEIGHBORS)
        return self._run_nc(text_vec, neighbor_indices)

    def predict_lp_inductive(self, text_vec: np.ndarray, top_k: int = 10) -> Dict[str, List[dict]]:
        neighbor_indices, all_similarities, _ = self._find_pseudo_neighbors(
            text_vec, k=K_NEIGHBORS
        )
        # Pre-cache LP embeddings before any graph_info patching
        for model_name in self._registry.lp_models:
            self._get_lp_embeddings(model_name)
        return self._run_lp(text_vec, neighbor_indices, all_similarities, top_k)

    # ── domain check ──────────────────────────────────────────────────────────

    def _check_domain(self, text_vec: np.ndarray, max_sim: float) -> Optional[str]:
        if self._registry._vectorizer_type == "sbert":
            # SBERT: all embeddings are dense, use cosine similarity only
            if max_sim < SBERT_DOMAIN_THRESHOLD:
                return (
                    f"This product has low similarity to electronics products "
                    f"(max cosine similarity: {max_sim:.3f}). "
                    "Predictions may be unreliable."
                )
        else:
            # TF-IDF: check word count + cosine similarity
            nonzero_features = int(np.count_nonzero(text_vec))
            if nonzero_features == 0:
                return ("Input text contains no recognized vocabulary words. "
                        "Predictions are unreliable (random).")
            elif nonzero_features < TFIDF_MIN_WORDS:
                return (
                    f"Only {nonzero_features} word(s) matched the electronics vocabulary. "
                    "This product may not be electronics-related — predictions are unreliable."
                )
            elif max_sim < TFIDF_DOMAIN_THRESHOLD:
                return (
                    f"This product has low similarity to electronics products "
                    f"(max cosine similarity: {max_sim:.3f}). Predictions may be unreliable."
                )
        return None

    # ── internal NC/LP runners ────────────────────────────────────────────────

    def _run_nc(self, text_vec: np.ndarray, neighbor_indices: np.ndarray) -> Dict[str, dict]:
        X_extended = np.vstack([self._registry.X_full, text_vec.reshape(1, -1)])
        new_idx = X_extended.shape[0] - 1
        extended_edges = self._build_extended_edges(neighbor_indices, new_idx)
        results = {}
        for model_name, model in self._registry.nc_models.items():
            node_probs = self._nc_forward(
                model_name, model, X_extended, new_idx, text_vec, extended_edges
            )
            results[model_name] = self._format_nc_result(node_probs)
        return results

    def _run_lp(self, text_vec: np.ndarray, neighbor_indices: np.ndarray,
                all_similarities: np.ndarray, top_k: int) -> Dict[str, List[dict]]:
        cos_sim_gate = (SBERT_LP_COS_GATE if self._registry._vectorizer_type == "sbert"
                        else TFIDF_LP_COS_GATE)

        X_extended = np.vstack([self._registry.X_full, text_vec.reshape(1, -1)])
        new_idx = X_extended.shape[0] - 1
        thresholds = self._registry.lp_config.get("thresholds", {})
        extended_edges = self._build_extended_edges(neighbor_indices, new_idx)

        results = {}
        for model_name, model in self._registry.lp_models.items():
            scores = self._lp_scores(
                model_name, model, X_extended, new_idx, text_vec, extended_edges
            )
            threshold = thresholds.get(model_name, 0.5)
            top_indices = top_k_indices(scores, top_k)

            suggestions = []
            for target_idx in top_indices:
                target_asin = str(self._registry.node_names[target_idx])
                score = float(scores[target_idx])
                cos_sim = float(all_similarities[target_idx])
                # Both model threshold AND cosine similarity gate must pass
                predicted_link = score >= threshold and cos_sim >= cos_sim_gate
                suggestions.append({
                    "asin": target_asin,
                    "score": score,
                    "predicted_link": predicted_link,
                    "target_class": self._registry.get_class_name(
                        int(self._registry.y_full[target_idx])
                    ),
                })
            results[model_name] = suggestions

        return results

    # ── neighbor injection ────────────────────────────────────────────────────

    def _find_pseudo_neighbors(self, text_vec: np.ndarray, k: int = 10):
        """Find k most similar existing nodes via cosine similarity.

        Returns (neighbor_indices, all_similarities, max_similarity).
        """
        if self._X_full_normalized is None:
            norms = np.linalg.norm(self._registry.X_full, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._X_full_normalized = self._registry.X_full / norms

        query_norm = np.linalg.norm(text_vec)
        if query_norm < 1e-8:
            random_idx = np.random.choice(self._registry.num_nodes, size=k, replace=False)
            return random_idx, np.zeros(self._registry.num_nodes), 0.0

        query_normalized = text_vec.flatten() / query_norm
        similarities = self._X_full_normalized @ query_normalized
        top_indices = np.argsort(similarities)[::-1][:k]
        max_similarity = float(similarities[top_indices[0]])
        return top_indices, similarities, max_similarity

    def _build_extended_edges(self, neighbor_indices: np.ndarray, new_node_idx: int) -> tf.Tensor:
        """Build (2, 2*k) tensor of bidirectional edges to pseudo-neighbors."""
        src = np.concatenate([
            np.full(len(neighbor_indices), new_node_idx),
            neighbor_indices
        ])
        dst = np.concatenate([
            neighbor_indices,
            np.full(len(neighbor_indices), new_node_idx)
        ])
        return tf.constant(np.stack([src, dst], axis=0), dtype=tf.int32)

    # ── model patching context managers ───────────────────────────────────────

    @contextmanager
    def _patch_sage_edges(self, model_layer, new_edges: tf.Tensor):
        """Temporarily extend edge_index on a StaticGraphSAGE layer."""
        orig_ei_conv1 = model_layer.conv1.edge_index
        orig_ei_conv2 = model_layer.conv2.edge_index
        extended_ei = tf.concat([orig_ei_conv1, new_edges], axis=1)
        model_layer.conv1.edge_index = extended_ei
        model_layer.conv2.edge_index = extended_ei
        try:
            yield
        finally:
            model_layer.conv1.edge_index = orig_ei_conv1
            model_layer.conv2.edge_index = orig_ei_conv2

    @contextmanager
    def _patch_gat_edges(self, model, new_edges: tf.Tensor, is_lp=False):
        """Temporarily extend edge_index on GAT layers."""
        if is_lp:
            gat1 = model.gat1
            gat2 = model.gat2
        else:
            gat1 = next(l for l in model.layers if isinstance(l, MultiHeadStaticGAT))
            gat2 = next(l for l in model.layers if isinstance(l, StaticGraphAttn))

        orig_gat1_ei = gat1.edge_index
        orig_head_eis = [h.edge_index for h in gat1.heads]
        orig_gat2_ei = gat2.edge_index

        extended_ei = tf.concat([orig_gat1_ei, new_edges], axis=1)
        gat1.edge_index = extended_ei
        for h in gat1.heads:
            h.edge_index = extended_ei
        gat2.edge_index = extended_ei
        try:
            yield
        finally:
            gat1.edge_index = orig_gat1_ei
            for h, orig_ei in zip(gat1.heads, orig_head_eis):
                h.edge_index = orig_ei
            gat2.edge_index = orig_gat2_ei

    @contextmanager
    def _patch_gcn_graph_info(self, model, new_edges: tf.Tensor, new_num_nodes: int, is_lp=False):
        """Temporarily extend graph_info on GCN layers.

        conv1 and conv2 share the SAME graph_info object, so patch it once.
        """
        if is_lp:
            graph_info = model.conv1.graph_info
        else:
            gcn_model_layer = next(l for l in model.layers if isinstance(l, StrictGCNModel))
            graph_info = gcn_model_layer.gcn1.graph_info

        orig_edges = graph_info.edges
        orig_num_nodes = graph_info.num_nodes

        new_self_loop = tf.constant([[new_num_nodes - 1], [new_num_nodes - 1]], dtype=tf.int32)
        extended = tf.concat([orig_edges, new_edges, new_self_loop], axis=1)
        graph_info.edges = extended
        graph_info.num_nodes = new_num_nodes
        try:
            yield
        finally:
            graph_info.edges = orig_edges
            graph_info.num_nodes = orig_num_nodes

    # ── NC internals ──────────────────────────────────────────────────────────

    def _nc_forward(self, model_name: str, model, X_extended: np.ndarray,
                    new_idx: int, text_vec: np.ndarray, extended_edges: tf.Tensor) -> np.ndarray:
        """Run one NC model with neighbor injection.

        Uses direct model call (eager) instead of model.predict() to avoid
        TF function tracing which caches the patched edge_index tensor.
        """

        if model_name == "MLP":
            preds = model(tf.constant(text_vec.reshape(1, -1), dtype=tf.float32), training=False)
            return preds.numpy()[0]

        X_tf = tf.constant(X_extended, dtype=tf.float32)

        if model_name == "GraphSAGE":
            sage_layer = next(l for l in model.layers if isinstance(l, StaticGraphSAGE))
            with self._patch_sage_edges(sage_layer, extended_edges):
                preds = model(X_tf, training=False)
            return preds.numpy()[new_idx]

        if model_name == "GAT":
            with self._patch_gat_edges(model, extended_edges, is_lp=False):
                preds = model(X_tf, training=False)
            return preds.numpy()[new_idx]

        if model_name == "GCN":
            with self._patch_gcn_graph_info(model, extended_edges, X_extended.shape[0], is_lp=False):
                preds = model(X_tf, training=False)
            return preds.numpy()[new_idx]

        logger.warning("NC inductive: unknown model %s", model_name)
        return np.ones(self._registry.num_classes, dtype=np.float32) / self._registry.num_classes

    def _format_nc_result(self, node_probs: np.ndarray) -> dict:
        predicted_class_idx = int(np.argmax(node_probs))
        top3_idx = np.argsort(node_probs)[::-1][:3]
        return {
            "predicted_class": self._registry.get_class_name(predicted_class_idx),
            "confidence": float(node_probs[predicted_class_idx]),
            "top_3": [
                {
                    "class": self._registry.get_class_name(int(i)),
                    "probability": float(node_probs[i]),
                }
                for i in top3_idx
            ],
        }

    # ── LP internals ──────────────────────────────────────────────────────────

    def _get_lp_embeddings(self, model_name: str) -> np.ndarray:
        """Get LP embeddings with caching. Must be called BEFORE any graph_info patching."""
        if model_name not in self._lp_embeddings_cache:
            model = self._registry.lp_models[model_name]
            self._lp_embeddings_cache[model_name] = model.get_embeddings(
                training=False
            ).numpy()
        return self._lp_embeddings_cache[model_name]

    def _lp_scores(self, model_name: str, model, X_extended: np.ndarray,
                   new_idx: int, text_vec: np.ndarray, extended_edges: tf.Tensor) -> np.ndarray:
        """Return (num_existing_nodes,) sigmoid-score array.

        For GNN models (GraphSAGE, GAT, GCN), we avoid passing X_extended
        through the encoder (TF tracing caches input shape, causing crashes).
        Instead: use cached embeddings + neighbor-mean approximation.
        """

        if model_name == "MLP":
            existing_emb = self._get_lp_embeddings(model_name)
            x = tf.constant(text_vec.reshape(1, -1), dtype=tf.float32)
            x = model.drop1(x, training=False)
            x = model.dense1(x)
            x = model.drop2(x, training=False)
            x = model.dense2(x)
            new_emb = tf.math.l2_normalize(x, axis=-1).numpy().flatten()
            scores = np.dot(existing_emb, new_emb)
            return 1.0 / (1.0 + np.exp(-scores))

        # For GNN LP models: use cached embeddings + neighbor-mean approximation
        if model_name in ("GraphSAGE", "GAT", "GCN"):
            existing_emb = self._get_lp_embeddings(model_name)
            # Approximate new node embedding as mean of pseudo-neighbors
            neighbor_indices = extended_edges[1, :K_NEIGHBORS].numpy()  # first K are new->neighbor
            new_emb = np.mean(existing_emb[neighbor_indices], axis=0)
            new_emb = new_emb / (np.linalg.norm(new_emb) + 1e-9)
            existing_emb_normed = existing_emb / (np.linalg.norm(existing_emb, axis=1, keepdims=True) + 1e-9)
            scores = np.dot(existing_emb_normed, new_emb)
            return 1.0 / (1.0 + np.exp(-scores))

        logger.warning("LP inductive: unknown model %s", model_name)
        return np.full(self._registry.num_nodes, 0.5)
