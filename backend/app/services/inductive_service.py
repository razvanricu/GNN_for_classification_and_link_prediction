import logging
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf

from app.services.model_registry import ModelRegistry
from app.ml.inference_helpers import top_k_indices

logger = logging.getLogger(__name__)


class InductiveService:
    """Cold-start inference for new products not in the graph.

    Model behaviour for an isolated new node:
    ───────────────────────────────────────────
    SAGE NC / SAGE LP
        SAGEConvLayer uses tf.shape(features)[0] (dynamic) for segment counts,
        so passing X_extended (N+1 rows) works correctly.  The new node has no
        edges, so aggregated_neighbors = 0 — the layer computes
        W @ concat([x_new, 0]) which is the correct inductive result.

    MLP NC / MLP LP
        No graph needed — runs directly on the new feature vector.

    GCN NC / GAT NC
        StrictGCNConv uses self.graph_info.num_nodes (fixed N) for
        unsorted_segment_sum, so X_extended causes OOB output.
        These fall back to the MLP NC prediction as an approximation.

    GCN LP / GAT LP
        Same fixed-N issue in the encoder.  We approximate using
        the MLP LP encoder for the new node.
    """

    def __init__(self, registry: ModelRegistry):
        self._registry = registry
        self._sage_lp_embeddings: Optional[np.ndarray] = None

    # ── public ────────────────────────────────────────────────────────────────

    def vectorize_text(self, title: str, description: str) -> np.ndarray:
        """Return (1, num_features) float32 TF-IDF vector."""
        text = f"{title} {description}"
        return self._registry.vectorizer.transform([text]).toarray().astype(np.float32)

    def predict_nc_inductive(self, text_vec: np.ndarray) -> Dict[str, dict]:
        """Run NC inference for a new product across all models."""
        X_extended = np.vstack([self._registry.X_full, text_vec.reshape(1, -1)])
        new_idx = X_extended.shape[0] - 1
        input_vec = text_vec.reshape(1, -1)

        results = {}
        for model_name, model in self._registry.nc_models.items():
            node_probs = self._nc_forward(model_name, model, X_extended, new_idx, input_vec)
            results[model_name] = self._format_nc_result(node_probs)

        return results

    def predict_lp_inductive(self, text_vec: np.ndarray, top_k: int = 10) -> Dict[str, List[dict]]:
        """Run LP inference for a new product against all existing nodes."""
        X_extended = np.vstack([self._registry.X_full, text_vec.reshape(1, -1)])
        new_idx = X_extended.shape[0] - 1
        thresholds = self._registry.lp_config.get("thresholds", {})

        results = {}
        for model_name, model in self._registry.lp_models.items():
            scores = self._lp_scores(model_name, model, X_extended, new_idx, text_vec)
            threshold = thresholds.get(model_name, 0.5)
            top_indices = top_k_indices(scores, top_k)

            suggestions = []
            for target_idx in top_indices:
                target_asin = str(self._registry.node_names[target_idx])
                score = float(scores[target_idx])
                suggestions.append({
                    "asin": target_asin,
                    "score": score,
                    "predicted_link": score >= threshold,
                    "target_class": self._registry.get_class_name(
                        int(self._registry.y_full[target_idx])
                    ),
                })
            results[model_name] = suggestions

        return results

    def predict_full(self, title: str, description: str, top_k: int = 10) -> dict:
        """Combined NC + LP inductive prediction."""
        text_vec = self.vectorize_text(title, description)
        warning = None
        if np.abs(text_vec).sum() < 1e-8:
            warning = ("Input text contains no recognized vocabulary words. "
                       "Predictions are unreliable (random).")
        nc_results = self.predict_nc_inductive(text_vec)
        lp_results = self.predict_lp_inductive(text_vec, top_k)
        result = {"nc_predictions": nc_results, "lp_predictions": lp_results}
        if warning:
            result["warning"] = warning
        return result

    # ── NC internals ──────────────────────────────────────────────────────────

    def _nc_forward(self, model_name: str, model, X_extended: np.ndarray,
                    new_idx: int, input_vec: np.ndarray) -> np.ndarray:
        """Run one NC model, returning (num_classes,) probability array."""

        if model_name == "MLP":
            # No graph needed — predict directly on new vector.
            preds = model.predict(input_vec, verbose=0)
            return preds[0]

        if model_name == "GraphSAGE":
            # SAGE uses dynamic tf.shape for segment counts → safe to extend.
            preds = model.predict(X_extended, batch_size=X_extended.shape[0], verbose=0)
            return preds[new_idx]

        # GCN / GAT use fixed graph_info.num_nodes → cannot extend safely.
        # Fall back to the MLP model; log so it's visible in server output.
        logger.debug(
            "NC inductive: %s uses fixed graph_info.num_nodes — using MLP approximation",
            model_name,
        )
        mlp = self._registry.nc_models.get("MLP")
        if mlp is not None:
            preds = mlp.predict(input_vec, verbose=0)
            return preds[0]

        # Last resort: uniform distribution
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

    def _lp_scores(self, model_name: str, model, X_extended: np.ndarray,
                   new_idx: int, text_vec: np.ndarray) -> np.ndarray:
        """Return (num_existing_nodes,) sigmoid-score array."""

        if model_name == "MLP":
            # Get all existing node embeddings (cached in model).
            existing_emb = model.get_embeddings(training=False).numpy()

            # Run new vector through MLP encoder to get its embedding.
            x = tf.constant(text_vec.reshape(1, -1), dtype=tf.float32)
            x = model.drop1(x, training=False)
            x = model.dense1(x)
            x = model.drop2(x, training=False)
            x = model.dense2(x)
            new_emb = tf.math.l2_normalize(x, axis=-1).numpy().flatten()

            # Dot product (embeddings already L2-normalised).
            scores = np.dot(existing_emb, new_emb)
            return 1.0 / (1.0 + np.exp(-scores))

        if model_name == "GraphSAGE":
            # Cache existing node embeddings (they don't change between calls)
            if self._sage_lp_embeddings is None:
                sage_model = self._registry.lp_models["GraphSAGE"]
                self._sage_lp_embeddings = sage_model.get_embeddings(training=False).numpy()

            # Encode the new node: pass X_extended through SAGE encoder.
            # The new node at index new_idx gets zero neighbour aggregation
            # (no edges point to it), which is the correct inductive result.
            all_emb = model.encoder(
                tf.constant(X_extended, dtype=tf.float32), training=False
            ).numpy()
            new_emb = all_emb[new_idx]

            scores = np.dot(self._sage_lp_embeddings, new_emb)
            return 1.0 / (1.0 + np.exp(-scores))

        # GCN / GAT use fixed N in their encoders → approximate with
        # cosine similarity between existing encoded embeddings.
        # Since we can't encode the new node, we use MLP LP if available,
        # otherwise fall back to raw-feature projection.
        logger.debug(
            "LP inductive: %s uses fixed N encoder — using MLP LP approximation",
            model_name,
        )
        mlp_lp = self._registry.lp_models.get("MLP")
        if mlp_lp is not None:
            # Use MLP encoder for the new node (no graph needed)
            x = tf.constant(text_vec.reshape(1, -1), dtype=tf.float32)
            x = mlp_lp.drop1(x, training=False)
            x = mlp_lp.dense1(x)
            x = mlp_lp.drop2(x, training=False)
            x = mlp_lp.dense2(x)
            new_emb = tf.math.l2_normalize(x, axis=-1).numpy().flatten()

            existing_emb = mlp_lp.get_embeddings(training=False).numpy()
            scores = np.dot(existing_emb, new_emb)
            return 1.0 / (1.0 + np.exp(-scores))

        # Last resort: use the model's own embeddings with a projected raw vector
        existing_emb = model.get_embeddings(training=False).numpy()
        emb_dim = existing_emb.shape[1]
        raw = text_vec.flatten()[:emb_dim]  # truncate to match dim
        new_emb = raw / (np.linalg.norm(raw) + 1e-9)

        scores = np.dot(existing_emb, new_emb)
        return 1.0 / (1.0 + np.exp(-scores))
