import logging
from typing import Dict, List

import numpy as np

from app.services.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class NCService:
    """Node Classification inference for existing graph nodes."""

    def __init__(self, registry: ModelRegistry):
        self._registry = registry
        self._predictions_cache: Dict[str, np.ndarray] = {}

    def _ensure_predictions(self, model_name: str) -> np.ndarray:
        """Run full-graph NC prediction and cache results."""
        if model_name in self._predictions_cache:
            return self._predictions_cache[model_name]

        model = self._registry.nc_models.get(model_name)
        if model is None:
            raise ValueError(f"NC model '{model_name}' not loaded")

        X = self._registry.X_full
        preds = model.predict(X, batch_size=X.shape[0], verbose=0)
        self._predictions_cache[model_name] = preds
        return preds

    def predict_for_node(self, asin: str) -> Dict[str, dict]:
        """Get NC predictions from all models for a single node."""
        idx = self._registry.asin_to_idx.get(asin)
        if idx is None:
            raise KeyError(f"ASIN '{asin}' not found in graph")

        results = {}
        for model_name in self._registry.nc_models:
            preds = self._ensure_predictions(model_name)
            node_probs = preds[idx]
            predicted_class_idx = int(np.argmax(node_probs))
            confidence = float(node_probs[predicted_class_idx])

            # Top 3
            top3_idx = np.argsort(node_probs)[::-1][:3]
            top3 = [
                {
                    "class": self._registry.get_class_name(int(i)),
                    "probability": float(node_probs[i]),
                }
                for i in top3_idx
            ]

            results[model_name] = {
                "predicted_class": self._registry.get_class_name(predicted_class_idx),
                "confidence": confidence,
                "top_3": top3,
            }

        return results

    def predict_batch(self, asins: List[str]) -> Dict[str, Dict[str, dict]]:
        """Get NC predictions for multiple nodes."""
        return {asin: self.predict_for_node(asin) for asin in asins}
