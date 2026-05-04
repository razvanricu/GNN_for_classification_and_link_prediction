import logging
import json
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Loads and serves precomputed dataset analytics."""

    def __init__(self, precomputed_dir: str):
        self._dir = Path(precomputed_dir)
        self._class_distribution = None
        self._degree_distribution = None
        self._tsne_coords = None
        self._feature_importance = None
        self._model_comparison = None
        self._loaded = False

    def load(self):
        """Load all precomputed files."""
        if self._loaded:
            return

        self._load_json("class_distribution.json", "_class_distribution")
        self._load_json("feature_importance.json", "_feature_importance")
        self._load_json("model_comparison.json", "_model_comparison")
        self._load_npz("degree_distribution.npz", "_degree_distribution")
        self._load_npz("tsne_coords.npz", "_tsne_coords")

        self._loaded = True
        logger.info("Analytics service loaded.")

    def _load_json(self, filename: str, attr: str):
        path = self._dir / filename
        if path.exists():
            with open(path, "r") as f:
                setattr(self, attr, json.load(f))
        else:
            logger.warning(f"Analytics file not found: {path}")

    def _load_npz(self, filename: str, attr: str):
        path = self._dir / filename
        if path.exists():
            data = dict(np.load(path, allow_pickle=True))
            # Convert numpy arrays to lists for JSON serialization
            setattr(self, attr, {k: v.tolist() for k, v in data.items()})
        else:
            logger.warning(f"Analytics file not found: {path}")

    def get_class_distribution(self):
        return self._class_distribution

    def get_degree_distribution(self):
        return self._degree_distribution

    def get_tsne(self, sample: int = 2000):
        if not self._tsne_coords:
            return None
        # Return up to 'sample' points — only subsample per-point arrays
        coords = self._tsne_coords
        num_points = len(coords.get("x", []))
        if num_points > sample:
            indices = list(range(sample))
            result = {}
            for k, v in coords.items():
                if len(v) == num_points:
                    result[k] = [v[i] for i in indices]
                else:
                    result[k] = v  # metadata arrays (e.g. class_names) stay as-is
            return result
        return coords

    def get_feature_importance(self):
        return self._feature_importance

    def get_model_comparison(self):
        return self._model_comparison

    def get_graph_stats(self) -> Optional[dict]:
        """Compute basic graph stats from model_comparison or return cached."""
        if self._model_comparison and "graph_stats" in self._model_comparison:
            return self._model_comparison["graph_stats"]
        return None
