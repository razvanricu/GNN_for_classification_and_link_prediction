import logging
import pickle
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from app.config import get_settings
from app.ml.custom_layers import (
    StrictGCNConv, StrictGCNModel,
    StaticGraphAttn, MultiHeadStaticGAT,
    SAGEConvLayer, StaticGraphSAGE,
)
from app.ml.lp_models import (
    GraphSAGELinkPredictor, MLPLinkPredictor,
    GCNLinkPredictor, GATLinkPredictor,
)
from app.ml.focal_loss import focal_loss_custom
from app.ml.graph_info import GraphInfo

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Reconstructs all ML models from saved weights + lp_config.json."""

    def __init__(self):
        self._settings = get_settings()
        self._models_dir = Path(self._settings.MODELS_DIR)
        self._loaded = False

        # Graph data
        self.X_full: Optional[np.ndarray] = None
        self.y_full: Optional[np.ndarray] = None
        self.edge_index: Optional[np.ndarray] = None
        self.train_mask: Optional[np.ndarray] = None
        self.val_mask: Optional[np.ndarray] = None
        self.test_mask: Optional[np.ndarray] = None
        self.node_names: Optional[np.ndarray] = None
        self.class_names: Optional[list] = None
        self.asin_to_idx: Optional[Dict[str, int]] = None
        self.vectorizer = None
        self._vectorizer_type: str = "tfidf"  # "tfidf" or "sbert"
        self.lp_config: Optional[dict] = None

        # Graph objects used by layers
        self.graph_info: Optional[GraphInfo] = None
        self.edge_index_tf = None

        # Models
        self.nc_models: Dict[str, keras.Model] = {}
        self.lp_models: Dict[str, keras.Model] = {}

    # ── public ────────────────────────────────────────────────────────────────

    def warmup(self):
        if self._loaded:
            return

        logger.info("Loading graph data...")
        self._load_graph_data()

        logger.info("Loading TF-IDF vectorizer...")
        self._load_vectorizer()

        logger.info("Loading LP config...")
        self._load_lp_config()

        logger.info("Building and loading NC models...")
        self._build_nc_models()

        logger.info("Building and loading LP models...")
        self._build_lp_models()

        logger.info("Running warmup predictions...")
        self._warmup_predictions()

        self._loaded = True
        logger.info(
            "Registry ready — NC: %s  LP: %s",
            list(self.nc_models.keys()),
            list(self.lp_models.keys()),
        )

    @property
    def num_nodes(self) -> int:
        return self.X_full.shape[0]

    @property
    def num_features(self) -> int:
        return self.X_full.shape[1]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def get_class_name(self, idx: int) -> str:
        return self.class_names[idx]

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_graph_data(self):
        path = self._models_dir / "graph_data.npz"
        with np.load(path, allow_pickle=True) as data:
            self.X_full     = data["X_full"].astype(np.float32)
            self.y_full     = data["y_full"].astype(np.int32)
            self.edge_index = data["edge_index"]
            self.train_mask = data["train_mask"]
            self.val_mask   = data["val_mask"]
            self.test_mask  = data["test_mask"]
            self.node_names = data["node_names"]
            self.class_names = data["class_names"].tolist()

        self.asin_to_idx = {
            asin: idx for idx, asin in enumerate(self.node_names)
        }

        # Edge index with self-loops (used by GCN layers)
        num_nodes = self.X_full.shape[0]
        self_loops = np.vstack((np.arange(num_nodes), np.arange(num_nodes)))
        edges_with_loops = np.hstack((self.edge_index, self_loops))
        self.edge_index_tf = tf.convert_to_tensor(edges_with_loops, dtype=tf.int32)

        # Edge index without self-loops (used by GAT / SAGE LP layers)
        self.edge_index_no_loops_tf = tf.convert_to_tensor(
            self.edge_index, dtype=tf.int32
        )

        self.graph_info = GraphInfo(
            edges=self.edge_index_tf, num_nodes=num_nodes
        )

        logger.info(
            "  Graph: %d nodes, %d edges, %d classes, %d features",
            num_nodes, self.edge_index.shape[1],
            len(self.class_names), self.X_full.shape[1],
        )

    def _load_vectorizer(self):
        # Try SBERT config first, fall back to TF-IDF pickle
        sbert_path = self._models_dir / "encoder_config.pkl"
        tfidf_path = self._models_dir / "tfidf_vectorizer.pkl"

        if sbert_path.exists():
            with open(sbert_path, "rb") as f:
                cfg = pickle.load(f)
            if isinstance(cfg, dict) and cfg.get("type") == "sbert":
                from sentence_transformers import SentenceTransformer
                self.vectorizer = SentenceTransformer(
                    cfg["model_name"], local_files_only=True
                )
                self._vectorizer_type = "sbert"
                logger.info("  Loaded SBERT encoder: %s", cfg["model_name"])
                return

        # Legacy TF-IDF
        if tfidf_path.exists():
            with open(tfidf_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            self._vectorizer_type = "tfidf"
            logger.info("  Loaded TF-IDF vectorizer")

    def _load_lp_config(self):
        with open(self._models_dir / "lp_config.json", "r") as f:
            self.lp_config = json.load(f)

    # ── model construction ────────────────────────────────────────────────────

    def _build_nc_models(self):
        hp     = self.lp_config["hp"]
        n_feat = self.num_features
        n_cls  = self.num_classes
        loss   = focal_loss_custom(gamma=2.0)

        builders = {
            "GCN":        lambda: self._build_gcn_nc(hp["gcn_nc"], n_feat, n_cls, loss),
            "GAT":        lambda: self._build_gat_nc(hp["gat_nc"], n_feat, n_cls, loss),
            "GraphSAGE":  lambda: self._build_sage_nc(hp["sage_nc"], n_feat, n_cls, loss),
            "MLP":        lambda: self._build_mlp_nc(hp["mlp_nc"], n_feat, n_cls, loss),
        }

        weight_files = {
            "GCN":       "gcn_nc_weights.weights.h5",
            "GAT":       "gat_nc_weights.weights.h5",
            "GraphSAGE": "sage_nc_weights.weights.h5",
            "MLP":       "mlp_nc_weights.weights.h5",
        }

        for name, build_fn in builders.items():
            self._load_model(name, build_fn, weight_files[name], self.nc_models)

    def _build_lp_models(self):
        hp = self.lp_config["hp"]

        builders = {
            "GraphSAGE": lambda: self._build_sage_lp(hp["sage_lp"]),
            "MLP":       lambda: self._build_mlp_lp(hp["mlp_lp"]),
            "GCN":       lambda: self._build_gcn_lp(hp["gcn_lp"]),
            "GAT":       lambda: self._build_gat_lp(hp["gat_lp"]),
        }

        weight_files = {
            "GraphSAGE": "sage_lp_weights.weights.h5",
            "MLP":       "mlp_lp_weights.weights.h5",
            "GCN":       "gcn_lp_weights.weights.h5",
            "GAT":       "gat_lp_weights.weights.h5",
        }

        # LP models are subclassed keras.Model — need a dummy forward pass to build
        dummy_edges = tf.constant([[0, 1]], dtype=tf.int32)

        for name, build_fn in builders.items():
            self._load_model(name, build_fn, weight_files[name], self.lp_models,
                            dummy_input=dummy_edges)

    def _load_model(self, name: str, build_fn, weight_file: str, target: dict,
                    dummy_input=None):
        path = self._models_dir / weight_file
        if not path.exists():
            logger.warning("Weight file not found: %s — skipping %s", path, name)
            return
        try:
            model = build_fn()
            # Subclassed models need a forward pass to build layers before loading weights
            if dummy_input is not None:
                model(dummy_input, training=False)
            model.load_weights(str(path))
            target[name] = model
            logger.info("  Loaded: %s", name)
        except Exception as e:
            logger.error("  Failed to load %s: %s", name, e)

    # ── NC architecture builders ──────────────────────────────────────────────

    def _build_gcn_nc(self, hp: dict, n_feat: int, n_cls: int, loss) -> keras.Model:
        out_feat = hp["out_feat"]
        dropout  = hp["dropout"]
        inputs   = keras.Input((n_feat,))
        outputs  = StrictGCNModel(
            n_feat, out_feat, n_cls, self.graph_info, dropout
        )(inputs)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=loss,
            weighted_metrics=["accuracy"],
        )
        return model

    def _build_gat_nc(self, hp: dict, n_feat: int, n_cls: int, loss) -> keras.Model:
        num_heads = hp["num_heads"]
        out_feat  = hp["out_feat"]
        dropout   = hp["dropout"]
        ei        = self.edge_index_no_loops_tf

        model = keras.Sequential([
            layers.InputLayer(input_shape=(n_feat,)),
            MultiHeadStaticGAT(num_heads, n_feat, out_feat, ei, dropout, activation="relu"),
            StaticGraphAttn(num_heads * out_feat, out_feat, ei, dropout, activation=None),
            layers.Dense(n_cls, activation="softmax"),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=loss,
            weighted_metrics=["accuracy"],
        )
        return model

    def _build_sage_nc(self, hp: dict, n_feat: int, n_cls: int, loss) -> keras.Model:
        out_feat  = hp["out_feat"]
        dropout   = hp["dropout"]
        normalize = hp.get("normalize", True)
        ei        = self.edge_index_no_loops_tf

        model = keras.Sequential([
            layers.InputLayer(input_shape=(n_feat,)),
            StaticGraphSAGE(out_feat, ei, dropout, normalize),
            layers.Dense(n_cls, activation="softmax"),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=loss,
            weighted_metrics=["accuracy"],
        )
        return model

    def _build_mlp_nc(self, hp: dict, n_feat: int, n_cls: int, loss) -> keras.Model:
        model = keras.Sequential([
            layers.InputLayer(input_shape=(n_feat,)),
            layers.Dense(hp["units_1"], activation="relu"),
            layers.Dropout(hp["dropout_1"]),
            layers.Dense(hp["units_2"], activation="relu"),
            layers.Dropout(hp["dropout_2"]),
            layers.Dense(n_cls, activation="softmax"),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=loss,
            weighted_metrics=["accuracy"],
        )
        return model

    # ── LP architecture builders ──────────────────────────────────────────────

    def _build_sage_lp(self, hp: dict) -> GraphSAGELinkPredictor:
        return GraphSAGELinkPredictor(
            out_feat       = hp["out_feat"],
            node_features  = self.X_full,
            edge_index_msg = self.edge_index_no_loops_tf,
            dropout_rate   = hp["dropout"],
            normalize      = True,
        )

    def _build_mlp_lp(self, hp: dict) -> MLPLinkPredictor:
        return MLPLinkPredictor(
            node_features = self.X_full,
            hidden_dim    = hp["hidden_dim"],
            dropout       = hp["dropout"],
        )

    def _build_gcn_lp(self, hp: dict) -> GCNLinkPredictor:
        return GCNLinkPredictor(
            in_feat      = self.num_features,
            out_feat     = hp["out_feat"],
            node_features= self.X_full,
            graph_info   = self.graph_info,
            dropout_rate = hp["dropout"],
        )

    def _build_gat_lp(self, hp: dict) -> GATLinkPredictor:
        return GATLinkPredictor(
            num_heads      = hp["num_heads"],
            out_feat       = hp["out_feat"],
            node_features  = self.X_full,
            edge_index_msg = self.edge_index_no_loops_tf,
            dropout_rate   = hp["dropout"],
        )

    # ── warmup ────────────────────────────────────────────────────────────────

    def _warmup_predictions(self):
        """Single dummy pass per model to compile TF graphs."""
        for name, model in self.nc_models.items():
            try:
                if name == "MLP":
                    # MLP has no graph dependency — 1 row suffices
                    model.predict(self.X_full[:1], batch_size=1, verbose=0)
                else:
                    # GNN models reference the full edge_index → need all N rows
                    model.predict(self.X_full, batch_size=self.X_full.shape[0], verbose=0)
                logger.info("  Warmed up NC: %s", name)
            except Exception as e:
                logger.warning("  Warmup failed NC %s: %s", name, e)

        dummy_edges = tf.constant([[0, 1]], dtype=tf.int32)
        for name, model in self.lp_models.items():
            try:
                model(dummy_edges, training=False)
                logger.info("  Warmed up LP: %s", name)
            except Exception as e:
                logger.warning("  Warmup failed LP %s: %s", name, e)
