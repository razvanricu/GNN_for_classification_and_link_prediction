import logging
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import tensorflow as tf

from app.services.model_registry import ModelRegistry
from app.ml.custom_layers import (
    StrictGCNModel, MultiHeadStaticGAT, StaticGraphAttn, StaticGraphSAGE,
)

logger = logging.getLogger(__name__)

# GNNExplainer hyperparameters
DEFAULT_NUM_STEPS = 200
DEFAULT_LR = 0.01
LAMBDA_L1 = 0.01
LAMBDA_ENTROPY = 0.01
MASK_THRESHOLD = 0.1
MAX_EXPLANATION_EDGES = 50


class ExplainerService:
    """GNNExplainer (Ying et al., 2019) — optimization-based explanation."""

    def __init__(self, registry: ModelRegistry):
        self._registry = registry
        self._adj_list: Optional[dict] = None

    def _get_adj_list(self) -> dict:
        if self._adj_list is None:
            edge_index = self._registry.edge_index
            adj: Dict[int, List[int]] = {}
            for i in range(edge_index.shape[1]):
                src, dst = int(edge_index[0, i]), int(edge_index[1, i])
                adj.setdefault(src, []).append(dst)
            self._adj_list = adj
        return self._adj_list

    def _extract_k_hop_subgraph(
        self, node_idx: int, k: int = 2, max_nodes: int = 200
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], Set[int]]:
        adj = self._get_adj_list()
        visited = {node_idx}
        hop_distance = {node_idx: 0}
        queue = deque([(node_idx, 0)])

        while queue and len(visited) < max_nodes:
            node, depth = queue.popleft()
            if depth >= k:
                continue
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    hop_distance[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1))

        one_hop_nodes = {n for n, d in hop_distance.items() if d <= 1}

        sub_nodes = np.array(sorted(visited), dtype=np.int32)
        node_set = set(sub_nodes.tolist())
        node_map = {int(n): i for i, n in enumerate(sub_nodes)}

        edge_index = self._registry.edge_index
        edges_local_src = []
        edges_local_dst = []
        for i in range(edge_index.shape[1]):
            s, d = int(edge_index[0, i]), int(edge_index[1, i])
            if s in node_set and d in node_set:
                edges_local_src.append(node_map[s])
                edges_local_dst.append(node_map[d])

        if len(edges_local_src) == 0:
            sub_edges = np.zeros((2, 0), dtype=np.int32)
        else:
            sub_edges = np.array([edges_local_src, edges_local_dst], dtype=np.int32)

        return sub_nodes, sub_edges, node_map, one_hop_nodes

    # ── Masked forward passes ────────────────────────────────────────────────

    def _masked_forward_gcn(
        self, model, X_sub: tf.Tensor, sub_edge_index: np.ndarray,
        edge_mask: tf.Tensor, num_nodes: int
    ) -> tf.Tensor:
        gcn_layer = next(l for l in model.layers if isinstance(l, StrictGCNModel))
        gcn1 = gcn_layer.gcn1
        gcn2 = gcn_layer.gcn2
        ln1 = gcn_layer.ln1
        ln2 = gcn_layer.ln2
        dense = gcn_layer.dense

        def gcn_conv_masked(h, conv_layer, mask_with_loops):
            h = tf.matmul(h, conv_layer.weight)
            src = tf.constant(sub_edge_index[0], dtype=tf.int32)
            dst = tf.constant(sub_edge_index[1], dtype=tf.int32)
            self_loops = tf.range(num_nodes, dtype=tf.int32)
            src = tf.concat([src, self_loops], axis=0)
            dst = tf.concat([dst, self_loops], axis=0)

            ones = tf.ones_like(src, dtype=tf.float32)
            deg = tf.math.unsorted_segment_sum(ones, dst, num_segments=num_nodes)
            deg_inv_sqrt = tf.pow(deg, -0.5)
            deg_inv_sqrt = tf.where(tf.math.is_inf(deg_inv_sqrt), 0.0, deg_inv_sqrt)
            norm_src = tf.gather(deg_inv_sqrt, src)
            norm_dst = tf.gather(deg_inv_sqrt, dst)
            norm = norm_src * norm_dst

            h_source = tf.gather(h, src)
            h_normalized = h_source * tf.expand_dims(norm * mask_with_loops, -1)
            aggregated = tf.math.unsorted_segment_sum(h_normalized, dst, num_segments=num_nodes)

            if conv_layer.activation:
                aggregated = conv_layer.activation(aggregated)
            return aggregated

        self_loop_mask = tf.ones(num_nodes, dtype=tf.float32)
        mask_with_loops = tf.concat([edge_mask, self_loop_mask], axis=0)

        x = gcn_conv_masked(X_sub, gcn1, mask_with_loops)
        x = ln1(x, training=False)
        x = gcn_conv_masked(x, gcn2, mask_with_loops)
        x = ln2(x, training=False)
        return dense(x)

    def _masked_forward_gat(
        self, model, X_sub: tf.Tensor, sub_edge_index: np.ndarray,
        edge_mask: tf.Tensor, num_nodes: int
    ) -> tf.Tensor:
        gat1 = next(l for l in model.layers if isinstance(l, MultiHeadStaticGAT))
        gat2 = next(l for l in model.layers if isinstance(l, StaticGraphAttn))
        dense = model.layers[-1]

        src = tf.constant(sub_edge_index[0], dtype=tf.int32)
        dst = tf.constant(sub_edge_index[1], dtype=tf.int32)

        def attn_layer_masked(features, layer, mask):
            h = tf.matmul(features, layer.W)
            h_i = tf.gather(h, dst)
            h_j = tf.gather(h, src)
            e = tf.squeeze(tf.matmul(tf.concat([h_i, h_j], axis=-1), layer.a), axis=-1)
            e = layer.leaky(e)

            max_e = tf.math.unsorted_segment_max(e, dst, num_nodes)
            max_e_edges = tf.gather(max_e, dst)
            exp_e = tf.exp(e - max_e_edges)
            sum_exp = tf.math.unsorted_segment_sum(exp_e, dst, num_nodes)
            alpha = exp_e / (tf.gather(sum_exp, dst) + 1e-9)

            alpha_masked = alpha * mask
            messages = tf.expand_dims(alpha_masked, -1) * h_j
            return tf.math.unsorted_segment_sum(messages, dst, num_nodes)

        head_outputs = []
        for head in gat1.heads:
            head_outputs.append(attn_layer_masked(X_sub, head, edge_mask))
        x = tf.concat(head_outputs, axis=-1)

        x = attn_layer_masked(x, gat2, edge_mask)
        return dense(x)

    def _masked_forward_sage(
        self, model, X_sub: tf.Tensor, sub_edge_index: np.ndarray,
        edge_mask: tf.Tensor, num_nodes: int
    ) -> tf.Tensor:
        sage_layer = next(l for l in model.layers if isinstance(l, StaticGraphSAGE))
        dense = model.layers[-1]

        src = tf.constant(sub_edge_index[0], dtype=tf.int32)
        dst = tf.constant(sub_edge_index[1], dtype=tf.int32)

        def sage_conv_masked(features, conv, mask):
            neighbor_feats = tf.gather(features, src)
            weighted_feats = neighbor_feats * tf.expand_dims(mask, -1)
            agg_sum = tf.math.unsorted_segment_sum(weighted_feats, dst, num_segments=num_nodes)
            mask_sum = tf.math.unsorted_segment_sum(mask, dst, num_segments=num_nodes)
            mask_sum = tf.maximum(mask_sum, 1e-9)
            agg_mean = agg_sum / tf.expand_dims(mask_sum, -1)

            combined = tf.concat([features, agg_mean], axis=-1)
            out = conv.W(combined)
            if conv.activation:
                out = conv.activation(out)
            if conv.normalize:
                out = tf.math.l2_normalize(out, axis=-1)
            return out

        x = sage_conv_masked(X_sub, sage_layer.conv1, edge_mask)
        x = sage_conv_masked(x, sage_layer.conv2, edge_mask)
        return dense(x)

    # ── LP masked forward passes ─────────────────────────────────────────────

    def _masked_forward_sage_lp(
        self, model, X_sub: tf.Tensor, sub_edge_index: np.ndarray,
        edge_mask: tf.Tensor, num_nodes: int
    ) -> tf.Tensor:
        encoder = model.encoder
        src = tf.constant(sub_edge_index[0], dtype=tf.int32)
        dst = tf.constant(sub_edge_index[1], dtype=tf.int32)

        def sage_conv_masked(features, conv, mask):
            neighbor_feats = tf.gather(features, src)
            weighted_feats = neighbor_feats * tf.expand_dims(mask, -1)
            agg_sum = tf.math.unsorted_segment_sum(weighted_feats, dst, num_segments=num_nodes)
            mask_sum = tf.math.unsorted_segment_sum(mask, dst, num_segments=num_nodes)
            mask_sum = tf.maximum(mask_sum, 1e-9)
            agg_mean = agg_sum / tf.expand_dims(mask_sum, -1)

            combined = tf.concat([features, agg_mean], axis=-1)
            out = conv.W(combined)
            if conv.activation:
                out = conv.activation(out)
            if conv.normalize:
                out = tf.math.l2_normalize(out, axis=-1)
            return out

        x = sage_conv_masked(X_sub, encoder.conv1, edge_mask)
        x = sage_conv_masked(x, encoder.conv2, edge_mask)
        return tf.math.l2_normalize(x, axis=-1)

    def _masked_forward_gcn_lp(
        self, model, X_sub: tf.Tensor, sub_edge_index: np.ndarray,
        edge_mask: tf.Tensor, num_nodes: int
    ) -> tf.Tensor:
        conv1 = model.conv1
        conv2 = model.conv2

        def gcn_conv_masked(h, conv_layer, mask_with_loops):
            h = tf.matmul(h, conv_layer.weight)
            src = tf.constant(sub_edge_index[0], dtype=tf.int32)
            dst = tf.constant(sub_edge_index[1], dtype=tf.int32)
            self_loops = tf.range(num_nodes, dtype=tf.int32)
            src = tf.concat([src, self_loops], axis=0)
            dst = tf.concat([dst, self_loops], axis=0)

            ones = tf.ones_like(src, dtype=tf.float32)
            deg = tf.math.unsorted_segment_sum(ones, dst, num_segments=num_nodes)
            deg_inv_sqrt = tf.pow(deg, -0.5)
            deg_inv_sqrt = tf.where(tf.math.is_inf(deg_inv_sqrt), 0.0, deg_inv_sqrt)
            norm_src = tf.gather(deg_inv_sqrt, src)
            norm_dst = tf.gather(deg_inv_sqrt, dst)
            norm = norm_src * norm_dst

            h_source = tf.gather(h, src)
            h_normalized = h_source * tf.expand_dims(norm * mask_with_loops, -1)
            aggregated = tf.math.unsorted_segment_sum(h_normalized, dst, num_segments=num_nodes)

            if conv_layer.activation:
                aggregated = conv_layer.activation(aggregated)
            return aggregated

        self_loop_mask = tf.ones(num_nodes, dtype=tf.float32)
        mask_with_loops = tf.concat([edge_mask, self_loop_mask], axis=0)

        x = gcn_conv_masked(X_sub, conv1, mask_with_loops)
        x = gcn_conv_masked(x, conv2, mask_with_loops)
        return tf.math.l2_normalize(x, axis=-1)

    def _masked_forward_gat_lp(
        self, model, X_sub: tf.Tensor, sub_edge_index: np.ndarray,
        edge_mask: tf.Tensor, num_nodes: int
    ) -> tf.Tensor:
        gat1 = model.gat1
        gat2 = model.gat2

        src = tf.constant(sub_edge_index[0], dtype=tf.int32)
        dst = tf.constant(sub_edge_index[1], dtype=tf.int32)

        def attn_layer_masked(features, layer, mask):
            h = tf.matmul(features, layer.W)
            h_i = tf.gather(h, dst)
            h_j = tf.gather(h, src)
            e = tf.squeeze(tf.matmul(tf.concat([h_i, h_j], axis=-1), layer.a), axis=-1)
            e = layer.leaky(e)

            max_e = tf.math.unsorted_segment_max(e, dst, num_nodes)
            max_e_edges = tf.gather(max_e, dst)
            exp_e = tf.exp(e - max_e_edges)
            sum_exp = tf.math.unsorted_segment_sum(exp_e, dst, num_nodes)
            alpha = exp_e / (tf.gather(sum_exp, dst) + 1e-9)

            alpha_masked = alpha * mask
            messages = tf.expand_dims(alpha_masked, -1) * h_j
            return tf.math.unsorted_segment_sum(messages, dst, num_nodes)

        head_outputs = []
        for head in gat1.heads:
            head_outputs.append(attn_layer_masked(X_sub, head, edge_mask))
        x = tf.concat(head_outputs, axis=-1)

        x = attn_layer_masked(x, gat2, edge_mask)
        return tf.math.l2_normalize(x, axis=-1)

    # ── Main explanation methods ─────────────────────────────────────────────

    def explain_node(
        self, node_idx: int, model_name: str,
        num_steps: int = DEFAULT_NUM_STEPS, lr: float = DEFAULT_LR
    ) -> dict:
        model = self._registry.nc_models.get(model_name)
        if model is None:
            raise ValueError(f"NC model '{model_name}' not found")

        X_full = self._registry.X_full
        original_pred = model.predict(
            X_full, batch_size=X_full.shape[0], verbose=0
        )[node_idx]
        predicted_class = int(np.argmax(original_pred))

        sub_nodes, sub_edges, node_map, one_hop_nodes = self._extract_k_hop_subgraph(node_idx, k=2)
        local_target = node_map[node_idx]
        num_sub_nodes = len(sub_nodes)
        num_sub_edges = sub_edges.shape[1]

        if num_sub_edges == 0:
            return self._build_explanation_graph(
                sub_nodes, sub_edges, np.array([]), node_idx, node_map,
                one_hop_nodes, predicted_class, float(original_pred[predicted_class])
            )

        X_sub = tf.constant(self._registry.X_full[sub_nodes], dtype=tf.float32)
        mask_logits = tf.Variable(tf.zeros(num_sub_edges), trainable=True)
        optimizer = tf.optimizers.Adam(learning_rate=lr)

        dispatch = {
            "GCN": self._masked_forward_gcn,
            "GAT": self._masked_forward_gat,
            "GraphSAGE": self._masked_forward_sage,
        }
        forward_fn = dispatch[model_name]

        for _ in range(num_steps):
            with tf.GradientTape() as tape:
                edge_mask = tf.sigmoid(mask_logits)
                logits = forward_fn(model, X_sub, sub_edges, edge_mask, num_sub_nodes)
                pred_probs = logits[local_target]

                pred_loss = -tf.math.log(pred_probs[predicted_class] + 1e-9)
                l1_loss = LAMBDA_L1 * tf.reduce_sum(edge_mask)
                entropy = -edge_mask * tf.math.log(edge_mask + 1e-9) \
                          - (1.0 - edge_mask) * tf.math.log(1.0 - edge_mask + 1e-9)
                entropy_loss = LAMBDA_ENTROPY * tf.reduce_mean(entropy)
                total_loss = pred_loss + l1_loss + entropy_loss

            grads = tape.gradient(total_loss, [mask_logits])
            optimizer.apply_gradients(zip(grads, [mask_logits]))

        final_mask = tf.sigmoid(mask_logits).numpy()
        return self._build_explanation_graph(
            sub_nodes, sub_edges, final_mask, node_idx, node_map,
            one_hop_nodes, predicted_class, float(original_pred[predicted_class])
        )

    def explain_link(
        self, src_idx: int, dst_idx: int, model_name: str,
        num_steps: int = DEFAULT_NUM_STEPS, lr: float = DEFAULT_LR
    ) -> dict:
        model = self._registry.lp_models.get(model_name)
        if model is None:
            raise ValueError(f"LP model '{model_name}' not found")

        sub_nodes_src, _, _, one_hop_src = self._extract_k_hop_subgraph(src_idx, k=2)
        sub_nodes_dst, _, _, one_hop_dst = self._extract_k_hop_subgraph(dst_idx, k=2)
        all_nodes = set(sub_nodes_src.tolist()) | set(sub_nodes_dst.tolist())
        one_hop_nodes = one_hop_src | one_hop_dst

        sub_nodes = np.array(sorted(all_nodes), dtype=np.int32)
        node_set = set(sub_nodes.tolist())
        node_map = {int(n): i for i, n in enumerate(sub_nodes)}

        edge_index = self._registry.edge_index
        edges_local_src = []
        edges_local_dst = []
        for i in range(edge_index.shape[1]):
            s, d = int(edge_index[0, i]), int(edge_index[1, i])
            if s in node_set and d in node_set:
                edges_local_src.append(node_map[s])
                edges_local_dst.append(node_map[d])

        if len(edges_local_src) == 0:
            sub_edges = np.zeros((2, 0), dtype=np.int32)
        else:
            sub_edges = np.array([edges_local_src, edges_local_dst], dtype=np.int32)

        num_sub_nodes = len(sub_nodes)
        num_sub_edges = sub_edges.shape[1]
        local_src = node_map[src_idx]
        local_dst = node_map[dst_idx]

        if num_sub_edges == 0:
            return self._build_explanation_graph(
                sub_nodes, sub_edges, np.array([]), src_idx, node_map,
                one_hop_nodes, lp_score=0.0, center_idx_2=dst_idx
            )

        X_sub = tf.constant(self._registry.X_full[sub_nodes], dtype=tf.float32)
        mask_logits = tf.Variable(tf.zeros(num_sub_edges), trainable=True)
        optimizer = tf.optimizers.Adam(learning_rate=lr)

        dispatch = {
            "GraphSAGE": self._masked_forward_sage_lp,
            "GCN": self._masked_forward_gcn_lp,
            "GAT": self._masked_forward_gat_lp,
        }
        forward_fn = dispatch[model_name]

        for _ in range(num_steps):
            with tf.GradientTape() as tape:
                edge_mask = tf.sigmoid(mask_logits)
                embeddings = forward_fn(model, X_sub, sub_edges, edge_mask, num_sub_nodes)

                src_emb = embeddings[local_src]
                dst_emb = embeddings[local_dst]
                score = tf.nn.sigmoid(tf.reduce_sum(src_emb * dst_emb))

                pred_loss = -tf.math.log(score + 1e-9)
                l1_loss = LAMBDA_L1 * tf.reduce_sum(edge_mask)
                entropy = -edge_mask * tf.math.log(edge_mask + 1e-9) \
                          - (1.0 - edge_mask) * tf.math.log(1.0 - edge_mask + 1e-9)
                entropy_loss = LAMBDA_ENTROPY * tf.reduce_mean(entropy)
                total_loss = pred_loss + l1_loss + entropy_loss

            grads = tape.gradient(total_loss, [mask_logits])
            optimizer.apply_gradients(zip(grads, [mask_logits]))

        final_mask = tf.sigmoid(mask_logits).numpy()
        lp_score = float(tf.nn.sigmoid(
            tf.reduce_sum(
                forward_fn(model, X_sub, sub_edges, tf.ones(num_sub_edges), num_sub_nodes)[local_src]
                * forward_fn(model, X_sub, sub_edges, tf.ones(num_sub_edges), num_sub_nodes)[local_dst]
            )
        ).numpy())

        return self._build_explanation_graph(
            sub_nodes, sub_edges, final_mask, src_idx, node_map,
            one_hop_nodes, lp_score=lp_score, center_idx_2=dst_idx
        )

    # ── Inductive explanation ───────────────────────────────────────────────

    def explain_node_inductive(
        self, text_vec: np.ndarray, model_name: str,
        num_steps: int = DEFAULT_NUM_STEPS, lr: float = DEFAULT_LR
    ) -> dict:
        """GNNExplainer for a new (inductive) node using pseudo-neighbor injection."""
        from app.deps import get_inductive_service
        inductive = get_inductive_service()

        model = self._registry.nc_models.get(model_name)
        if model is None:
            raise ValueError(f"NC model '{model_name}' not found")

        K_NEIGHBORS = 10
        neighbor_indices, _, _, _ = inductive._find_pseudo_neighbors(text_vec, k=K_NEIGHBORS)

        X_extended = np.vstack([self._registry.X_full, text_vec.reshape(1, -1)])
        new_idx = X_extended.shape[0] - 1
        extended_edges = inductive._build_extended_edges(neighbor_indices, new_idx)

        original_pred = inductive._nc_forward(
            model_name, model, X_extended, new_idx, text_vec, extended_edges
        )
        predicted_class = int(np.argmax(original_pred))
        confidence = float(original_pred[predicted_class])

        # Build subgraph: new node + neighbors + their 1-hop neighborhoods
        adj = self._get_adj_list()
        sub_node_set = {new_idx}
        one_hop_of_new = set(neighbor_indices.tolist())
        sub_node_set.update(one_hop_of_new)

        for n_idx in neighbor_indices:
            for neighbor in adj.get(int(n_idx), [])[:20]:
                sub_node_set.add(neighbor)
                if len(sub_node_set) >= 200:
                    break
            if len(sub_node_set) >= 200:
                break

        sub_nodes = np.array(sorted(sub_node_set), dtype=np.int32)
        node_map = {int(n): i for i, n in enumerate(sub_nodes)}

        # Build local edges: original edges within subgraph + synthetic edges to new node
        edge_index = self._registry.edge_index
        edges_local_src = []
        edges_local_dst = []
        for i in range(edge_index.shape[1]):
            s, d = int(edge_index[0, i]), int(edge_index[1, i])
            if s in node_map and d in node_map:
                edges_local_src.append(node_map[s])
                edges_local_dst.append(node_map[d])

        # Add synthetic edges: new_node <-> pseudo-neighbors
        local_new = node_map[new_idx]
        for n_idx in neighbor_indices:
            if int(n_idx) in node_map:
                local_n = node_map[int(n_idx)]
                edges_local_src.append(local_new)
                edges_local_dst.append(local_n)
                edges_local_src.append(local_n)
                edges_local_dst.append(local_new)

        if len(edges_local_src) == 0:
            return self._build_explanation_graph_inductive(
                sub_nodes, np.zeros((2, 0), dtype=np.int32), np.array([]),
                new_idx, node_map, one_hop_of_new | {new_idx},
                predicted_class, confidence, text_vec
            )

        sub_edges = np.array([edges_local_src, edges_local_dst], dtype=np.int32)
        num_sub_nodes = len(sub_nodes)
        num_sub_edges = sub_edges.shape[1]

        # Feature matrix for subgraph (use extended features for new node)
        X_sub = tf.constant(X_extended[sub_nodes], dtype=tf.float32)

        mask_logits = tf.Variable(tf.zeros(num_sub_edges), trainable=True)
        optimizer = tf.optimizers.Adam(learning_rate=lr)

        dispatch = {
            "GCN": self._masked_forward_gcn,
            "GAT": self._masked_forward_gat,
            "GraphSAGE": self._masked_forward_sage,
        }
        forward_fn = dispatch[model_name]

        for _ in range(num_steps):
            with tf.GradientTape() as tape:
                edge_mask = tf.sigmoid(mask_logits)
                logits = forward_fn(model, X_sub, sub_edges, edge_mask, num_sub_nodes)
                pred_probs = logits[local_new]

                pred_loss = -tf.math.log(pred_probs[predicted_class] + 1e-9)
                l1_loss = LAMBDA_L1 * tf.reduce_sum(edge_mask)
                entropy = -edge_mask * tf.math.log(edge_mask + 1e-9) \
                          - (1.0 - edge_mask) * tf.math.log(1.0 - edge_mask + 1e-9)
                entropy_loss = LAMBDA_ENTROPY * tf.reduce_mean(entropy)
                total_loss = pred_loss + l1_loss + entropy_loss

            grads = tape.gradient(total_loss, [mask_logits])
            optimizer.apply_gradients(zip(grads, [mask_logits]))

        final_mask = tf.sigmoid(mask_logits).numpy()
        return self._build_explanation_graph_inductive(
            sub_nodes, sub_edges, final_mask, new_idx, node_map,
            one_hop_of_new | {new_idx}, predicted_class, confidence, text_vec
        )

    def _build_explanation_graph_inductive(
        self, sub_nodes: np.ndarray, sub_edges: np.ndarray,
        mask_values: np.ndarray, new_node_idx: int,
        node_map: Dict[int, int], one_hop_nodes: Set[int],
        predicted_class: int, confidence: float,
        text_vec: np.ndarray,
    ) -> dict:
        """Build explanation graph with the synthetic new node included."""
        edges = []
        seen_edges = set()
        edge_candidates = []
        for i in range(len(mask_values)):
            val = float(mask_values[i])
            if val < MASK_THRESHOLD:
                continue
            src_global = int(sub_nodes[sub_edges[0, i]])
            dst_global = int(sub_nodes[sub_edges[1, i]])
            if src_global not in one_hop_nodes and dst_global not in one_hop_nodes:
                continue
            edge_key = (min(src_global, dst_global), max(src_global, dst_global))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edge_candidates.append((val, src_global, dst_global))

        edge_candidates.sort(key=lambda x: x[0], reverse=True)
        final_edges = edge_candidates[:MAX_EXPLANATION_EDGES]

        relevant_nodes = {new_node_idx}
        node_importance = {}
        for val, src_global, dst_global in final_edges:
            relevant_nodes.add(src_global)
            relevant_nodes.add(dst_global)
            node_importance[src_global] = max(node_importance.get(src_global, 0.0), val)
            node_importance[dst_global] = max(node_importance.get(dst_global, 0.0), val)

        nodes = []
        for global_idx in sub_nodes:
            global_idx = int(global_idx)
            if global_idx not in relevant_nodes:
                continue
            if global_idx == new_node_idx:
                node_data = {
                    "id": "__new__",
                    "label": "New Product",
                    "class": self._registry.get_class_name(predicted_class),
                    "is_center": True,
                    "is_new": True,
                }
            else:
                asin = str(self._registry.node_names[global_idx])
                class_name = self._registry.get_class_name(int(self._registry.y_full[global_idx]))
                node_data = {
                    "id": asin,
                    "label": asin,
                    "class": class_name,
                    "is_center": False,
                }
            if global_idx in node_importance:
                node_data["importance"] = round(node_importance[global_idx], 4)
            nodes.append({"data": node_data})

        for val, src_global, dst_global in final_edges:
            src_id = "__new__" if src_global == new_node_idx else str(self._registry.node_names[src_global])
            dst_id = "__new__" if dst_global == new_node_idx else str(self._registry.node_names[dst_global])
            edges.append({
                "data": {
                    "source": src_id,
                    "target": dst_id,
                    "type": "explained",
                    "importance": round(val, 4),
                }
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "predicted_class_idx": predicted_class,
            "predicted_class": self._registry.get_class_name(predicted_class),
            "confidence": round(confidence, 4),
        }

    # ── Build response ───────────────────────────────────────────────────────

    def _build_explanation_graph(
        self, sub_nodes: np.ndarray, sub_edges: np.ndarray,
        mask_values: np.ndarray, center_idx: int,
        node_map: Dict[int, int],
        one_hop_nodes: Set[int],
        predicted_class: Optional[int] = None,
        confidence: Optional[float] = None,
        lp_score: Optional[float] = None,
        center_idx_2: Optional[int] = None,
    ) -> dict:
        edges = []
        seen_edges = set()
        edge_candidates = []
        for i in range(len(mask_values)):
            val = float(mask_values[i])
            if val < MASK_THRESHOLD:
                continue
            src_global = int(sub_nodes[sub_edges[0, i]])
            dst_global = int(sub_nodes[sub_edges[1, i]])
            if src_global not in one_hop_nodes and dst_global not in one_hop_nodes:
                continue
            edge_key = (min(src_global, dst_global), max(src_global, dst_global))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edge_candidates.append((val, src_global, dst_global))

        edge_candidates.sort(key=lambda x: x[0], reverse=True)
        final_edges = edge_candidates[:MAX_EXPLANATION_EDGES]

        relevant_nodes = {center_idx}
        if center_idx_2 is not None:
            relevant_nodes.add(center_idx_2)
        node_importance = {}
        for val, src_global, dst_global in final_edges:
            relevant_nodes.add(src_global)
            relevant_nodes.add(dst_global)
            node_importance[src_global] = max(node_importance.get(src_global, 0.0), val)
            node_importance[dst_global] = max(node_importance.get(dst_global, 0.0), val)

        nodes = []
        for global_idx in sub_nodes:
            global_idx = int(global_idx)
            if global_idx not in relevant_nodes:
                continue
            asin = str(self._registry.node_names[global_idx])
            class_name = self._registry.get_class_name(int(self._registry.y_full[global_idx]))
            node_data = {
                "id": asin,
                "label": asin,
                "class": class_name,
                "is_center": global_idx == center_idx or global_idx == center_idx_2,
            }
            if global_idx in node_importance:
                node_data["importance"] = round(node_importance[global_idx], 4)
            nodes.append({"data": node_data})

        for val, src_global, dst_global in final_edges:
            src_asin = str(self._registry.node_names[src_global])
            dst_asin = str(self._registry.node_names[dst_global])
            edges.append({
                "data": {
                    "source": src_asin,
                    "target": dst_asin,
                    "type": "explained",
                    "importance": round(val, 4),
                }
            })

        result = {"nodes": nodes, "edges": edges}
        if predicted_class is not None:
            result["predicted_class_idx"] = predicted_class
            result["predicted_class"] = self._registry.get_class_name(predicted_class)
            result["confidence"] = round(confidence, 4) if confidence else None
        if lp_score is not None:
            result["lp_score"] = round(lp_score, 4)
        return result
