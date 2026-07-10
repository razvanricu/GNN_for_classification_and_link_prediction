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
LAMBDA_L1 = 0.005
LAMBDA_ENTROPY = 1.0
MASK_THRESHOLD = 0.1
MAX_EXPLANATION_EDGES = 50


class ExplainerService:
    """GNNExplainer - optimization-based explanation."""

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
                if len(visited) >= max_nodes:
                    break
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
            src = tf.concat([src, self_loops, self_loops], axis=0)
            dst = tf.concat([dst, self_loops, self_loops], axis=0)
            
            # Compute degree normalization for GCN aggregation
            ones = tf.ones_like(src, dtype=tf.float32)
            deg = tf.math.unsorted_segment_sum(ones, dst, num_segments=num_nodes)
            deg_inv_sqrt = tf.pow(deg, -0.5)
            deg_inv_sqrt = tf.where(tf.math.is_inf(deg_inv_sqrt), 0.0, deg_inv_sqrt)
            norm_src = tf.gather(deg_inv_sqrt, src)
            norm_dst = tf.gather(deg_inv_sqrt, dst)
            norm = norm_src * norm_dst

            # Apply the edge mask to the normalized messages
            h_source = tf.gather(h, src)
            h_normalized = h_source * tf.expand_dims(norm * mask_with_loops, -1)
            aggregated = tf.math.unsorted_segment_sum(h_normalized, dst, num_segments=num_nodes)

            if conv_layer.activation:
                aggregated = conv_layer.activation(aggregated)
            return aggregated

        # Add self-loops to the edge mask for GCN layers
        self_loop_mask = tf.ones(num_nodes * 2, dtype=tf.float32)
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

        sl = tf.range(num_nodes, dtype=tf.int32)
        src = tf.concat([tf.constant(sub_edge_index[0], dtype=tf.int32), sl], axis=0)
        dst = tf.concat([tf.constant(sub_edge_index[1], dtype=tf.int32), sl], axis=0)
        self_loop_mask = tf.ones(num_nodes, dtype=tf.float32)
        edge_mask_with_loops = tf.concat([edge_mask, self_loop_mask], axis=0)

        def attn_layer_masked(features, layer, mask):
            h = tf.matmul(features, layer.W)
            h_i = tf.gather(h, dst)
            h_j = tf.gather(h, src)
            # Compute attention coefficients
            e = tf.squeeze(tf.matmul(tf.concat([h_i, h_j], axis=-1), layer.a), axis=-1)
            e = layer.leaky(e)

            # Apply masked softmax to compute attention weights
            max_e = tf.math.unsorted_segment_max(e, dst, num_nodes)
            max_e_edges = tf.gather(max_e, dst)
            exp_e = tf.exp(e - max_e_edges)
            sum_exp = tf.math.unsorted_segment_sum(exp_e, dst, num_nodes)
            alpha = exp_e / (tf.gather(sum_exp, dst) + 1e-9)

            # Apply the edge mask to the attention weights
            alpha_masked = alpha * mask
            messages = tf.expand_dims(alpha_masked, -1) * h_j
            out = tf.math.unsorted_segment_sum(messages, dst, num_nodes)
            if layer.activation:
                out = layer.activation(out)
            return out

        head_outputs = []
        for head in gat1.heads:
            head_outputs.append(attn_layer_masked(X_sub, head, edge_mask_with_loops))
        x = tf.concat(head_outputs, axis=-1)

        x = attn_layer_masked(x, gat2, edge_mask_with_loops)
        return dense(x)

    def _masked_forward_sage(
        self, model, X_sub: tf.Tensor, sub_edge_index: np.ndarray,
        edge_mask: tf.Tensor, num_nodes: int
    ) -> tf.Tensor:
        sage_layer = next(l for l in model.layers if isinstance(l, StaticGraphSAGE))
        dense = model.layers[-1]

        src = tf.constant(sub_edge_index[0], dtype=tf.int32)
        dst = tf.constant(sub_edge_index[1], dtype=tf.int32)

        degree = tf.math.unsorted_segment_sum(
            tf.ones_like(src, dtype=tf.float32), dst, num_segments=num_nodes
        )
        degree = tf.maximum(degree, 1.0)

        def sage_conv_masked(features, conv, mask):
            # Aggregate neighbor features with masking
            neighbor_feats = tf.gather(features, src)
            weighted_feats = neighbor_feats * tf.expand_dims(mask, -1)
            agg_sum = tf.math.unsorted_segment_sum(weighted_feats, dst, num_segments=num_nodes)
            agg_mean = agg_sum / tf.expand_dims(degree, -1)

            # Combine with self features and apply linear transformation
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
   

    def explain_node(
        self, node_idx: int, model_name: str,
        num_steps: int = DEFAULT_NUM_STEPS, lr: float = DEFAULT_LR
    ) -> dict:
        model = self._registry.nc_models.get(model_name)
        if model is None:
            raise ValueError(f"NC model '{model_name}' not found")

        # Get the full model prediction for the node to determine the target class for explanation
        X_full = self._registry.X_full
        full_pred = model.predict(
            X_full, batch_size=X_full.shape[0], verbose=0
        )[node_idx]
        full_class = int(np.argmax(full_pred))

        # Extract the 2-hop subgraph around the node
        sub_nodes, sub_edges, node_map, one_hop_nodes = self._extract_k_hop_subgraph(node_idx, k=2)
        local_target = node_map[node_idx]
        num_sub_nodes = len(sub_nodes)
        num_sub_edges = sub_edges.shape[1]

        # If there are no edges in the subgraph, we cannot optimize a mask, so we return the explanation graph with an empty mask.
        if num_sub_edges == 0:
            return self._build_explanation_graph(
                sub_nodes, sub_edges, np.array([]), node_idx, node_map,
                one_hop_nodes, full_class, float(full_pred[full_class])
            )

        X_sub = tf.constant(self._registry.X_full[sub_nodes], dtype=tf.float32)

        # Select the appropriate forward function based on the model type
        dispatch = {
            "GCN": self._masked_forward_gcn,
            "GAT": self._masked_forward_gat,
            "GraphSAGE": self._masked_forward_sage,
        }
        forward_fn = dispatch[model_name]

        # Verify the local subgraph preserves the model's real prediction.
        # If not, the subgraph is too restrictive and the explanation would target a class the full model does not actually predict.
        local_pred = forward_fn(
            model, X_sub, sub_edges, tf.ones(num_sub_edges), num_sub_nodes
        )[local_target].numpy()
        local_class = int(np.argmax(local_pred))
        if local_class != full_class:
            raise RuntimeError(
                "Local explanation subgraph does not preserve the original "
                f"prediction: full_class={full_class}, local_class={local_class}. "
                "Explanation cannot be produced for this node."
            )
        predicted_class = full_class
        confidence = float(full_pred[full_class])
        # Initialize the edge mask logits for optimization
        mask_logits = tf.Variable(tf.zeros(num_sub_edges), trainable=True)
        optimizer = tf.optimizers.Adam(learning_rate=lr)

        # Optimize the edge mask to maximize the prediction probability of the target class while regularizing for sparsity and entropy
        for _ in range(num_steps):
            with tf.GradientTape() as tape:
                edge_mask = tf.sigmoid(mask_logits)
                logits = forward_fn(model, X_sub, sub_edges, edge_mask, num_sub_nodes)
                pred_probs = logits[local_target]

                # Compute the loss components
                pred_loss = -tf.math.log(pred_probs[predicted_class] + 1e-9)
                l1_loss = LAMBDA_L1 * tf.reduce_sum(edge_mask)
                entropy = -edge_mask * tf.math.log(edge_mask + 1e-9) \
                          - (1.0 - edge_mask) * tf.math.log(1.0 - edge_mask + 1e-9)
                entropy_loss = LAMBDA_ENTROPY * tf.reduce_mean(entropy)
                total_loss = pred_loss + l1_loss + entropy_loss

            # Compute gradients and update the mask logits
            grads = tape.gradient(total_loss, [mask_logits])
            optimizer.apply_gradients(zip(grads, [mask_logits]))

        final_mask = tf.sigmoid(mask_logits).numpy()
        return self._build_explanation_graph(
            sub_nodes, sub_edges, final_mask, node_idx, node_map,
            one_hop_nodes, predicted_class, confidence
        )

    def explain_node_inductive(
        self, text_vec: np.ndarray, model_name: str,
        num_steps: int = DEFAULT_NUM_STEPS, lr: float = DEFAULT_LR
    ) -> dict:
        """GNNExplainer for a new node using pseudo-neighbor injection."""
        from app.deps import get_inductive_service
        inductive = get_inductive_service()

        model = self._registry.nc_models.get(model_name)
        if model is None:
            raise ValueError(f"NC model '{model_name}' not found")

        K_NEIGHBORS = 10
        neighbor_indices, _, _, _ = inductive._find_pseudo_neighbors(text_vec, k=K_NEIGHBORS)

        X_extended = np.vstack([self._registry.X_full, text_vec.reshape(1, -1)])
        new_idx = X_extended.shape[0] - 1

        # Build the extended edges for the new node and its pseudo-neighbors       
        extended_edges_full = inductive._build_extended_edges(neighbor_indices, new_idx)
        actual_pred = inductive._nc_forward(
            model_name, model, X_extended, new_idx, text_vec, extended_edges_full
        )
        actual_class = int(np.argmax(actual_pred))
        actual_confidence = float(actual_pred[actual_class])

        # Extract a subgraph around the new node and its pseudo-neighbors for explanation
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

        # Build the local edges for the subgraph, including the new node and its pseudo-neighbors
        edge_index = self._registry.edge_index
        edges_local_src = []
        edges_local_dst = []
        for i in range(edge_index.shape[1]):
            s, d = int(edge_index[0, i]), int(edge_index[1, i])
            if s in node_map and d in node_map:
                edges_local_src.append(node_map[s])
                edges_local_dst.append(node_map[d])

        # Add edges from the new node to its pseudo-neighbors in the local subgraph
        local_new = node_map[new_idx]
        for n_idx in neighbor_indices:
            if int(n_idx) in node_map:
                local_n = node_map[int(n_idx)]
                edges_local_src.append(local_new)
                edges_local_dst.append(local_n)
                edges_local_src.append(local_n)
                edges_local_dst.append(local_new)
        
        # If there are no edges in the local subgraph, we cannot optimize a mask, so we return the explanation graph with an empty mask.
        if len(edges_local_src) == 0:
            return self._build_explanation_graph_inductive(
                sub_nodes, np.zeros((2, 0), dtype=np.int32), np.array([]),
                new_idx, one_hop_of_new | {new_idx},
                actual_class, actual_confidence
            )

        # Build the local edge index for the subgraph
        sub_edges = np.array([edges_local_src, edges_local_dst], dtype=np.int32)
        num_sub_nodes = len(sub_nodes)
        num_sub_edges = sub_edges.shape[1]

        # Create a TensorFlow constant for the features of the subgraph nodes
        X_sub = tf.constant(X_extended[sub_nodes], dtype=tf.float32)

        # Select the appropriate forward function based on the model type
        dispatch = {
            "GCN": self._masked_forward_gcn,
            "GAT": self._masked_forward_gat,
            "GraphSAGE": self._masked_forward_sage,
        }
        forward_fn = dispatch[model_name]

        # Verify the local subgraph preserves the inductive prediction the
        # rest of the system reports. If not, we'd be explaining a class the user never sees.
        local_pred = forward_fn(
            model, X_sub, sub_edges, tf.ones(num_sub_edges), num_sub_nodes
        )[local_new].numpy()
        local_class = int(np.argmax(local_pred))
        if local_class != actual_class:
            raise RuntimeError(
                "Inductive explanation subgraph does not preserve the "
                f"original prediction: actual_class={actual_class}, "
                f"local_class={local_class}. Explanation cannot be produced."
            )
        predicted_class = actual_class
        confidence = actual_confidence

        # Initialize the edge mask logits for optimization
        mask_logits = tf.Variable(tf.zeros(num_sub_edges), trainable=True)
        optimizer = tf.optimizers.Adam(learning_rate=lr)

        # Optimize the edge mask to maximize the prediction probability of the target class while regularizing for sparsity and entropy
        for _ in range(num_steps):
            with tf.GradientTape() as tape:
                edge_mask = tf.sigmoid(mask_logits)
                logits = forward_fn(model, X_sub, sub_edges, edge_mask, num_sub_nodes)
                pred_probs = logits[local_new]

                # Compute the loss components
                pred_loss = -tf.math.log(pred_probs[predicted_class] + 1e-9)
                l1_loss = LAMBDA_L1 * tf.reduce_sum(edge_mask)
                entropy = -edge_mask * tf.math.log(edge_mask + 1e-9) \
                          - (1.0 - edge_mask) * tf.math.log(1.0 - edge_mask + 1e-9)
                entropy_loss = LAMBDA_ENTROPY * tf.reduce_mean(entropy)
                total_loss = pred_loss + l1_loss + entropy_loss

            # Compute gradients and update the mask logits
            grads = tape.gradient(total_loss, [mask_logits])
            optimizer.apply_gradients(zip(grads, [mask_logits]))

        # Apply the final sigmoid to get the edge mask values   
        final_mask = tf.sigmoid(mask_logits).numpy()
        return self._build_explanation_graph_inductive(
            sub_nodes, sub_edges, final_mask, new_idx,
            one_hop_of_new | {new_idx}, predicted_class, confidence
        )

    def _build_explanation_graph_inductive(
        self, sub_nodes: np.ndarray, sub_edges: np.ndarray,
        mask_values: np.ndarray, new_node_idx: int,
        one_hop_nodes: Set[int],
        predicted_class: int, confidence: float,
    ) -> dict:
        """Build explanation graph with the synthetic new node included."""
        edges = []
        # Aggregate undirected edges by averaging both directions' importance.
        # Mean softens asymmetric attention - edges whose mask is only
        # high in one direction get a balanced score and may stay below the threshold
        edge_acc: Dict[Tuple[int, int], List] = {}
        for i in range(len(mask_values)):
            val = float(mask_values[i])
            # Only consider edges that connect to the new node or its one-hop neighbors
            src_global = int(sub_nodes[sub_edges[0, i]])
            dst_global = int(sub_nodes[sub_edges[1, i]])
            if src_global not in one_hop_nodes and dst_global not in one_hop_nodes:
                continue
            # Use a consistent ordering for undirected edges to aggregate importance
            edge_key = (min(src_global, dst_global), max(src_global, dst_global))
            acc = edge_acc.get(edge_key)
            if acc is None:
                edge_acc[edge_key] = [val, 1, src_global, dst_global]
            else:
                acc[0] += val
                acc[1] += 1
        # Aggregate edges by averaging importance and filter by threshold
        edge_candidates = []
        for total, count, src_global, dst_global in edge_acc.values():
            mean_val = total / count
            if mean_val < MASK_THRESHOLD:
                continue
            edge_candidates.append((mean_val, src_global, dst_global))
        edge_candidates.sort(key=lambda x: x[0], reverse=True)
        final_edges = edge_candidates[:MAX_EXPLANATION_EDGES]
        
        # Build the set of relevant nodes and their importance scores based on the final edges
        relevant_nodes = {new_node_idx}
        node_importance = {}
        for val, src_global, dst_global in final_edges:
            relevant_nodes.add(src_global)
            relevant_nodes.add(dst_global)
            node_importance[src_global] = max(node_importance.get(src_global, 0.0), val)
            node_importance[dst_global] = max(node_importance.get(dst_global, 0.0), val)

        # Build the node data for the explanation graph, including the synthetic new node
        nodes = []
        for global_idx in sub_nodes:
            global_idx = int(global_idx)
            if global_idx not in relevant_nodes:
                continue
            # If the node is the synthetic new node, assign it a special ID and label
            if global_idx == new_node_idx:
                node_data = {
                    "id": "__new__",
                    "label": "New Product",
                    "class": self._registry.get_class_name(predicted_class),
                    "class_source": "predicted",
                    "is_center": True,
                    "is_new": True,
                }
            else:
                # For existing nodes, use their ASIN and ground truth class
                asin = str(self._registry.node_names[global_idx])
                class_name = self._registry.get_class_name(int(self._registry.y_full[global_idx]))
                node_data = {
                    "id": asin,
                    "label": asin,
                    "class": class_name,
                    "class_source": "ground_truth",
                    "is_center": False,
                }
            if global_idx in node_importance:
                node_data["importance"] = round(node_importance[global_idx], 4)
            nodes.append({"data": node_data})

        # Build the edge data for the explanation graph, using the special ID for the synthetic new node
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
        # Aggregate undirected edges by averaging both directions' importance.
        # Mean softens asymmetric attention - edges whose mask is only
        # high in one direction get a balanced score and may stay below the threshold
        edge_acc: Dict[Tuple[int, int], List] = {}
        for i in range(len(mask_values)):
            val = float(mask_values[i])
            # Only consider edges that connect to the center node or its one-hop neighbors
            src_global = int(sub_nodes[sub_edges[0, i]])
            dst_global = int(sub_nodes[sub_edges[1, i]])
            if src_global not in one_hop_nodes and dst_global not in one_hop_nodes:
                continue
            # Use a consistent ordering for undirected edges to aggregate importance
            edge_key = (min(src_global, dst_global), max(src_global, dst_global))
            acc = edge_acc.get(edge_key)
            if acc is None:
                edge_acc[edge_key] = [val, 1, src_global, dst_global]
            else:
                acc[0] += val
                acc[1] += 1
        # Aggregate edges by averaging importance and filter by threshold
        edge_candidates = []
        for total, count, src_global, dst_global in edge_acc.values():
            mean_val = total / count
            if mean_val < MASK_THRESHOLD:
                continue
            edge_candidates.append((mean_val, src_global, dst_global))
        edge_candidates.sort(key=lambda x: x[0], reverse=True)
        final_edges = edge_candidates[:MAX_EXPLANATION_EDGES]

        # Build the set of relevant nodes and their importance scores based on the final edges
        relevant_nodes = {center_idx}
        if center_idx_2 is not None:
            relevant_nodes.add(center_idx_2)
        node_importance = {}
        for val, src_global, dst_global in final_edges:
            relevant_nodes.add(src_global)
            relevant_nodes.add(dst_global)
            node_importance[src_global] = max(node_importance.get(src_global, 0.0), val)
            node_importance[dst_global] = max(node_importance.get(dst_global, 0.0), val)

        # Build the node data for the explanation graph
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
                "class_source": "ground_truth",
                "is_center": global_idx == center_idx or global_idx == center_idx_2,
            }
            if global_idx in node_importance:
                node_data["importance"] = round(node_importance[global_idx], 4)
            nodes.append({"data": node_data})

        # Build the edge data for the explanation graph
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
        # Build the final result dictionary with nodes, edges, and optional prediction information
        result = {"nodes": nodes, "edges": edges}
        if predicted_class is not None:
            result["predicted_class_idx"] = predicted_class
            result["predicted_class"] = self._registry.get_class_name(predicted_class)
            result["confidence"] = round(confidence, 4) if confidence is not None else None
        if lp_score is not None:
            result["lp_score"] = round(lp_score, 4)
        return result
