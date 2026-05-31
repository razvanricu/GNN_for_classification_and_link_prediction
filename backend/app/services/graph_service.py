import logging
from collections import deque

import numpy as np

from app.services.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class GraphService:
    """Subgraph extraction for Cytoscape visualization."""

    def __init__(self, registry: ModelRegistry):
        self._registry = registry
        self._adj_list = None

    def _get_adj_list(self) -> dict:
        """Lazily build and cache adjacency list from edge_index."""
        if self._adj_list is None:
            edge_index = self._registry.edge_index
            adj = {}
            for i in range(edge_index.shape[1]):
                src, dst = int(edge_index[0, i]), int(edge_index[1, i])
                adj.setdefault(src, []).append(dst)
            self._adj_list = adj
        return self._adj_list

    def _bfs_raw(self, asin: str, hops: int = 2, max_nodes: int = 80):
        """BFS from given node, return raw (visited, edges_set, center_idx)."""
        idx = self._registry.asin_to_idx.get(asin)
        if idx is None:
            raise KeyError(f"ASIN '{asin}' not found in graph")

        adj = self._get_adj_list()

        visited = {idx}
        queue = deque([(idx, 0)])

        while queue and len(visited) < max_nodes:
            node, depth = queue.popleft()
            if depth >= hops:
                continue
            for neighbor in adj.get(node, []):
                if neighbor not in visited and len(visited) < max_nodes:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        edges_set = set()
        for node in visited:
            for neighbor in adj.get(node, []):
                if neighbor in visited:
                    edge_key = (min(node, neighbor), max(node, neighbor))
                    edges_set.add(edge_key)

        return visited, edges_set, idx

    def k_hop_neighborhood(self, asin: str, hops: int = 2, max_nodes: int = 80) -> dict:
        """BFS from given node, cap at max_nodes."""
        visited, edges_set, center_idx = self._bfs_raw(asin, hops, max_nodes)
        return self._build_cytoscape_graph(visited, edges_set, center_idx=center_idx)

    def with_predicted_links(
        self, asin: str, model_name: str = "GraphSAGE", top_k: int = 5,
        hops: int = 2, max_nodes: int = 80
    ) -> dict:
        """Neighborhood graph + predicted edges from LP model."""
        base_graph = self.k_hop_neighborhood(asin, hops, max_nodes)

        # Get LP predictions — use lazy import to avoid circular deps
        from app.deps import get_lp_service
        lp_service = get_lp_service()
        lp_results = lp_service.top_k_for_node(asin, k=top_k)

        model_suggestions = lp_results.get(model_name, [])
        for suggestion in model_suggestions:
            target_asin = suggestion["asin"]
            target_idx = self._registry.asin_to_idx.get(target_asin)
            if target_idx is None:
                continue

            node_id = target_asin
            if not any(n["data"]["id"] == node_id for n in base_graph["nodes"]):
                base_graph["nodes"].append({
                    "data": {
                        "id": node_id,
                        "label": node_id,
                        "class": suggestion["target_class"],
                        "is_predicted_neighbor": True,
                    }
                })

            base_graph["edges"].append({
                "data": {
                    "source": asin,
                    "target": node_id,
                    "type": "predicted",
                    "model": model_name,
                    "score": suggestion["score"],
                }
            })

        return base_graph

    def with_tp_fp_fn(
        self, asin: str, model_name: str = "GraphSAGE",
        hops: int = 1, max_nodes: int = 30
    ) -> dict:
        """Neighborhood graph with edges classified as TP, FP, or FN."""
        from app.deps import get_lp_service
        lp_service = get_lp_service()

        visited, edges_set, center_idx = self._bfs_raw(asin, hops, max_nodes)
        result = lp_service.score_neighborhood_edges(
            model_name, center_idx, visited, edges_set
        )

        nodes = []
        node_idx_set = set(visited)
        for idx in visited:
            node_asin = str(self._registry.node_names[idx])
            class_name = self._registry.get_class_name(int(self._registry.y_full[idx]))
            nodes.append({
                "data": {
                    "id": node_asin,
                    "label": node_asin,
                    "class": class_name,
                    "is_center": idx == center_idx,
                }
            })

        for fp in result["fp_edges"]:
            dst = fp["dst_idx"]
            if dst not in node_idx_set:
                node_idx_set.add(dst)
                fp_asin = str(self._registry.node_names[dst])
                class_name = self._registry.get_class_name(
                    int(self._registry.y_full[dst])
                )
                nodes.append({
                    "data": {
                        "id": fp_asin,
                        "label": fp_asin,
                        "class": class_name,
                        "is_fp_neighbor": True,
                    }
                })

        edges = []
        for edge_type, edge_list in [
            ("tp", result["tp_edges"]),
            ("fn", result["fn_edges"]),
            ("fp", result["fp_edges"]),
        ]:
            for e in edge_list:
                src_asin = str(self._registry.node_names[e["src_idx"]])
                dst_asin = str(self._registry.node_names[e["dst_idx"]])
                edges.append({
                    "data": {
                        "source": src_asin,
                        "target": dst_asin,
                        "type": edge_type,
                        "model": model_name,
                        "score": e["score"],
                    }
                })

        return {"nodes": nodes, "edges": edges}

    def inductive_subgraph(
        self, title: str, description: str, top_k: int = 5, model_name: str = "GraphSAGE"
    ) -> dict:
        """New node in center + predicted neighbors via LP model.
        Fetches a larger candidate pool, filters to predicted_link=True, shows top-k.
        """
        from app.deps import get_inductive_service
        inductive = get_inductive_service()
        text_vec = inductive.vectorize_text(title, description)
        lp_results = inductive.predict_lp_inductive(text_vec, top_k=50)

        nodes = [{
            "data": {
                "id": "__new__",
                "label": title[:50],
                "class": "New Product",
                "is_new": True,
                "is_center": True,
            }
        }]
        edges = []

        model_suggestions = lp_results.get(model_name, [])
        yes_links = [s for s in model_suggestions if s["predicted_link"]][:top_k]
        for suggestion in yes_links:
            target_asin = suggestion["asin"]
            nodes.append({
                "data": {
                    "id": target_asin,
                    "label": target_asin,
                    "class": suggestion["target_class"],
                }
            })
            edges.append({
                "data": {
                    "source": "__new__",
                    "target": target_asin,
                    "type": "predicted",
                    "model": model_name,
                    "score": suggestion["score"],
                }
            })

        return {"nodes": nodes, "edges": edges}

    def _build_cytoscape_graph(self, node_indices: set, edges_set: set, center_idx: int) -> dict:
        """Convert internal indices to Cytoscape JSON format."""
        nodes = []
        for idx in node_indices:
            asin = str(self._registry.node_names[idx])
            class_name = self._registry.get_class_name(int(self._registry.y_full[idx]))
            nodes.append({
                "data": {
                    "id": asin,
                    "label": asin,
                    "class": class_name,
                    "is_center": idx == center_idx,
                }
            })

        edges = []
        for src_idx, dst_idx in edges_set:
            src_asin = str(self._registry.node_names[src_idx])
            dst_asin = str(self._registry.node_names[dst_idx])
            edges.append({
                "data": {
                    "source": src_asin,
                    "target": dst_asin,
                    "type": "real",
                }
            })

        return {"nodes": nodes, "edges": edges}
