import typing


class GraphInfo:
    """Graph context object holding edge list and node count.
    Used by GCN layers to perform message passing."""

    def __init__(self, edges: typing.Tuple[list, list], num_nodes: int):
        self.edges = edges
        self.num_nodes = num_nodes
