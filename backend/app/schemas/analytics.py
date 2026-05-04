from typing import List, Optional
from pydantic import BaseModel


class ClassDistributionItem(BaseModel):
    class_name: str
    count: int


class GraphStats(BaseModel):
    num_nodes: int
    num_edges: int
    num_classes: int
    avg_degree: float
    density: float
