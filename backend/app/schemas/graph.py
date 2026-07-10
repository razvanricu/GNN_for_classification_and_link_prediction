from typing import List, Optional
from pydantic import BaseModel


class CytoscapeNodeData(BaseModel):
    id: str
    label: str
    class_: Optional[str] = None
    is_new: Optional[bool] = None
    is_center: Optional[bool] = None


class CytoscapeEdgeData(BaseModel):
    source: str
    target: str
    type: str  # real or predicted
    model: Optional[str] = None
    score: Optional[float] = None


class CytoscapeGraph(BaseModel):
    nodes: List[dict]
    edges: List[dict]
