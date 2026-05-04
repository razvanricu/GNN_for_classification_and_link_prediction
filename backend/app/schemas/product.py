from typing import List, Optional
from pydantic import BaseModel


class ProductDetail(BaseModel):
    asin: str
    title: str
    description: Optional[str]
    price: Optional[float]
    imUrl: Optional[str]
    categories: Optional[list]
    target_class: str
    in_graph: bool
    graph_idx: Optional[int]
    also_bought: Optional[List[str]]
    also_viewed: Optional[List[str]]


class NCPredictionItem(BaseModel):
    predicted_class: str
    confidence: float
    top_3: List[dict]


class LinkSuggestion(BaseModel):
    asin: str
    score: float
    predicted_link: bool
    target_class: str
