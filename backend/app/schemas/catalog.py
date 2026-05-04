from typing import Optional
from pydantic import BaseModel, Field


class CatalogQuery(BaseModel):
    q: Optional[str] = None
    category: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    in_graph: Optional[bool] = None
    page: int = Field(default=1, ge=1)
    size: int = Field(default=20, ge=1, le=100)


class ProductSummary(BaseModel):
    asin: str
    title: str
    price: Optional[float]
    imUrl: Optional[str]
    target_class: str
    in_graph: bool
    graph_idx: Optional[int]


class CatalogPage(BaseModel):
    items: list
    total: int
    page: int
    size: int
