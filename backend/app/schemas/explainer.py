from typing import Optional
from pydantic import BaseModel, Field


class NCExplainRequest(BaseModel):
    asin: str = Field(..., min_length=1)
    model: str = Field(..., pattern="^(GCN|GAT|GraphSAGE)$")
    num_steps: int = Field(default=200, ge=50, le=500)


class LPExplainRequest(BaseModel):
    src_asin: str = Field(..., min_length=1)
    dst_asin: str = Field(..., min_length=1)
    model: str = Field(..., pattern="^(GCN|GAT|GraphSAGE)$")
    num_steps: int = Field(default=200, ge=50, le=500)


class InductiveNCExplainRequest(BaseModel):
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    model: str = Field(..., pattern="^(GCN|GAT|GraphSAGE)$")
    num_steps: int = Field(default=200, ge=50, le=500)


class NCExplainResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    confidence: float
    graph: dict


class LPExplainResponse(BaseModel):
    lp_score: float
    linked: bool
    graph: dict
