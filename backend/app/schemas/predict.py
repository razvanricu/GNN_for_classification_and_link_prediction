from typing import List, Optional
from pydantic import BaseModel, Field


class NewProductInput(BaseModel):
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)


class LPInductiveRequest(BaseModel):
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    candidate_asins: Optional[List[str]] = None


class FullInductiveRequest(BaseModel):
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
