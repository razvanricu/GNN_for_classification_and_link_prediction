from fastapi import APIRouter, Depends, Query, HTTPException

from app.deps import get_graph_service
from app.schemas.predict import FullInductiveRequest

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("/neighborhood/{asin}")
def get_neighborhood(
    asin: str,
    hops: int = Query(default=2, ge=1, le=3),
    max_nodes: int = Query(default=80, ge=10, le=300),
    graph_service=Depends(get_graph_service),
):
    try:
        return graph_service.k_hop_neighborhood(asin, hops, max_nodes)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"ASIN '{asin}' not found in graph")


@router.get("/with-predicted-links/{asin}")
def get_with_predicted_links(
    asin: str,
    model: str = Query(default="GraphSAGE"),
    top_k: int = Query(default=5, ge=1, le=20),
    hops: int = Query(default=2, ge=1, le=3),
    max_nodes: int = Query(default=80, ge=10, le=300),
    graph_service=Depends(get_graph_service),
):
    try:
        return graph_service.with_predicted_links(asin, model, top_k, hops, max_nodes)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"ASIN '{asin}' not found in graph")


@router.get("/with-tp-fp-fn/{asin}")
def get_with_tp_fp_fn(
    asin: str,
    model: str = Query(default="GraphSAGE"),
    hops: int = Query(default=1, ge=1, le=3),
    max_nodes: int = Query(default=30, ge=10, le=300),
    graph_service=Depends(get_graph_service),
):
    try:
        return graph_service.with_tp_fp_fn(asin, model, hops, max_nodes)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"ASIN '{asin}' not found in graph")


@router.post("/inductive-preview")
def inductive_preview(
    body: FullInductiveRequest,
    model: str = Query(default="GraphSAGE"),
    graph_service=Depends(get_graph_service),
):
    return graph_service.inductive_subgraph(
        body.title, body.description, top_k=body.top_k, model_name=model
    )
