import logging

from fastapi import APIRouter, Depends, HTTPException

from app.deps import get_explainer_service, get_model_registry, get_inductive_service
from app.schemas.explainer import (
    NCExplainRequest, NCExplainResponse,
    LPExplainRequest, LPExplainResponse,
    InductiveNCExplainRequest,
)

router = APIRouter(prefix="/api/explainer", tags=["explainer"])
logger = logging.getLogger(__name__)


@router.post("/nc", response_model=NCExplainResponse)
def explain_nc(body: NCExplainRequest, service=Depends(get_explainer_service),
               registry=Depends(get_model_registry)):
    node_idx = registry.asin_to_idx.get(body.asin)
    if node_idx is None:
        raise HTTPException(status_code=404, detail=f"ASIN '{body.asin}' not in graph")
    try:
        result = service.explain_node(node_idx, body.model, num_steps=body.num_steps)
    except Exception as e:
        logger.error(f"Explainer NC error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Explanation failed")
    return NCExplainResponse(
        predicted_class=result.get("predicted_class_idx", 0),
        predicted_label=result.get("predicted_class", ""),
        confidence=result.get("confidence", 0.0),
        graph={"nodes": result["nodes"], "edges": result["edges"]},
    )


@router.post("/lp", response_model=LPExplainResponse)
def explain_lp(body: LPExplainRequest, service=Depends(get_explainer_service),
               registry=Depends(get_model_registry)):
    src_idx = registry.asin_to_idx.get(body.src_asin)
    dst_idx = registry.asin_to_idx.get(body.dst_asin)
    if src_idx is None:
        raise HTTPException(status_code=404, detail=f"ASIN '{body.src_asin}' not in graph")
    if dst_idx is None:
        raise HTTPException(status_code=404, detail=f"ASIN '{body.dst_asin}' not in graph")
    try:
        result = service.explain_link(src_idx, dst_idx, body.model, num_steps=body.num_steps)
    except Exception as e:
        logger.error(f"Explainer LP error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Explanation failed")
    return LPExplainResponse(
        lp_score=result.get("lp_score", 0.0),
        linked=result.get("lp_score", 0.0) > 0.5,
        graph={"nodes": result["nodes"], "edges": result["edges"]},
    )


@router.post("/nc-inductive", response_model=NCExplainResponse)
def explain_nc_inductive(body: InductiveNCExplainRequest,
                         service=Depends(get_explainer_service),
                         inductive=Depends(get_inductive_service)):
    text_vec = inductive.vectorize_text(body.title, body.description)
    try:
        result = service.explain_node_inductive(text_vec, body.model, num_steps=body.num_steps)
    except Exception as e:
        logger.error(f"Explainer NC-inductive error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Explanation failed")
    return NCExplainResponse(
        predicted_class=result.get("predicted_class_idx", 0),
        predicted_label=result.get("predicted_class", ""),
        confidence=result.get("confidence", 0.0),
        graph={"nodes": result["nodes"], "edges": result["edges"]},
    )
