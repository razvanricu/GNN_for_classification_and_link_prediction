from fastapi import APIRouter, Depends, HTTPException, Query

from app.deps import get_catalog_service, get_nc_service, get_lp_service

router = APIRouter(prefix="/api/products", tags=["products"])


@router.get("/{asin}")
def get_product(asin: str, catalog=Depends(get_catalog_service)):
    product = catalog.get_by_asin(asin)
    if not product:
        raise HTTPException(status_code=404, detail=f"Product '{asin}' not found")
    return product


@router.get("/{asin}/predictions")
def get_predictions(asin: str, nc_service=Depends(get_nc_service)):
    try:
        predictions = nc_service.predict_for_node(asin)
    except KeyError:
        raise HTTPException(
            status_code=422,
            detail=f"ASIN '{asin}' is not in the graph. Use POST /api/predict/full for new products.",
        )
    return predictions


@router.get("/{asin}/link-suggestions")
def get_link_suggestions(
    asin: str,
    top_k: int = Query(default=10, ge=1, le=50),
    lp_service=Depends(get_lp_service),
):
    try:
        suggestions = lp_service.top_k_for_node(asin, k=top_k)
    except KeyError:
        raise HTTPException(
            status_code=422,
            detail=f"ASIN '{asin}' is not in the graph. Use POST /api/predict/lp for new products.",
        )
    return suggestions
