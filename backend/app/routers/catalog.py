from fastapi import APIRouter, Depends, Query

from app.deps import get_catalog_service
from app.schemas.catalog import CatalogQuery, CatalogPage

router = APIRouter(prefix="/api/catalog", tags=["catalog"])


@router.get("")
def list_products(
    q: str = None,
    category: str = None,
    min_price: float = None,
    max_price: float = None,
    in_graph: bool = None,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    catalog=Depends(get_catalog_service),
):
    query = CatalogQuery(
        q=q, category=category, min_price=min_price,
        max_price=max_price, in_graph=in_graph, page=page, size=size,
    )
    items, total = catalog.search(query)
    return {"items": items, "total": total, "page": page, "size": size}


@router.get("/categories")
def get_categories(catalog=Depends(get_catalog_service)):
    return catalog.get_categories()


@router.get("/stats")
def get_stats(catalog=Depends(get_catalog_service)):
    return catalog.get_stats()
