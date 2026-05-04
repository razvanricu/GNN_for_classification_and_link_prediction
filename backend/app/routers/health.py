from fastapi import APIRouter, Depends
from app.deps import get_model_registry

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health_check(registry=Depends(get_model_registry)):
    return {
        "status": "ok",
        "models_loaded": registry.is_loaded,
        "nc_models": list(registry.nc_models.keys()),
        "lp_models": list(registry.lp_models.keys()),
    }
