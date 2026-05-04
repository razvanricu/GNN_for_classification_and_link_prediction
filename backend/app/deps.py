from functools import lru_cache

from app.config import get_settings
from app.services.model_registry import ModelRegistry
from app.services.catalog_service import CatalogService
from app.services.nc_service import NCService
from app.services.lp_service import LPService
from app.services.graph_service import GraphService
from app.services.analytics_service import AnalyticsService
from app.services.inductive_service import InductiveService
from app.utils.ws_manager import ConnectionManager


@lru_cache()
def get_model_registry() -> ModelRegistry:
    return ModelRegistry()


@lru_cache()
def get_catalog_service() -> CatalogService:
    settings = get_settings()
    return CatalogService(settings.CATALOG_DB_PATH)


@lru_cache()
def get_nc_service() -> NCService:
    return NCService(get_model_registry())


@lru_cache()
def get_lp_service() -> LPService:
    return LPService(get_model_registry())


@lru_cache()
def get_graph_service() -> GraphService:
    return GraphService(get_model_registry())


@lru_cache()
def get_analytics_service() -> AnalyticsService:
    settings = get_settings()
    return AnalyticsService(settings.PRECOMPUTED_DIR)


@lru_cache()
def get_inductive_service() -> InductiveService:
    return InductiveService(get_model_registry())


@lru_cache()
def get_ws_manager() -> ConnectionManager:
    return ConnectionManager()
