import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.utils.logging import setup_logging
from app.deps import get_model_registry, get_analytics_service

from app.routers import health, catalog, products, predict, graph, analytics, ws, explainer

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models and analytics. Shutdown: cleanup."""
    logger.info("Starting up — loading models...")
    registry = get_model_registry()
    registry.warmup()

    analytics_svc = get_analytics_service()
    analytics_svc.load()

    logger.info("Startup complete.")
    yield
    logger.info("Shutting down...")


settings = get_settings()

app = FastAPI(
    title="GNN Thesis API",
    description="Backend for the Amazon Photo GNN thesis demo — NC, LP, inductive inference",
    version="0.1.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router)
app.include_router(catalog.router)
app.include_router(products.router)
app.include_router(predict.router)
app.include_router(graph.router)
app.include_router(analytics.router)
app.include_router(ws.router)
app.include_router(explainer.router)


# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(status_code=404, content={"detail": "Not found"})


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
