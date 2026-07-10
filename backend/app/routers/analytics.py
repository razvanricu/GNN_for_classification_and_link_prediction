from fastapi import APIRouter, Depends, Query

from app.deps import get_analytics_service, get_model_registry

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/class-distribution")
def class_distribution(analytics=Depends(get_analytics_service)):
    return analytics.get_class_distribution()


@router.get("/degree-distribution")
def degree_distribution(analytics=Depends(get_analytics_service)):
    return analytics.get_degree_distribution()


@router.get("/tsne")
def tsne(sample: int = Query(default=2000, ge=100, le=5000), analytics=Depends(get_analytics_service)):
    return analytics.get_tsne(sample)


@router.get("/model-comparison")
def model_comparison(analytics=Depends(get_analytics_service)):
    return analytics.get_model_comparison()


@router.get("/graph-stats")
def graph_stats(registry=Depends(get_model_registry)):
    edge_index = registry.edge_index
    num_nodes = registry.num_nodes
    num_edges = edge_index.shape[1] // 2  
    avg_degree = edge_index.shape[1] / num_nodes
    density = num_edges / (num_nodes * (num_nodes - 1) / 2)

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_classes": registry.num_classes,
        "avg_degree": round(avg_degree, 2),
        "density": round(density, 6),
    }
