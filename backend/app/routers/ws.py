import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["websocket"])
logger = logging.getLogger(__name__)


@router.websocket("/ws/inductive-stream")
async def inductive_stream(websocket: WebSocket):
    """Bulk inductive inference via WebSocket.
    Client sends: {"products": [{"title": ..., "description": ...}, ...]}
    Server streams: {"index": i, "total": n, "result": {...}} per item.
    """
    from app.deps import get_inductive_service
    service = get_inductive_service()

    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            products = data.get("products", [])
            total = len(products)

            for i, product in enumerate(products):
                title = product.get("title", "")
                description = product.get("description", "")

                result = service.predict_full(title, description, top_k=5)
                await websocket.send_json({
                    "index": i,
                    "total": total,
                    "status": "result",
                    "result": result,
                })

            await websocket.send_json({"status": "done", "total": total})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected (inductive-stream)")


@router.websocket("/ws/graph-explore")
async def graph_explore(websocket: WebSocket):
    """Interactive graph expansion via WebSocket.
    Client sends: {"action": "expand", "asin": "..."}
    Server responds with new nodes/edges.
    """
    from app.deps import get_graph_service
    service = get_graph_service()

    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "expand":
                asin = data.get("asin")
                hops = data.get("hops", 1)
                max_nodes = data.get("max_nodes", 30)
                try:
                    subgraph = service.k_hop_neighborhood(asin, hops, max_nodes)
                    await websocket.send_json({"status": "ok", "graph": subgraph})
                except KeyError:
                    await websocket.send_json({"status": "error", "detail": f"ASIN '{asin}' not found"})
            else:
                await websocket.send_json({"status": "error", "detail": f"Unknown action: {action}"})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected (graph-explore)")
