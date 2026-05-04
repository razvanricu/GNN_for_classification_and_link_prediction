from fastapi import APIRouter, Depends

from app.deps import get_inductive_service
from app.schemas.predict import NewProductInput, LPInductiveRequest, FullInductiveRequest

router = APIRouter(prefix="/api/predict", tags=["predict"])


@router.post("/nc")
def predict_nc(body: NewProductInput, service=Depends(get_inductive_service)):
    text_vec = service.vectorize_text(body.title, body.description)
    return service.predict_nc_inductive(text_vec)


@router.post("/lp")
def predict_lp(body: LPInductiveRequest, service=Depends(get_inductive_service)):
    text_vec = service.vectorize_text(body.title, body.description)
    return service.predict_lp_inductive(text_vec, top_k=body.top_k)


@router.post("/full")
def predict_full(body: FullInductiveRequest, service=Depends(get_inductive_service)):
    return service.predict_full(body.title, body.description, top_k=body.top_k)
