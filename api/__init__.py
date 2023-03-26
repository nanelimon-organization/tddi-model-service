from fastapi import APIRouter
from .controllers.model_controller import model_router

router = APIRouter()

router.include_router(
    model_router,
    tags=["BERT-Based Turkish Text Classification API"]
)