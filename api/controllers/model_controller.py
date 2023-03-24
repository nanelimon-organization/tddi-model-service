from fastapi import APIRouter

model_router = APIRouter()

@model_router.post("/single_prediction")
async def single_prediction():
    return {"message": "Hello, World!"}


@model_router.post("/bulk_prediction")
async def bulk_prediction():
    return {"message": "Hello, World!"}