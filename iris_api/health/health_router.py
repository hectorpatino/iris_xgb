from fastapi import APIRouter
from .health_schema import HealthSchema


health_router = APIRouter()


@health_router.get("/health", response_model=HealthSchema)
def get_health():
    notes = ["Currently without testing"]
    return HealthSchema(notes=notes)
