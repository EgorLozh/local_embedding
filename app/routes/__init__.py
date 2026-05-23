from fastapi import APIRouter

from app.routes.embed import router as embed_router
from app.routes.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(embed_router)
