# legal_assistant/api/routers/health.py
from fastapi import APIRouter
from ...config import FeatureFlags

router = APIRouter()

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "features": {
            "ai_enabled": FeatureFlags.AI_ENABLED,
            "ocr_available": FeatureFlags.OCR_AVAILABLE,
            "hybrid_search": FeatureFlags.HYBRID_SEARCH_AVAILABLE
        }
    }
