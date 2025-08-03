# legal_assistant/api/routers/health.py
"""Health check endpoints"""
import logging
from datetime import datetime
from fastapi import APIRouter
from ...config import FeatureFlags

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "features": {
            "ai_enabled": FeatureFlags.AI_ENABLED,
            "ocr_available": FeatureFlags.OCR_AVAILABLE,
            "hybrid_search": FeatureFlags.HYBRID_SEARCH_AVAILABLE,
            "pymupdf_available": FeatureFlags.PYMUPDF_AVAILABLE,
            "pdfplumber_available": FeatureFlags.PDFPLUMBER_AVAILABLE,
            "unstructured_available": FeatureFlags.UNSTRUCTURED_AVAILABLE
        }
    }

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with component status"""
    from ...services.container_manager import get_container_manager
    from ...storage.managers import uploaded_files, user_sessions
    
    try:
        # Check if container manager is accessible
        container_manager = get_container_manager()
        container_status = "healthy" if container_manager else "unhealthy"
    except Exception as e:
        container_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "container_manager": container_status,
            "uploaded_files_count": len(uploaded_files),
            "active_sessions": len(user_sessions),
            "ai_enabled": FeatureFlags.AI_ENABLED
        },
        "features": {
            "ai_enabled": FeatureFlags.AI_ENABLED,
            "ocr_available": FeatureFlags.OCR_AVAILABLE,
            "hybrid_search": FeatureFlags.HYBRID_SEARCH_AVAILABLE,
            "document_processors": {
                "pymupdf": FeatureFlags.PYMUPDF_AVAILABLE,
                "pdfplumber": FeatureFlags.PDFPLUMBER_AVAILABLE,
                "unstructured": FeatureFlags.UNSTRUCTURED_AVAILABLE
            }
        }
    }
