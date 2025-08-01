"""Main FastAPI application entry point"""
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import DEFAULT_CHROMA_PATH, USER_CONTAINERS_PATH, FeatureFlags
from .core import initialize_nlp_models, initialize_feature_flags
from .services import initialize_container_manager
from .api.routers import query, documents, analysis, admin, health, external  # ADD external HERE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories
os.makedirs(USER_CONTAINERS_PATH, exist_ok=True)

# Initialize everything
logger.info(f"Using DEFAULT_CHROMA_PATH: {DEFAULT_CHROMA_PATH}")
logger.info(f"Using USER_CONTAINERS_PATH: {USER_CONTAINERS_PATH}")

initialize_feature_flags()
initialize_nlp_models()
initialize_container_manager()

# Create FastAPI app
app = FastAPI(
    title="Unified Legal Assistant API",
    description="Multi-User Legal Assistant with Enhanced RAG, Comprehensive Analysis, and External Database Integration",
    version="10.0.0-SmartRAG-ComprehensiveAnalysis"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(query.router, tags=["queries"])
app.include_router(documents.router, tags=["documents"])
app.include_router(analysis.router, tags=["analysis"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(external.router, prefix="/external", tags=["external"])  # ADD THIS LINE
app.include_router(health.router, tags=["health"])

# Mount the home page from health router
app.mount("/", health.router)

def create_app():
    """Application factory"""
    return app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting Modular Legal Assistant on port {port}")
    logger.info(f"ChromaDB Path: {DEFAULT_CHROMA_PATH}")
    logger.info(f"User Containers Path: {USER_CONTAINERS_PATH}")
    logger.info(f"AI Status: {'ENABLED' if FeatureFlags.AI_ENABLED else 'DISABLED - Set OPENAI_API_KEY to enable'}")
    logger.info(f"PDF processing: PyMuPDF={FeatureFlags.PYMUPDF_AVAILABLE}, pdfplumber={FeatureFlags.PDFPLUMBER_AVAILABLE}")
    logger.info("Features: Comprehensive analysis, document-specific targeting, container cleanup, enhanced error handling")
    logger.info("Version: 10.0.0-SmartRAG-ComprehensiveAnalysis")
    logger.info("📁 MODULAR ARCHITECTURE - Clean separation of concerns!")
    uvicorn.run("legal_assistant.main:app", host="0.0.0.0", port=port, reload=True)

# DELETE EVERYTHING BELOW THIS LINE!
