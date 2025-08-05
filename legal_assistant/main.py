"""Main FastAPI application entry point - ENHANCED WITH NOSQL"""
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List  # FIXED: Added missing imports
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import DEFAULT_CHROMA_PATH, USER_CONTAINERS_PATH, FeatureFlags
from .core.dependencies import initialize_nlp_models
from .services.container_manager import initialize_container_manager
from .api.routers import query, documents, health, immigration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories
os.makedirs(USER_CONTAINERS_PATH, exist_ok=True)
os.makedirs(DEFAULT_CHROMA_PATH, exist_ok=True)

# Initialize everything
logger.info(f"Using DEFAULT_CHROMA_PATH: {DEFAULT_CHROMA_PATH}")
logger.info(f"Using USER_CONTAINERS_PATH: {USER_CONTAINERS_PATH}")
initialize_nlp_models()
initialize_container_manager()

# MODERN: Lifespan event handler (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan manager with NoSQL initialization"""
    # Startup
    logger.info("🚀 Legal Assistant API starting up...")
    
    try:
        # ACTIVATE NOSQL PERFORMANCE
        from .storage.managers import get_enhanced_storage
        storage = await get_enhanced_storage()
        
        nosql_status = {
            'mongodb': storage.nosql_manager.mongodb_available if storage.nosql_manager else False,
            'redis': storage.nosql_manager.redis_available if storage.nosql_manager else False
        }
        
        if nosql_status['mongodb']:
            logger.info("🎯 HIGH PERFORMANCE MODE: MongoDB connected!")
            logger.info("📊 Documents will be stored in persistent database")
            logger.info("⚡ 10-100x faster document operations enabled")
        else:
            logger.warning("⚠️ BASIC MODE: Using in-memory storage")
            logger.warning("📝 Install MongoDB for high performance: sudo apt install mongodb-org")
        
        if nosql_status['redis']:
            logger.info("🚀 CACHING ACTIVE: Redis connected!")
            logger.info("💾 Search results will be cached for instant responses")
        else:
            logger.warning("⚠️ NO CACHING: Install Redis for caching: sudo apt install redis-server")
        
        # Store performance mode globally
        app.state.performance_mode = "high" if nosql_status['mongodb'] else "basic"
        app.state.nosql_status = nosql_status
        
    except Exception as e:
        logger.error(f"❌ NoSQL initialization failed: {e}")
        logger.warning("🔄 Falling back to in-memory storage")
        app.state.performance_mode = "basic"
        app.state.nosql_status = {'mongodb': False, 'redis': False}
    
    yield  # App runs here
    
    # Shutdown
    logger.info("👋 Legal Assistant API shutting down...")
    try:
        # Cleanup NoSQL connections
        from .storage.managers import get_enhanced_storage
        storage = await get_enhanced_storage()
        if storage.nosql_manager:
            await storage.nosql_manager.close_connections()
            logger.info("🔌 NoSQL connections closed")
    except Exception as e:
        logger.error(f"Error during NoSQL cleanup: {e}")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Unified Legal Assistant API",
    description="Multi-User Legal Assistant with Enhanced RAG, Comprehensive Analysis, and External Database Integration",
    version="10.0.0-SmartRAG-ComprehensiveAnalysis-NoSQL",
    lifespan=lifespan
)

# Request size limit middleware
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    if request.headers.get("content-length"):
        content_length = int(request.headers["content-length"])
        if content_length > 100 * 1024 * 1024:  # 100MB limit
            return JSONResponse(status_code=413, content={"detail": "Request too large"})
    return await call_next(request)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(query.router, tags=["queries"])
app.include_router(documents.router, tags=["documents"])

# ENHANCED: Startup/shutdown events with NoSQL initialization
@app.on_event("startup")
async def startup_event():
    """Initialize on startup with NoSQL activation"""
    logger.info("🚀 Legal Assistant API starting up...")
    
    try:
        # ACTIVATE NOSQL PERFORMANCE
        from .storage.managers import get_enhanced_storage
        storage = await get_enhanced_storage()
        
        nosql_status = {
            'mongodb': storage.nosql_manager.mongodb_available if storage.nosql_manager else False,
            'redis': storage.nosql_manager.redis_available if storage.nosql_manager else False
        }
        
        if nosql_status['mongodb']:
            logger.info("🎯 HIGH PERFORMANCE MODE: MongoDB connected!")
            logger.info("📊 Documents will be stored in persistent database")
            logger.info("⚡ 10-100x faster document operations enabled")
        else:
            logger.warning("⚠️ BASIC MODE: Using in-memory storage")
            logger.warning("📝 Install MongoDB for high performance: sudo apt install mongodb")
        
        if nosql_status['redis']:
            logger.info("🚀 CACHING ACTIVE: Redis connected!")
            logger.info("💾 Search results will be cached for instant responses")
        else:
            logger.warning("⚠️ NO CACHING: Install Redis for caching: sudo apt install redis-server")
        
        # Store performance mode globally
        app.state.performance_mode = "high" if nosql_status['mongodb'] else "basic"
        app.state.nosql_status = nosql_status
        
    except Exception as e:
        logger.error(f"❌ NoSQL initialization failed: {e}")
        logger.warning("🔄 Falling back to in-memory storage")
        app.state.performance_mode = "basic"
        app.state.nosql_status = {'mongodb': False, 'redis': False}

@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Legal Assistant API shutting down...")
    
    try:
        # Cleanup NoSQL connections
        from .storage.managers import get_enhanced_storage
        storage = await get_enhanced_storage()
        if storage.nosql_manager:
            await storage.nosql_manager.close_connections()
            logger.info("🔌 NoSQL connections closed")
    except Exception as e:
        logger.error(f"Error during NoSQL cleanup: {e}")

# Add performance monitoring endpoint
@app.get("/debug/performance")
async def check_performance_mode():
    """Check current performance mode and NoSQL status"""
    try:
        from .storage.managers import get_enhanced_storage
        storage = await get_enhanced_storage()
        
        stats = await storage.get_system_stats()
        
        return {
            "performance_mode": getattr(app.state, 'performance_mode', 'unknown'),
            "nosql_status": getattr(app.state, 'nosql_status', {}),
            "system_stats": stats,
            "recommendations": _get_performance_recommendations(stats)
        }
    except Exception as e:
        return {
            "performance_mode": "error",
            "error": str(e),
            "recommendations": ["Install MongoDB and Redis for high performance"]
        }

def _get_performance_recommendations(stats: Dict) -> List[str]:
    """Get performance recommendations based on current state"""
    recommendations = []
    
    if stats.get('storage_backend') == 'memory':
        recommendations.extend([
            "🚀 Install MongoDB for 10-100x faster document operations",
            "💾 Install Redis for instant search result caching",
            "📊 Enable persistent storage across server restarts"
        ])
    
    memory_stats = stats.get('memory_stats', {})
    if memory_stats.get('uploaded_files', 0) > 100:
        recommendations.append("⚠️ Large number of in-memory files - MongoDB recommended")
    
    if memory_stats.get('conversations', 0) > 50:
        recommendations.append("⚠️ Many active conversations - Redis caching recommended")
    
    if not recommendations:
        recommendations.append("✅ Performance optimization active!")
    
    return recommendations

def create_app():
    """Application factory"""
    return app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting Legal Assistant with NoSQL Performance on port {port}")
    logger.info(f"ChromaDB Path: {DEFAULT_CHROMA_PATH}")
    logger.info(f"User Containers Path: {USER_CONTAINERS_PATH}")
    logger.info(f"AI Status: {'ENABLED' if FeatureFlags.AI_ENABLED else 'DISABLED - Set OPENAI_API_KEY to enable'}")
    logger.info(f"PDF processing: PyMuPDF={FeatureFlags.PYMUPDF_AVAILABLE}, pdfplumber={FeatureFlags.PDFPLUMBER_AVAILABLE}")
    logger.info("Features: NoSQL performance, comprehensive analysis, document-specific targeting")
    logger.info("Version: 10.0.0-SmartRAG-ComprehensiveAnalysis-NoSQL")
    logger.info("🎯 HIGH-PERFORMANCE MODE READY - Install MongoDB/Redis to activate!")
    uvicorn.run("legal_assistant.main:app", host="0.0.0.0", port=port, reload=True)

