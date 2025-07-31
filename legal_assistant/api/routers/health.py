"""Health check endpoints"""
import os
from datetime import datetime
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from ...config import (
    DEFAULT_CHROMA_PATH, USER_CONTAINERS_PATH, OPENROUTER_API_KEY, FeatureFlags
)
from ...models import ConversationHistory
from ...storage.managers import conversations
from ...core.dependencies import get_nlp, get_sentence_model, get_embeddings, get_sentence_model_name

router = APIRouter()

@router.get("/health")
def health_check():
    """Enhanced system health check with comprehensive analysis capabilities"""
    db_exists = os.path.exists(DEFAULT_CHROMA_PATH)
    
    nlp = get_nlp()
    sentence_model = get_sentence_model()
    embeddings = get_embeddings()
    sentence_model_name = get_sentence_model_name()
    
    return {
        "status": "healthy",
        "version": "10.0.0-SmartRAG-ComprehensiveAnalysis",
        "timestamp": datetime.utcnow().isoformat(),
        "ai_enabled": FeatureFlags.AI_ENABLED,
        "openrouter_api_configured": bool(OPENROUTER_API_KEY),
        "components": {
            "default_database": {
                "exists": db_exists,
                "path": DEFAULT_CHROMA_PATH
            },
            "user_containers": {
                "enabled": True,
                "base_path": USER_CONTAINERS_PATH,
                "active_containers": len(os.listdir(USER_CONTAINERS_PATH)) if os.path.exists(USER_CONTAINERS_PATH) else 0,
                "document_specific_retrieval": True,
                "file_id_tracking": True
            },
            "external_databases": {
                "lexisnexis": {
                    "configured": bool(os.environ.get("LEXISNEXIS_API_KEY")),
                    "status": "ready" if bool(os.environ.get("LEXISNEXIS_API_KEY")) else "not_configured"
                },
                "westlaw": {
                    "configured": bool(os.environ.get("WESTLAW_API_KEY")),
                    "status": "ready" if bool(os.environ.get("WESTLAW_API_KEY")) else "not_configured"
                }
            },
            "comprehensive_analysis": {
                "enabled": True,
                "analysis_types": [
                    "comprehensive",
                    "document_summary", 
                    "key_clauses",
                    "risk_assessment",
                    "timeline_deadlines", 
                    "party_obligations",
                    "missing_clauses"
                ],
                "structured_output": True,
                "document_specific": True,
                "confidence_scoring": True,
                "single_api_call": True
            },
            "enhanced_rag": {
                "enabled": True,
                "features": [
                    "multi_query_strategies",
                    "query_expansion",
                    "entity_extraction",
                    "sub_query_decomposition",
                    "confidence_scoring",
                    "duplicate_removal",
                    "document_specific_filtering"
                ],
                "nlp_model": nlp is not None,
                "sentence_model": sentence_model is not None,
                "sentence_model_name": sentence_model_name if sentence_model else "none",
                "embedding_model": getattr(embeddings, 'model_name', 'unknown') if embeddings else "none"
            },
            "document_processing": {
                "pdf_support": FeatureFlags.PYMUPDF_AVAILABLE or FeatureFlags.PDFPLUMBER_AVAILABLE,
                "pymupdf_available": FeatureFlags.PYMUPDF_AVAILABLE,
                "pdfplumber_available": FeatureFlags.PDFPLUMBER_AVAILABLE,
                "unstructured_available": FeatureFlags.UNSTRUCTURED_AVAILABLE,
                "docx_support": True,
                "txt_support": True,
                "safe_document_processor": True,
                "enhanced_page_estimation": True,
                "bert_semantic_chunking": sentence_model is not None,
                "advanced_legal_chunking": True,
                "embedding_model": sentence_model_name if sentence_model else "none"
            }
        },
        "new_endpoints": [
            "POST /comprehensive-analysis - Full structured analysis",
            "POST /quick-analysis/{document_id} - Quick single document analysis", 
            "Enhanced /ask - Detects comprehensive analysis requests",
            "Enhanced /user/upload - Stores file_id for targeting",
            "GET /admin/document-health - Check system health",
            "POST /admin/cleanup-containers - Clean orphaned containers",
            "POST /admin/emergency-clear-tracking - Reset document tracking"
        ],
        "features": [
            "✅ User-specific document containers",
            "✅ Enhanced RAG with multi-query strategies",
            "✅ Combined search across all sources",
            "✅ External legal database integration (ready)",
            "✅ Subscription tier management",
            "✅ Document access control",
            "✅ Source attribution (default/user/external)",
            "✅ Dynamic confidence scoring",
            "✅ Query expansion and decomposition",
            "✅ SafeDocumentProcessor for file handling",
            "🔧 Optional authentication for debugging",
            "🆕 Comprehensive multi-analysis in single API call",
            "🆕 Document-specific analysis targeting",
            "🆕 Structured analysis responses with sections",
            "🆕 Enhanced confidence scoring per section",
            "🆕 File ID tracking for precise document retrieval",
            "🆕 Automatic comprehensive analysis detection",
            "🆕 Container cleanup and health monitoring",
            "🆕 Enhanced error handling and recovery",
            "🆕 Fixed page estimation with content analysis",
            "🆕 Unstructured.io integration for advanced processing",
            "🆕 BERT-based semantic chunking for better retrieval",
            "🆕 Enhanced information extraction (bills, sponsors, etc.)",
            "🆕 Legal-specific BERT models (InCaseLawBERT, legal-bert-base-uncased)",
            "🆕 Advanced semantic similarity for intelligent chunking",
            "🆕 Legal document pattern recognition for better segmentation"
        ],
        # Frontend compatibility fields
        "unified_mode": True,
        "enhanced_rag": True,
        "database_exists": db_exists,
        "database_path": DEFAULT_CHROMA_PATH,
        "api_key_configured": bool(OPENROUTER_API_KEY),
        "active_conversations": len(conversations)
    }

@router.get("/conversation/{session_id}", response_model=ConversationHistory)
async def get_conversation(session_id: str):
    """Get the conversation history for a session"""
    from fastapi import HTTPException
    
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return ConversationHistory(
        session_id=session_id,
        messages=conversations[session_id]['messages']
    )

@router.get("/subscription/status")
async def get_subscription_status():
    """Get user's subscription status and available features"""
    from ...core.security import get_current_user
    from fastapi import Depends
    
    current_user = await get_current_user()
    
    features = {
        "free": {
            "default_database_access": True,
            "user_container": True,
            "max_documents": 10,
            "external_databases": [],
            "ai_analysis": True,
            "api_calls_per_month": 100,
            "enhanced_rag": True,
            "comprehensive_analysis": True
        },
        "premium": {
            "default_database_access": True,
            "user_container": True,
            "max_documents": 100,
            "external_databases": ["lexisnexis", "westlaw"],
            "ai_analysis": True,
            "api_calls_per_month": 1000,
            "priority_support": True,
            "enhanced_rag": True,
            "comprehensive_analysis": True,
            "document_specific_analysis": True
        },
        "enterprise": {
            "default_database_access": True,
            "user_container": True,
            "max_documents": "unlimited",
            "external_databases": ["lexisnexis", "westlaw", "bloomberg_law"],
            "ai_analysis": True,
            "api_calls_per_month": "unlimited",
            "priority_support": True,
            "custom_integrations": True,
            "enhanced_rag": True,
            "comprehensive_analysis": True,
            "document_specific_analysis": True,
            "bulk_analysis": True
        }
    }
    
    return {
        "user_id": current_user.user_id,
        "subscription_tier": current_user.subscription_tier,
        "features": features.get(current_user.subscription_tier, features["free"]),
        "external_db_access": current_user.external_db_access
    }

@router.get("/", response_class=HTMLResponse)
def get_interface():
    """Web interface with updated documentation for comprehensive analysis"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Legal Assistant - Complete Multi-Analysis Edition [MODULAR]</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f8f9fa; }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { color: #2c3e50; }
            .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0; }
            .feature-card { background: #fff; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; }
            .endpoint { background: #f1f3f4; padding: 10px; margin: 10px 0; border-radius: 5px; font-family: monospace; }
            .status { padding: 5px 10px; border-radius: 15px; font-size: 12px; }
            .status-active { background: #d4edda; color: #155724; }
            .status-ready { background: #cce5ff; color: #004085; }
            .status-modular { background: #28a745; color: white; }
            .badge-modular { background: #17a2b8; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-left: 5px; }
            .code-example { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 5px; padding: 15px; margin: 10px 0; font-family: monospace; font-size: 12px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>⚖️ Legal Assistant API v10.0 <span class="badge-modular">MODULAR ARCHITECTURE</span></h1>
            <p>Complete Multi-User Platform with Enhanced RAG, Comprehensive Analysis, and Container Management</p>
            <div class="status status-modular">🎯 Clean Modular Structure - Easy to Maintain and Extend!</div>
            
            <div class="feature-grid">
                <div class="feature-card">
                    <h3>📁 Modular Architecture</h3>
                    <p>Clean separation of concerns</p>
                    <ul>
                        <li>✅ Services for business logic</li>
                        <li>✅ API routers for endpoints</li>
                        <li>✅ Models for data structures</li>
                        <li>✅ Utils for helper functions</li>
                    </ul>
                </div>
                
                <div class="feature-card">
                    <h3>🚀 Comprehensive Analysis</h3>
                    <p>All analysis types in a single efficient API call</p>
                    <ul>
                        <li>✅ Document summary</li>
                        <li>✅ Key clauses extraction</li>
                        <li>✅ Risk assessment</li>
                        <li>✅ Timeline & deadlines</li>
                        <li>✅ Party obligations</li>
                        <li>✅ Missing clauses detection</li>
                    </ul>
                </div>
                
                <div class="feature-card">
                    <h3>🛠️ Enhanced Error Handling</h3>
                    <p>Robust container management with auto-recovery</p>
                    <ul>
                        <li>✅ Timeout protection</li>
                        <li>✅ Container auto-recovery</li>
                        <li>✅ Graceful degradation</li>
                        <li>✅ Health monitoring</li>
                    </ul>
                </div>
            </div>
            
            <h2>📡 API Reference</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <h4>Core Endpoints</h4>
                    <div class="endpoint">POST /api/ask - Enhanced chat</div>
                    <div class="endpoint">POST /api/user/upload - Document upload</div>
                    <div class="endpoint">GET /api/user/documents - List documents</div>
                </div>
                
                <div class="feature-card">
                    <h4>Analysis Endpoints</h4>
                    <div class="endpoint">POST /api/comprehensive-analysis</div>
                    <div class="endpoint">POST /api/quick-analysis/{id}</div>
                </div>
                
                <div class="feature-card">
                    <h4>Admin Endpoints</h4>
                    <div class="endpoint">GET /api/admin/document-health</div>
                    <div class="endpoint">POST /api/admin/cleanup-containers</div>
                </div>
            </div>
            
            <p style="text-align: center; color: #7f8c8d; margin-top: 30px;">
                🎉 Modular Legal Assistant Backend - Clean Architecture 🎉
                <br>Version 10.0.0-SmartRAG-ComprehensiveAnalysis
                <br>Fully modularized for easy maintenance and extensibility!
            </p>
        </div>
    </body>
    </html>
    """
