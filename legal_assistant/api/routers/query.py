# legal_assistant/api/routers/query.py - COMPLETE FIXED VERSION
"""Query endpoints - Updated to use new async QueryProcessor with external search capabilities and fixed reporting"""
import uuid
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException

from ...models import Query, QueryResponse, User
from ...core.security import get_current_user
from ...storage.managers import conversations, cleanup_expired_conversations

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/ask", response_model=QueryResponse)
async def ask_question(query: Query, current_user: User = Depends(get_current_user)):
    """Enhanced ask endpoint using new async QueryProcessor with external search"""
    logger.info(f"Received ask request: {query}")
    
    cleanup_expired_conversations()
    
    session_id = query.session_id or str(uuid.uuid4())
    user_id = query.user_id or current_user.user_id
    
    # Initialize conversation if needed
    if session_id not in conversations:
        conversations[session_id] = {
            "messages": [],
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow()
        }
    else:
        conversations[session_id]["last_accessed"] = datetime.utcnow()
    
    user_question = query.question.strip()
    if not user_question:
        return QueryResponse(
            response=None,
            error="Question cannot be empty.",
            context_found=False,
            sources=[],
            session_id=session_id,
            confidence_score=0.0,
            sources_searched=[]
        )
    
    try:
        # Import and use the new async QueryProcessor
        from ...processors.query_processor import get_query_processor
        
        processor = get_query_processor()
        
        # Use the new async processor with external search capabilities
        response = await processor.process_query_with_enhanced_timeout(
            question=user_question,
            session_id=session_id,
            user_id=user_id,
            search_scope=query.search_scope or "all",
            response_style=query.response_style or "balanced",
            use_enhanced_rag=query.use_enhanced_rag if query.use_enhanced_rag is not None else True,
            document_id=query.document_id,
            search_external=None,  # Let the processor decide based on query content
            progress_callback=None  # Could add WebSocket progress updates later
        )
        
        # FIXED: Better external API reporting
        external_api_used = any(
            'external' in source.lower() or 
            'comprehensive' in source.lower() or
            'congress' in source.lower() or
            'federal_register' in source.lower() or
            'harvard' in source.lower() or
            'courtlistener' in source.lower() or
            'justia' in source.lower()
            for source in response.sources_searched
        )
        
        # Count external sources in response
        external_sources_count = len([
            s for s in (response.sources or []) 
            if s.get('source_type') in ['external_legal', 'external_fast', 'external_api']
        ])
        
        logger.info(f"✅ Async query processing completed - External APIs: {'✅ Used' if external_api_used else '❌ Not used'}")
        logger.info(f"   Sources searched: {response.sources_searched}")
        logger.info(f"   External sources found: {external_sources_count}")
        logger.info(f"   Total sources in response: {len(response.sources or [])}")
        logger.info(f"   Confidence score: {response.confidence_score:.2f}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Async query processing failed: {e}")
        
        # Fallback to legacy processor if async fails
        logger.info("🔄 Falling back to legacy processor")
        try:
            from ...processors.query_processor import process_query
            
            response = process_query(
                user_question, 
                session_id, 
                user_id,
                query.search_scope or "all",
                query.response_style or "balanced",
                query.use_enhanced_rag if query.use_enhanced_rag is not None else True,
                query.document_id
            )
            
            # Add note about fallback
            if response.response:
                response.response += "\n\n*Note: Using basic search mode - advanced features temporarily unavailable.*"
            
            logger.info("🔄 Legacy processor completed successfully")
            
            return response
            
        except Exception as fallback_error:
            logger.error(f"❌ Fallback processor also failed: {fallback_error}")
            
            return QueryResponse(
                response="I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                error=f"Both processors failed: {str(e)} | {str(fallback_error)}",
                context_found=False,
                sources=[],
                session_id=session_id,
                confidence_score=0.0,
                sources_searched=["error_fallback"],
                retrieval_method="error"
            )

@router.post("/ask-debug", response_model=QueryResponse)
async def ask_question_debug(query: Query):
    """Debug version using async processor without authentication"""
    logger.info(f"Debug ask request received: {query}")
    
    cleanup_expired_conversations()
    
    session_id = query.session_id or str(uuid.uuid4())
    user_id = query.user_id or "debug_user"
    
    if session_id not in conversations:
        conversations[session_id] = {
            "messages": [],
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow()
        }
    else:
        conversations[session_id]["last_accessed"] = datetime.utcnow()
    
    user_question = query.question.strip()
    if not user_question:
        return QueryResponse(
            response=None,
            error="Question cannot be empty.",
            context_found=False,
            sources=[],
            session_id=session_id,
            confidence_score=0.0,
            sources_searched=[]
        )
    
    # Use async processor for debug endpoint too
    try:
        from ...processors.query_processor import get_query_processor
        
        processor = get_query_processor()
        
        response = await processor.process_query_with_enhanced_timeout(
            question=user_question,
            session_id=session_id,
            user_id=user_id,
            search_scope=query.search_scope or "all",
            response_style=query.response_style or "detailed",  # More detailed for debug
            use_enhanced_rag=query.use_enhanced_rag if query.use_enhanced_rag is not None else True,
            document_id=query.document_id,
            search_external=None,
            progress_callback=None
        )
        
        # FIXED: Better debug logging
        external_api_used = any(
            'external' in source.lower() or 
            'comprehensive' in source.lower() or
            'congress' in source.lower() or
            'federal_register' in source.lower() or
            'harvard' in source.lower() or
            'courtlistener' in source.lower() or
            'justia' in source.lower()
            for source in response.sources_searched
        )
        
        logger.info(f"🔬 Debug processing completed - External APIs: {'✅ Used' if external_api_used else '❌ Not used'}")
        logger.info(f"   Debug sources searched: {response.sources_searched}")
        logger.info(f"   Debug retrieval method: {response.retrieval_method}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Debug query processing failed: {e}")
        
        return QueryResponse(
            response=f"Debug processing failed: {str(e)}",
            error=str(e),
            context_found=False,
            sources=[],
            session_id=session_id,
            confidence_score=0.0,
            sources_searched=["debug_error"],
            retrieval_method="debug_error"
        )

@router.post("/ask-with-external", response_model=QueryResponse)
async def ask_question_force_external(query: Query, current_user: User = Depends(get_current_user)):
    """Force external database search for testing"""
    logger.info(f"Force external search request: {query}")
    
    session_id = query.session_id or str(uuid.uuid4())
    user_id = query.user_id or current_user.user_id
    
    # Initialize conversation if needed
    if session_id not in conversations:
        conversations[session_id] = {
            "messages": [],
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow()
        }
    else:
        conversations[session_id]["last_accessed"] = datetime.utcnow()
    
    try:
        from ...processors.query_processor import get_query_processor
        
        processor = get_query_processor()
        
        # Force external search
        response = await processor.process_query_with_enhanced_timeout(
            question=query.question,
            session_id=session_id,
            user_id=user_id,
            search_scope=query.search_scope or "all",
            response_style=query.response_style or "balanced",
            use_enhanced_rag=True,
            document_id=query.document_id,
            search_external=True,  # FORCE external search
            progress_callback=None
        )
        
        # Enhanced logging for forced external search
        external_sources_count = len([
            s for s in (response.sources or []) 
            if s.get('source_type') in ['external_legal', 'external_fast', 'external_api']
        ])
        
        logger.info(f"🚀 FORCED external search completed")
        logger.info(f"   Sources searched: {response.sources_searched}")
        logger.info(f"   External sources found: {external_sources_count}")
        logger.info(f"   Context found: {response.context_found}")
        logger.info(f"   Retrieval method: {response.retrieval_method}")
        
        # Add debug information to response
        if response.response:
            debug_info = f"""

--- DEBUG INFO ---
External APIs Used: {', '.join(response.sources_searched)}
External Sources Found: {external_sources_count}
Total Sources: {len(response.sources or [])}
Confidence: {response.confidence_score:.2f}
Method: {response.retrieval_method}
"""
            response.response += debug_info
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Force external query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search-status")
async def get_search_status():
    """Get current search system status"""
    try:
        from ...processors.query_processor import get_query_processor
        from ...services.external_db_service import get_fast_external_optimizer
        
        processor = get_query_processor()
        optimizer = get_fast_external_optimizer()
        
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "processor_stats": processor.get_processing_stats(),
            "external_optimizer": {
                "failed_apis": list(optimizer.failed_apis),
                "api_response_times": optimizer.api_response_times,
                "cached_queries": len(optimizer.api_cache)
            },
            "features_available": processor.features,
            "active_queries": len(processor.active_queries),
            "system_status": "operational"
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get search status: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "system_status": "error",
            "error": str(e)
        }

@router.post("/test-external-apis")
async def test_external_apis(query: str = "federal court contract law"):
    """Test external APIs directly"""
    logger.info(f"Testing external APIs with query: {query}")
    
    try:
        from ...services.external_db_service import search_external_databases, get_fast_external_optimizer
        
        # Test 1: Direct external database search
        logger.info("🔬 Testing direct external database search...")
        direct_results = search_external_databases(query, ["congress_gov", "federal_register"], None)
        
        # Test 2: Fast optimizer search
        logger.info("🚀 Testing fast optimizer search...")
        optimizer = get_fast_external_optimizer()
        
        import asyncio
        optimizer_results = await optimizer.search_external_fast(query, None)
        
        return {
            "test_query": query,
            "direct_search_results": len(direct_results),
            "optimizer_results": len(optimizer_results),
            "direct_results_sample": direct_results[:2] if direct_results else [],
            "optimizer_results_sample": optimizer_results[:2] if optimizer_results else [],
            "failed_apis": list(optimizer.failed_apis),
            "api_response_times": optimizer.api_response_times,
            "test_status": "success" if (direct_results or optimizer_results) else "no_results",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ External API test failed: {e}")
        return {
            "test_query": query,
            "test_status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
