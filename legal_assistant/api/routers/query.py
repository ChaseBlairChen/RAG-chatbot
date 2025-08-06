# legal_assistant/api/routers/query.py - UPDATED FOR ASYNC PROCESSOR
"""Query endpoints - Updated to use new async QueryProcessor with external search capabilities"""
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
        
        logger.info(f"✅ Async query processing completed - External APIs: {'✅ Used' if 'external' in response.sources_searched else '❌ Not used'}")
        
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

# Add new endpoint for testing external search specifically
@router.post("/ask-with-external", response_model=QueryResponse)
async def ask_question_force_external(query: Query, current_user: User = Depends(get_current_user)):
    """Force external database search for testing"""
    logger.info(f"Force external search request: {query}")
    
    session_id = query.session_id or str(uuid.uuid4())
    user_id = query.user_id or current_user.user_id
    
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
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Force external query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
