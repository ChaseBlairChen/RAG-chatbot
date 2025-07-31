"""Query endpoints"""
import uuid
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException

from ...models import Query, QueryResponse, User
from ...core.security import get_current_user
from ...storage.managers import conversations, cleanup_expired_conversations
from ...processors.query_processor import process_query

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/ask", response_model=QueryResponse)
async def ask_question(query: Query, current_user: User = Depends(get_current_user)):
    """Enhanced ask endpoint with comprehensive analysis detection"""
    logger.info(f"Received ask request: {query}")
    
    cleanup_expired_conversations()
    
    session_id = query.session_id or str(uuid.uuid4())
    user_id = query.user_id or current_user.user_id
    
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
    
    response = process_query(
        user_question, 
        session_id, 
        user_id,
        query.search_scope or "all",
        query.response_style or "balanced",
        query.use_enhanced_rag if query.use_enhanced_rag is not None else True,
        query.document_id
    )
    return response

@router.post("/ask-debug", response_model=QueryResponse)
async def ask_question_debug(query: Query):
    """Debug version of ask endpoint without authentication"""
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
    
    response = process_query(
        user_question, 
        session_id, 
        user_id,
        query.search_scope or "all",
        query.response_style or "balanced",
        query.use_enhanced_rag if query.use_enhanced_rag is not None else True,
        query.document_id
    )
    return response
