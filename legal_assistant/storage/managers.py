"""Global state management with document processing status"""
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Global state stores
conversations: Dict[str, Dict] = {}
uploaded_files: Dict[str, Dict] = {}
user_sessions: Dict[str, Any] = {}
document_processing_status: Dict[str, Dict] = {}  # ADD THIS LINE - NEW: Track processing status

def add_to_conversation(session_id: str, role: str, content: str, sources: Optional[List] = None):
    """Add message to conversation"""
    if session_id not in conversations:
        conversations[session_id] = {
            'messages': [],
            'created_at': datetime.utcnow(),
            'last_accessed': datetime.utcnow()
        }
    
    message = {
        'role': role,
        'content': content,
        'timestamp': datetime.utcnow().isoformat(),
        'sources': sources or []
    }
    
    conversations[session_id]['messages'].append(message)
    conversations[session_id]['last_accessed'] = datetime.utcnow()

def get_conversation_context(session_id: str, max_length: int = 2000) -> str:
    """Get conversation context for a session"""
    if session_id not in conversations:
        return ""
    
    messages = conversations[session_id]['messages']
    context_parts = []
    recent_messages = messages[-4:]
    
    for msg in recent_messages:
        role = msg['role'].upper()
        content = msg['content']
        if len(content) > 800:
            content = content[:800] + "..."
        context_parts.append(f"{role}: {content}")
    
    if context_parts:
        return "Previous conversation:\n" + "\n".join(context_parts)
    return ""

def cleanup_expired_conversations():
    """Clean up expired conversations"""
    now = datetime.utcnow()
    expired_sessions = [
        session_id for session_id, data in conversations.items()
        if now - data['last_accessed'] > timedelta(hours=1)
    ]
    for session_id in expired_sessions:
        del conversations[session_id]
    if expired_sessions:
        logger.info(f"Cleaned up {len(expired_sessions)} expired conversations")
