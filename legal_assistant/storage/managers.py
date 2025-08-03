"""Global state management with document processing status and cleanup"""
import asyncio
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Global state stores
conversations: Dict[str, Dict] = {}
uploaded_files: Dict[str, Dict] = {}
user_sessions: Dict[str, Any] = {}
document_processing_status: Dict[str, Dict] = {}  # Track processing status

# Immigration-specific storage
immigration_cases: Dict[str, Any] = {}
deadline_alerts: Dict[str, Any] = {}
resource_library: Dict[str, Any] = {}
case_documents: Dict[str, List[str]] = {}  # case_id -> [document_ids]
translation_queue: Dict[str, Dict] = {}

# Database connection cache (for UserContainerManager integration)
database_cache: Dict[str, Any] = {}

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

# Document processing status management
def set_processing_status(file_id: str, status: str, progress: int = 0, details: str = ""):
    """Set processing status for a document"""
    document_processing_status[file_id] = {
        'status': status,
        'progress': progress,
        'details': details,
        'updated_at': datetime.utcnow().isoformat(),
        'started_at': document_processing_status.get(file_id, {}).get('started_at', datetime.utcnow().isoformat())
    }
    
    if status in ['completed', 'failed']:
        document_processing_status[file_id]['completed_at'] = datetime.utcnow().isoformat()

def get_processing_status(file_id: str) -> Optional[Dict]:
    """Get processing status for a document"""
    return document_processing_status.get(file_id)

def clear_processing_status(file_id: str):
    """Clear processing status for a document"""
    if file_id in document_processing_status:
        del document_processing_status[file_id]

# Database cache management
def cache_database_connection(user_id: str, db_connection: Any):
    """Cache a database connection for a user"""
    database_cache[user_id] = {
        'connection': db_connection,
        'cached_at': datetime.utcnow(),
        'last_used': datetime.utcnow()
    }
    logger.debug(f"Cached database connection for user {user_id}")

def get_cached_database(user_id: str) -> Optional[Any]:
    """Get cached database connection for a user"""
    if user_id in database_cache:
        cache_entry = database_cache[user_id]
        cache_entry['last_used'] = datetime.utcnow()
        logger.debug(f"Retrieved cached database connection for user {user_id}")
        return cache_entry['connection']
    return None

def clear_database_cache(user_id: str = None):
    """Clear database cache for specific user or all users"""
    if user_id:
        if user_id in database_cache:
            del database_cache[user_id]
            logger.info(f"Cleared database cache for user {user_id}")
    else:
        database_cache.clear()
        logger.info("Cleared all database cache")

def get_cache_info() -> Dict:
    """Get information about current cache state"""
    return {
        'cached_users': list(database_cache.keys()),
        'cache_size': len(database_cache),
        'total_conversations': len(conversations),
        'active_processing': len([s for s in document_processing_status.values() if s.get('status') == 'processing']),
        'user_sessions': len(user_sessions)
    }

# Immigration case management
def create_immigration_case(case_id: str, case_data: Dict):
    """Create a new immigration case"""
    immigration_cases[case_id] = {
        **case_data,
        'created_at': datetime.utcnow().isoformat(),
        'updated_at': datetime.utcnow().isoformat(),
        'status': 'active'
    }
    case_documents[case_id] = []

def update_immigration_case(case_id: str, updates: Dict):
    """Update an immigration case"""
    if case_id in immigration_cases:
        immigration_cases[case_id].update(updates)
        immigration_cases[case_id]['updated_at'] = datetime.utcnow().isoformat()

def get_immigration_case(case_id: str) -> Optional[Dict]:
    """Get an immigration case"""
    return immigration_cases.get(case_id)

def add_case_document(case_id: str, document_id: str):
    """Add document to a case"""
    if case_id not in case_documents:
        case_documents[case_id] = []
    if document_id not in case_documents[case_id]:
        case_documents[case_id].append(document_id)

def get_case_documents(case_id: str) -> List[str]:
    """Get documents for a case"""
    return case_documents.get(case_id, [])

# Deadline alerts management
def add_deadline_alert(alert_id: str, case_id: str, deadline: datetime, alert_type: str, description: str):
    """Add a deadline alert"""
    deadline_alerts[alert_id] = {
        'case_id': case_id,
        'deadline': deadline.isoformat(),
        'alert_type': alert_type,
        'description': description,
        'created_at': datetime.utcnow().isoformat(),
        'status': 'active'
    }

def get_upcoming_deadlines(days_ahead: int = 30) -> List[Dict]:
    """Get upcoming deadlines"""
    cutoff_date = datetime.utcnow() + timedelta(days=days_ahead)
    upcoming = []
    
    for alert_id, alert in deadline_alerts.items():
        if alert['status'] == 'active':
            deadline = datetime.fromisoformat(alert['deadline'])
            if deadline <= cutoff_date:
                upcoming.append({
                    'alert_id': alert_id,
                    **alert,
                    'days_until': (deadline - datetime.utcnow()).days
                })
    
    return sorted(upcoming, key=lambda x: x['deadline'])

# Translation queue management
def add_to_translation_queue(task_id: str, source_text: str, source_lang: str, target_lang: str, priority: str = 'normal'):
    """Add translation task to queue"""
    translation_queue[task_id] = {
        'source_text': source_text,
        'source_lang': source_lang,
        'target_lang': target_lang,
        'priority': priority,
        'status': 'queued',
        'created_at': datetime.utcnow().isoformat(),
        'updated_at': datetime.utcnow().isoformat()
    }

def update_translation_status(task_id: str, status: str, translated_text: str = None):
    """Update translation task status"""
    if task_id in translation_queue:
        translation_queue[task_id]['status'] = status
        translation_queue[task_id]['updated_at'] = datetime.utcnow().isoformat()
        if translated_text:
            translation_queue[task_id]['translated_text'] = translated_text
            translation_queue[task_id]['completed_at'] = datetime.utcnow().isoformat()

# Session management
def create_user_session(session_id: str, user_data: Dict):
    """Create a user session"""
    user_sessions[session_id] = {
        **user_data,
        'created_at': datetime.utcnow(),
        'last_active': datetime.utcnow(),
        'status': 'active'
    }

def update_session_activity(session_id: str):
    """Update session last activity"""
    if session_id in user_sessions:
        user_sessions[session_id]['last_active'] = datetime.utcnow()

def get_user_session(session_id: str) -> Optional[Dict]:
    """Get user session"""
    if session_id in user_sessions:
        update_session_activity(session_id)
        return user_sessions[session_id]
    return None

# Comprehensive cleanup function
async def cleanup_old_data():
    """Periodic cleanup of old data"""
    while True:
        try:
            logger.info("Starting periodic cleanup...")
            
            # Clean up old processing status (24 hours)
            cutoff_processing = datetime.utcnow() - timedelta(hours=24)
            cleaned_processing = 0
            for file_id, status in list(document_processing_status.items()):
                if status.get('completed_at'):
                    completed = datetime.fromisoformat(status['completed_at'])
                    if completed < cutoff_processing:
                        del document_processing_status[file_id]
                        cleaned_processing += 1
            
            # Clean up old sessions (8 hours)
            cutoff_sessions = datetime.utcnow() - timedelta(hours=8)
            cleaned_sessions = 0
            for session_id, data in list(user_sessions.items()):
                last_active = data.get('last_active', datetime.utcnow())
                if last_active < cutoff_sessions:
                    del user_sessions[session_id]
                    cleaned_sessions += 1
            
            # Clean up old conversations (1 hour)
            cleanup_expired_conversations()
            
            # Clean up old database cache (2 hours unused)
            cutoff_cache = datetime.utcnow() - timedelta(hours=2)
            cleaned_cache = 0
            for user_id, cache_data in list(database_cache.items()):
                last_used = cache_data.get('last_used', datetime.utcnow())
                if last_used < cutoff_cache:
                    del database_cache[user_id]
                    cleaned_cache += 1
            
            # Clean up completed translation tasks (48 hours)
            cutoff_translation = datetime.utcnow() - timedelta(hours=48)
            cleaned_translations = 0
            for task_id, task in list(translation_queue.items()):
                if task.get('status') == 'completed' and task.get('completed_at'):
                    completed = datetime.fromisoformat(task['completed_at'])
                    if completed < cutoff_translation:
                        del translation_queue[task_id]
                        cleaned_translations += 1
            
            # Clean up old deadline alerts (past deadlines older than 30 days)
            cutoff_alerts = datetime.utcnow() - timedelta(days=30)
            cleaned_alerts = 0
            for alert_id, alert in list(deadline_alerts.items()):
                deadline = datetime.fromisoformat(alert['deadline'])
                if deadline < cutoff_alerts:
                    del deadline_alerts[alert_id]
                    cleaned_alerts += 1
            
            # Log cleanup results
            if any([cleaned_processing, cleaned_sessions, cleaned_cache, cleaned_translations, cleaned_alerts]):
                logger.info(f"Cleanup completed: "
                          f"Processing: {cleaned_processing}, "
                          f"Sessions: {cleaned_sessions}, "
                          f"Cache: {cleaned_cache}, "
                          f"Translations: {cleaned_translations}, "
                          f"Alerts: {cleaned_alerts}")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        
        # Wait 1 hour before next cleanup
        await asyncio.sleep(3600)

# Statistics and monitoring
def get_system_stats() -> Dict:
    """Get comprehensive system statistics"""
    now = datetime.utcnow()
    
    # Processing status stats
    processing_stats = {}
    for status in document_processing_status.values():
        status_type = status.get('status', 'unknown')
        processing_stats[status_type] = processing_stats.get(status_type, 0) + 1
    
    # Session activity stats
    active_sessions = sum(1 for session in user_sessions.values() 
                         if now - session.get('last_active', now) < timedelta(hours=1))
    
    # Immigration case stats
    case_stats = {}
    for case in immigration_cases.values():
        case_status = case.get('status', 'unknown')
        case_stats[case_status] = case_stats.get(case_status, 0) + 1
    
    # Translation queue stats
    translation_stats = {}
    for task in translation_queue.values():
        task_status = task.get('status', 'unknown')
        translation_stats[task_status] = translation_stats.get(task_status, 0) + 1
    
    return {
        'timestamp': now.isoformat(),
        'conversations': {
            'total': len(conversations),
            'messages': sum(len(conv['messages']) for conv in conversations.values())
        },
        'document_processing': processing_stats,
        'sessions': {
            'total': len(user_sessions),
            'active': active_sessions
        },
        'database_cache': {
            'cached_connections': len(database_cache)
        },
        'immigration_cases': case_stats,
        'translation_queue': translation_stats,
        'deadline_alerts': {
            'total': len(deadline_alerts),
            'active': sum(1 for alert in deadline_alerts.values() if alert.get('status') == 'active')
        },
        'uploaded_files': len(uploaded_files)
    }

# Initialization function
def initialize_storage_manager():
    """Initialize the storage manager and start cleanup task"""
    logger.info("Initializing storage manager...")
    
    # Start the cleanup task
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create task
            asyncio.create_task(cleanup_old_data())
        else:
            # If no loop is running, start one
            asyncio.run(cleanup_old_data())
    except RuntimeError:
        # Handle case where event loop is already running
        asyncio.create_task(cleanup_old_data())
    
    logger.info("Storage manager initialized with cleanup task")

# Emergency cleanup function
def emergency_cleanup():
    """Emergency cleanup function for critical memory situations"""
    logger.warning("Performing emergency cleanup...")
    
    # Clear all non-essential caches
    database_cache.clear()
    
    # Keep only recent conversations (last 30 minutes)
    cutoff = datetime.utcnow() - timedelta(minutes=30)
    recent_conversations = {
        session_id: data for session_id, data in conversations.items()
        if data['last_accessed'] > cutoff
    }
    conversations.clear()
    conversations.update(recent_conversations)
    
    # Clear completed processing status
    active_processing = {
        file_id: status for file_id, status in document_processing_status.items()
        if status.get('status') not in ['completed', 'failed']
    }
    document_processing_status.clear()
    document_processing_status.update(active_processing)
    
    # Clear old translation tasks
    active_translations = {
        task_id: task for task_id, task in translation_queue.items()
        if task.get('status') in ['queued', 'processing']
    }
    translation_queue.clear()
    translation_queue.update(active_translations)
    
    logger.warning("Emergency cleanup completed")
