"""
FIXED: Enhanced storage managers with lazy imports to prevent startup crashes
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
import logging

logger = logging.getLogger(__name__)

class EnhancedStorageManager:
    """
    Enhanced storage manager with NoSQL support and in-memory fallback.
    FIXED: Lazy imports to prevent startup crashes.
    """
    
    def __init__(self):
        self.nosql_manager = None
        self._initialized = False
        
        # Maintain in-memory storage for compatibility
        self.conversations = {}
        self.uploaded_files = {}
        self.user_sessions = {}
        self.document_processing_status = {}
        self.immigration_cases = {}
        self.deadline_alerts = {}
    
    async def initialize(self):
        """Initialize NoSQL connections with error handling"""
        if not self._initialized:
            try:
                # FIXED: Import nosql_manager safely
                from .nosql_manager import get_nosql_manager
                self.nosql_manager = await get_nosql_manager()
                self._initialized = True
                
                # Migrate existing in-memory data if any
                await self._migrate_existing_data()
                
                logger.info("✅ Enhanced storage initialized successfully")
                
            except Exception as e:
                logger.warning(f"⚠️ NoSQL initialization failed: {e}")
                logger.info("🔄 Continuing with in-memory storage only")
                self._initialized = True  # Mark as initialized even without NoSQL
    
    async def _migrate_existing_data(self):
        """Migrate existing in-memory data to NoSQL"""
        if not (self.nosql_manager and getattr(self.nosql_manager, 'mongodb_available', False)):
            return
        
        try:
            # Migrate uploaded files
            for file_id, file_data in self.uploaded_files.items():
                await self.save_uploaded_file(file_id, file_data)
            
            # Migrate conversations
            for session_id, conv_data in self.conversations.items():
                await self.save_conversation(session_id, conv_data)
            
            logger.info("✅ Migrated existing data to MongoDB")
            
        except Exception as e:
            logger.error(f"Data migration failed: {e}")
    
    # === USER MANAGEMENT ===
    
    async def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user data with NoSQL/fallback support"""
        if self.nosql_manager and getattr(self.nosql_manager, 'mongodb_available', False):
            try:
                # FIXED: Import only when needed
                from .nosql_models import UserDocument
                
                user_doc = await UserDocument.find_one(UserDocument.user_id == user_id)
                if user_doc:
                    return user_doc.dict()
            except Exception as e:
                logger.error(f"MongoDB user lookup failed: {e}")
        
        # Fallback to in-memory
        return self.user_sessions.get(user_id)
    
    async def save_user(self, user_id: str, user_data: Dict):
        """Save user data with NoSQL/fallback support"""
        if self.nosql_manager and getattr(self.nosql_manager, 'mongodb_available', False):
            try:
                # FIXED: Import only when needed
                from .nosql_models import UserDocument
                
                user_doc = await UserDocument.find_one(UserDocument.user_id == user_id)
                if user_doc:
                    # Update existing
                    for key, value in user_data.items():
                        if hasattr(user_doc, key):
                            setattr(user_doc, key, value)
                    user_doc.last_active = datetime.utcnow()
                    await user_doc.save()
                else:
                    # Create new
                    new_user = UserDocument(user_id=user_id, **user_data)
                    await new_user.save()
                
                logger.debug(f"User {user_id} saved to MongoDB")
                
            except Exception as e:
                logger.error(f"MongoDB user save failed: {e}")
                # Fall back to in-memory
                self.user_sessions[user_id] = user_data
        else:
            # Use in-memory storage
            self.user_sessions[user_id] = user_data
    
    # === DOCUMENT MANAGEMENT ===
    
    async def save_uploaded_file(self, file_id: str, file_data: Dict):
        """Save uploaded file metadata with NoSQL/fallback support"""
        if self.nosql_manager and getattr(self.nosql_manager, 'mongodb_available', False):
            try:
                # FIXED: Import only when needed
                from .nosql_models import UploadedFileDocument
                
                # Check if exists
                existing = await UploadedFileDocument.find_one(UploadedFileDocument.file_id == file_id)
                
                if existing:
                    # Update existing
                    for key, value in file_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    await existing.save()
                else:
                    # Create new
                    file_doc = UploadedFileDocument(file_id=file_id, **file_data)
                    await file_doc.save()
                
                logger.debug(f"File {file_id} saved to MongoDB")
                
            except Exception as e:
                logger.error(f"MongoDB file save failed: {e}")
                # Fall back to in-memory
                self.uploaded_files[file_id] = file_data
        else:
            # Use in-memory storage
            self.uploaded_files[file_id] = file_data
    
    # === STATISTICS AND MONITORING ===
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            'timestamp': datetime.utcnow().isoformat(),
            'storage_backend': 'mongodb' if self.nosql_manager and getattr(self.nosql_manager, 'mongodb_available', False) else 'memory',
            'redis_available': self.nosql_manager and getattr(self.nosql_manager, 'redis_available', False)
        }
        
        if self.nosql_manager and getattr(self.nosql_manager, 'mongodb_available', False):
            try:
                # FIXED: Import only when needed
                from .nosql_models import (
                    UserDocument, UploadedFileDocument, ProcessingStatusDocument,
                    ConversationDocument, ImmigrationCaseDocument
                )
                
                # MongoDB statistics
                stats['mongodb_stats'] = {
                    'users': await UserDocument.count(),
                    'uploaded_files': await UploadedFileDocument.count(),
                    'active_processing': await ProcessingStatusDocument.find(
                        ProcessingStatusDocument.status == "processing"
                    ).count(),
                    'conversations': await ConversationDocument.count(),
                    'immigration_cases': await ImmigrationCaseDocument.count()
                }
                
            except Exception as e:
                logger.error(f"MongoDB stats failed: {e}")
                stats['mongodb_error'] = str(e)
        
        # In-memory statistics
        stats['memory_stats'] = {
            'conversations': len(self.conversations),
            'uploaded_files': len(self.uploaded_files),
            'user_sessions': len(self.user_sessions),
            'processing_status': len(self.document_processing_status)
        }
        
        return stats

# FIXED: More robust enhanced storage getter
async def get_enhanced_storage() -> EnhancedStorageManager:
    """Get or create enhanced storage manager with error handling"""
    global _enhanced_storage
    if _enhanced_storage is None:
        _enhanced_storage = EnhancedStorageManager()
        try:
            await _enhanced_storage.initialize()
        except Exception as e:
            logger.error(f"Storage initialization failed: {e}")
            # Continue with in-memory only
    return _enhanced_storage

# === BACKWARD COMPATIBLE FUNCTIONS ===

# Global variables for backward compatibility
conversations: Dict[str, Dict] = {}
uploaded_files: Dict[str, Dict] = {}
user_sessions: Dict[str, Any] = {}
document_processing_status: Dict[str, Dict] = {}
immigration_cases: Dict[str, Any] = {}
deadline_alerts: Dict[str, Any] = {}

def add_to_conversation(session_id: str, role: str, content: str, sources: Optional[List] = None):
    """BACKWARD COMPATIBLE: Add message to conversation"""
    # FIXED: Simpler approach that works without async issues
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
    """BACKWARD COMPATIBLE: Get conversation context"""
    # FIXED: Simpler approach that works reliably
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
    """BACKWARD COMPATIBLE: Clean up expired conversations"""
    # FIXED: Simple cleanup that works without async
    now = datetime.utcnow()
    expired_sessions = [
        session_id for session_id, data in conversations.items()
        if now - data.get('last_accessed', now) > timedelta(hours=1)
    ]
    for session_id in expired_sessions:
        del conversations[session_id]
    
    if expired_sessions:
        logger.info(f"Cleaned up {len(expired_sessions)} expired conversations")

# Global storage reference
_enhanced_storage = None
