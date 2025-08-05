
"""
Enhanced storage managers with NoSQL support and fallback compatibility.
Maintains exact same interface as your current in-memory managers.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
import logging  # FIXED: Added this missing import

from .nosql_models import (
    UserDocument, UploadedFileDocument, ProcessingStatusDocument,
    ConversationDocument, ImmigrationCaseDocument
)
from .nosql_manager import get_nosql_manager

logger = logging.getLogger(__name__)

class EnhancedStorageManager:
    """
    Enhanced storage manager with NoSQL support and in-memory fallback.
    Maintains backward compatibility with existing code.
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
        """Initialize NoSQL connections"""
        if not self._initialized:
            self.nosql_manager = await get_nosql_manager()
            self._initialized = True
            
            # Migrate existing in-memory data if any
            await self._migrate_existing_data()
    
    async def _migrate_existing_data(self):
        """Migrate existing in-memory data to NoSQL"""
        if not self.nosql_manager.mongodb_available:
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
        if self.nosql_manager and self.nosql_manager.mongodb_available:
            try:
                user_doc = await UserDocument.find_one(UserDocument.user_id == user_id)
                if user_doc:
                    return user_doc.dict()
            except Exception as e:
                logger.error(f"MongoDB user lookup failed: {e}")
        
        # Fallback to in-memory
        return self.user_sessions.get(user_id)
    
    async def save_user(self, user_id: str, user_data: Dict):
        """Save user data with NoSQL/fallback support"""
        if self.nosql_manager and self.nosql_manager.mongodb_available:
            try:
                user_doc = await UserDocument.find_one(UserDocument.user_id == user_id)
                if user_doc:
                    # Update existing
                    for key, value in user_data.items():
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
        if self.nosql_manager and self.nosql_manager.mongodb_available:
            try:
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
    
    async def get_uploaded_file(self, file_id: str) -> Optional[Dict]:
        """Get uploaded file metadata"""
        if self.nosql_manager and self.nosql_manager.mongodb_available:
            try:
                file_doc = await UploadedFileDocument.find_one(UploadedFileDocument.file_id == file_id)
                if file_doc:
                    return file_doc.dict()
            except Exception as e:
                logger.error(f"MongoDB file lookup failed: {e}")
        
        # Fallback to in-memory
        return self.uploaded_files.get(file_id)
    
    async def get_user_files(self, user_id: str) -> List[Dict]:
        """Get all files for a user"""
        if self.nosql_manager and self.nosql_manager.mongodb_available:
            try:
                file_docs = await UploadedFileDocument.find(
                    UploadedFileDocument.user_id == user_id
                ).sort(-UploadedFileDocument.uploaded_at).to_list()
                
                return [doc.dict() for doc in file_docs]
                
            except Exception as e:
                logger.error(f"MongoDB user files lookup failed: {e}")
        
        # Fallback to in-memory
        user_files = []
        for file_id, file_data in self.uploaded_files.items():
            if file_data.get('user_id') == user_id:
                user_files.append({**file_data, 'file_id': file_id})
        
        # Sort by upload date
        user_files.sort(key=lambda x: x.get('uploaded_at', datetime.min), reverse=True)
        return user_files
    
    async def delete_uploaded_file(self, file_id: str) -> bool:
        """Delete uploaded file metadata"""
        deleted = False
        
        if self.nosql_manager and self.nosql_manager.mongodb_available:
            try:
                result = await UploadedFileDocument.find_one(
                    UploadedFileDocument.file_id == file_id
                ).delete()
                deleted = result.deleted_count > 0
                
            except Exception as e:
                logger.error(f"MongoDB file delete failed: {e}")
        
        # Also remove from in-memory
        if file_id in self.uploaded_files:
            del self.uploaded_files[file_id]
            deleted = True
        
        return deleted
    
    # === PROCESSING STATUS MANAGEMENT ===
    
    async def set_processing_status(self, file_id: str, status: str, progress: int = 0,
                                  message: str = "", details: Dict = None, error: str = None):
        """Set processing status with NoSQL/fallback support"""
        
        status_data = {
            'file_id': file_id,
            'status': status,
            'progress': progress,
            'message': message,
            'details': details or {},
            'updated_at': datetime.utcnow()
        }
        
        if error:
            status_data['error'] = error
            status_data['failed_at'] = datetime.utcnow()
        
        if status == 'completed':
            status_data['completed_at'] = datetime.utcnow()
        
        if self.nosql_manager and self.nosql_manager.mongodb_available:
            try:
                # Upsert processing status
                existing = await ProcessingStatusDocument.find_one(
                    ProcessingStatusDocument.file_id == file_id
                )
                
                if existing:
                    for key, value in status_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    await existing.save()
                else:
                    status_doc = ProcessingStatusDocument(**status_data)
                    await status_doc.save()
                
            except Exception as e:
                logger.error(f"MongoDB status save failed: {e}")
                # Fall back to in-memory
                self.document_processing_status[file_id] = status_data
        else:
            # Use in-memory storage
            self.document_processing_status[file_id] = status_data
    
    async def get_processing_status(self, file_id: str) -> Optional[Dict]:
        """Get processing status"""
        if self.nosql_manager and self.nosql_manager.mongodb_available:
            try:
                status_doc = await ProcessingStatusDocument.find_one(
                    ProcessingStatusDocument.file_id == file_id
                )
                if status_doc:
                    return status_doc.dict()
            except Exception as e:
                logger.error(f"MongoDB status lookup failed: {e}")
        
        # Fallback to in-memory
        return self.document_processing_status.get(file_id)
    
    # === CONVERSATION MANAGEMENT ===
    
    async def add_to_conversation(self, session_id: str, role: str, content: str, sources: Optional[List] = None):
        """Add message to conversation with NoSQL/fallback support"""
        
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.utcnow().isoformat(),
            'sources': sources or []
        }
        
        if self.nosql_manager and self.nosql_manager.mongodb_available:
            try:
                conv_doc = await ConversationDocument.find_one(
                    ConversationDocument.session_id == session_id
                )
                
                if conv_doc:
                    conv_doc.messages.append(message)
                    conv_doc.last_accessed = datetime.utcnow()
                    await conv_doc.save()
                else:
                    # Create new conversation
                    new_conv = ConversationDocument(
                        session_id=session_id,
                        messages=[message],
                        created_at=datetime.utcnow(),
                        last_accessed=datetime.utcnow()
                    )
                    await new_conv.save()
                
            except Exception as e:
                logger.error(f"MongoDB conversation save failed: {e}")
                # Fall back to in-memory
                if session_id not in self.conversations:
                    self.conversations[session_id] = {
                        'messages': [],
                        'created_at': datetime.utcnow(),
                        'last_accessed': datetime.utcnow()
                    }
                self.conversations[session_id]['messages'].append(message)
                self.conversations[session_id]['last_accessed'] = datetime.utcnow()
        else:
            # Use in-memory storage
            if session_id not in self.conversations:
                self.conversations[session_id] = {
                    'messages': [],
                    'created_at': datetime.utcnow(),
                    'last_accessed': datetime.utcnow()
                }
            self.conversations[session_id]['messages'].append(message)
            self.conversations[session_id]['last_accessed'] = datetime.utcnow()
    
    async def get_conversation_context(self, session_id: str, max_length: int = 2000) -> str:
        """Get conversation context"""
        messages = []
        
        if self.nosql_manager and self.nosql_manager.mongodb_available:
            try:
                conv_doc = await ConversationDocument.find_one(
                    ConversationDocument.session_id == session_id
                )
                if conv_doc:
                    messages = conv_doc.messages
            except Exception as e:
                logger.error(f"MongoDB conversation lookup failed: {e}")
        
        # Fallback to in-memory
        if not messages and session_id in self.conversations:
            messages = self.conversations[session_id]['messages']
        
        if not messages:
            return ""
        
        # Format recent messages
        context_parts = []
        recent_messages = messages[-4:]  # Last 4 messages
        
        for msg in recent_messages:
            role = msg['role'].upper()
            content = msg['content']
            if len(content) > 800:
                content = content[:800] + "..."
            context_parts.append(f"{role}: {content}")
        
        if context_parts:
            return "Previous conversation:\n" + "\n".join(context_parts)
        return ""
    
    # === REDIS CACHING METHODS ===
    
    async def cache_set(self, key: str, value: Any, ttl: int = 3600):
        """Set cache value with TTL"""
        if self.nosql_manager and self.nosql_manager.redis_available:
            try:
                import json
                await self.nosql_manager.redis_client.set(
                    key, json.dumps(value), ex=ttl
                )
            except Exception as e:
                logger.error(f"Redis cache set failed: {e}")
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        if self.nosql_manager and self.nosql_manager.redis_available:
            try:
                import json
                data = await self.nosql_manager.redis_client.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.error(f"Redis cache get failed: {e}")
        return None
    
    # === CLEANUP METHODS ===
    
    async def cleanup_old_data(self):
        """Clean up old data from NoSQL and in-memory storage"""
        
        if self.nosql_manager and self.nosql_manager.mongodb_available:
            try:
                # Clean up old conversations (older than 1 hour)
                cutoff = datetime.utcnow() - timedelta(hours=1)
                await ConversationDocument.find(
                    ConversationDocument.last_accessed < cutoff
                ).delete()
                
                # Clean up old processing status (older than 24 hours)
                status_cutoff = datetime.utcnow() - timedelta(hours=24)
                await ProcessingStatusDocument.find(
                    ProcessingStatusDocument.started_at < status_cutoff,
                    ProcessingStatusDocument.status.in_(["completed", "failed"])
                ).delete()
                
                logger.info("✅ MongoDB cleanup completed")
                
            except Exception as e:
                logger.error(f"MongoDB cleanup failed: {e}")
        
        # Also clean up in-memory storage
        await self._cleanup_memory_storage()
    
    async def _cleanup_memory_storage(self):
        """Clean up in-memory storage"""
        try:
            # Clean up old conversations
            cutoff = datetime.utcnow() - timedelta(hours=1)
            expired_sessions = [
                session_id for session_id, data in self.conversations.items()
                if data.get('last_accessed', datetime.utcnow()) < cutoff
            ]
            for session_id in expired_sessions:
                del self.conversations[session_id]
            
            # Clean up old processing status
            status_cutoff = datetime.utcnow() - timedelta(hours=24)
            expired_status = [
                file_id for file_id, status in self.document_processing_status.items()
                if status.get('completed_at') and 
                datetime.fromisoformat(status['completed_at']) < status_cutoff
            ]
            for file_id in expired_status:
                del self.document_processing_status[file_id]
            
            if expired_sessions or expired_status:
                logger.info(f"Cleaned up {len(expired_sessions)} conversations and {len(expired_status)} statuses")
                
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
    
    # === STATISTICS AND MONITORING ===
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            'timestamp': datetime.utcnow().isoformat(),
            'storage_backend': 'mongodb' if self.nosql_manager and self.nosql_manager.mongodb_available else 'memory',
            'redis_available': self.nosql_manager and self.nosql_manager.redis_available
        }
        
        if self.nosql_manager and self.nosql_manager.mongodb_available:
            try:
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

# Global enhanced storage manager
_enhanced_storage = None

async def get_enhanced_storage() -> EnhancedStorageManager:
    """Get or create enhanced storage manager"""
    global _enhanced_storage
    if _enhanced_storage is None:
        _enhanced_storage = EnhancedStorageManager()
        await _enhanced_storage.initialize()
    return _enhanced_storage

# === BACKWARD COMPATIBLE FUNCTIONS ===
# These maintain the same interface your existing code expects

# Global variables for backward compatibility (will be populated from NoSQL)
conversations: Dict[str, Dict] = {}
uploaded_files: Dict[str, Dict] = {}
user_sessions: Dict[str, Any] = {}
document_processing_status: Dict[str, Dict] = {}
immigration_cases: Dict[str, Any] = {}
deadline_alerts: Dict[str, Any] = {}

async def _sync_from_nosql():
    """Sync data from NoSQL to global variables for backward compatibility"""
    try:
        storage = await get_enhanced_storage()
        
        # Sync key data structures
        global conversations, uploaded_files, user_sessions, document_processing_status
        conversations = storage.conversations
        uploaded_files = storage.uploaded_files
        user_sessions = storage.user_sessions
        document_processing_status = storage.document_processing_status
        
    except Exception as e:
        logger.error(f"NoSQL sync failed: {e}")

def add_to_conversation(session_id: str, role: str, content: str, sources: Optional[List] = None):
    """BACKWARD COMPATIBLE: Add message to conversation"""
    # Create async task for NoSQL operation
    async def _async_add():
        storage = await get_enhanced_storage()
        await storage.add_to_conversation(session_id, role, content, sources)
    
    # Run async operation
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(_async_add())
        else:
            asyncio.run(_async_add())
    except Exception as e:
        logger.error(f"Async conversation add failed: {e}")
        # Fallback to direct in-memory update
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
    # Try to get from enhanced storage
    try:
        async def _async_get():
            storage = await get_enhanced_storage()
            return await storage.get_conversation_context(session_id, max_length)
        
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Can't await in sync context, use fallback
            pass
        else:
            return asyncio.run(_async_get())
    except:
        pass
    
    # Fallback to in-memory
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
    async def _async_cleanup():
        storage = await get_enhanced_storage()
        await storage.cleanup_old_data()
    
    try:
        asyncio.create_task(_async_cleanup())
    except Exception as e:
        logger.error(f"Async cleanup failed: {e}")
        # Fallback to in-memory cleanup
        now = datetime.utcnow()
        expired_sessions = [
            session_id for session_id, data in conversations.items()
            if now - data.get('last_accessed', now) > timedelta(hours=1)
        ]
        for session_id in expired_sessions:
            del conversations[session_id]



# Immigration storage
immigration_cases: Dict[str, Any] = {}
case_documents: Dict[str, List[str]] = {}
deadline_alerts: Dict[str, Any] = {}
resource_library: Dict[str, Any] = {}
