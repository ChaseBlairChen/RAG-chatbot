# legal_assistant/storage/redis_adapter.py
"""
Redis compatibility adapter to bridge new Redis-based processing 
with existing in-memory storage system.
"""
import json
import logging
from typing import Dict, Optional, Any
from datetime import datetime
import redis
from ..config import FeatureFlags

logger = logging.getLogger(__name__)

class RedisAdapter:
    """Adapter to handle both Redis and in-memory storage"""
    
    def __init__(self):
        self.redis_available = False
        self.redis_client = None
        self._fallback_storage = {}  # In-memory fallback
        
        # Try to connect to Redis
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection with fallback"""
        try:
            self.redis_client = redis.Redis.from_url(
                "redis://localhost:6379/0", 
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            self.redis_available = True
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.warning(f"⚠️ Redis not available, using in-memory storage: {e}")
            self.redis_available = False
    
    def set_status(self, file_id: str, status_data: Dict[str, Any], ttl: int = 86400):
        """Set processing status (Redis or in-memory)"""
        try:
            if self.redis_available:
                # Store in Redis
                self.redis_client.set(
                    f"doc_status:{file_id}", 
                    json.dumps(status_data), 
                    ex=ttl
                )
            else:
                # Store in memory
                self._fallback_storage[file_id] = status_data
        except Exception as e:
            logger.error(f"Failed to set status for {file_id}: {e}")
            # Always fallback to memory
            self._fallback_storage[file_id] = status_data
    
    def get_status(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get processing status (Redis or in-memory)"""
        try:
            if self.redis_available:
                data = self.redis_client.get(f"doc_status:{file_id}")
                if data:
                    return json.loads(data)
            
            # Check in-memory fallback
            return self._fallback_storage.get(file_id)
            
        except Exception as e:
            logger.error(f"Failed to get status for {file_id}: {e}")
            # Fallback to in-memory
            return self._fallback_storage.get(file_id)
    
    def delete_status(self, file_id: str):
        """Delete processing status"""
        try:
            if self.redis_available:
                self.redis_client.delete(f"doc_status:{file_id}")
            
            # Also remove from memory
            self._fallback_storage.pop(file_id, None)
            
        except Exception as e:
            logger.error(f"Failed to delete status for {file_id}: {e}")

# Global adapter instance
_redis_adapter = None

def get_redis_adapter() -> RedisAdapter:
    """Get or create global Redis adapter"""
    global _redis_adapter
    if _redis_adapter is None:
        _redis_adapter = RedisAdapter()
    return _redis_adapter

# --- Updated document_tasks.py that works with your current app ---

"""
Updated Background tasks for document processing - Compatible with existing app
"""
import logging
import time
from datetime import datetime
from typing import Dict, Any

from ..services.document_processor import SafeDocumentProcessor
from ..services.container_manager import get_container_manager
from ..storage.managers import uploaded_files, document_processing_status
from .redis_adapter import get_redis_adapter

logger = logging.getLogger(__name__)

async def process_document_background(
    file_id: str,
    file_content: bytes,
    file_ext: str,
    filename: str,
    user_id: str
):
    """
    IMPROVED: Process document with Redis + in-memory compatibility
    """
    redis_adapter = get_redis_adapter()
    start_time = time.time()
    
    try:
        # Step 1: Initial status - Extracting
        initial_status = {
            'status': 'extracting',
            'progress': 10,
            'message': 'Extracting text from document...',
            'started_at': datetime.utcnow().isoformat()
        }
        
        # Update both Redis (if available) and in-memory for compatibility
        redis_adapter.set_status(file_id, initial_status)
        document_processing_status[file_id] = initial_status
        
        logger.info(f"Started processing document {filename} for user {user_id}")
        
        # Step 2: Process document
        content, pages_processed, warnings = SafeDocumentProcessor.process_document_from_bytes(
            file_content, filename, file_ext
        )
        
        if not content or len(content.strip()) < 50:
            raise ValueError("Document appears to be empty or could not be processed")
        
        logger.info(f"Extracted {len(content)} characters from {pages_processed} pages")
        
        # Step 3: Update status - Chunking
        chunking_status = {
            'status': 'chunking',
            'progress': 40,
            'message': f'Creating searchable chunks from {pages_processed} pages...',
            'pages': pages_processed
        }
        
        redis_adapter.set_status(file_id, chunking_status)
        document_processing_status[file_id] = chunking_status
        
        # Step 4: Prepare metadata
        metadata = {
            'source': filename,
            'file_id': file_id,
            'upload_date': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'file_type': file_ext,
            'pages': pages_processed,
            'file_size': len(file_content),
            'content_length': len(content),
            'processing_warnings': warnings
        }
        
        # Step 5: Add to container with progress tracking
        container_manager = get_container_manager()
        
        def progress_callback(p: int):
            """Update progress during chunking"""
            progress_status = {
                'status': 'chunking',
                'progress': min(40 + int(p * 0.5), 90),
                'message': f'Processing chunks... ({p}%)',
                'pages': pages_processed
            }
            redis_adapter.set_status(file_id, progress_status)
            document_processing_status[file_id] = progress_status
        
        # Use async-friendly chunking
        chunks_created = await container_manager.add_document_to_container_async(
            user_id,
            content,
            metadata,
            file_id,
            progress_callback=progress_callback
        )
        
        processing_time = time.time() - start_time
        
        # Step 6: Update final status - Completed
        final_status = {
            'status': 'completed',
            'progress': 100,
            'message': f'Successfully processed {pages_processed} pages into {chunks_created} searchable chunks',
            'completed_at': datetime.utcnow().isoformat(),
            'processing_time': processing_time,
            'pages_processed': pages_processed,
            'chunks_created': chunks_created
        }
        
        # Update both storage systems
        redis_adapter.set_status(file_id, final_status)
        document_processing_status[file_id] = final_status
        
        # Update uploaded_files for compatibility
        if file_id in uploaded_files:
            uploaded_files[file_id].update({
                'status': 'completed',
                'pages_processed': pages_processed,
                'chunks_created': chunks_created,
                'processing_time': processing_time,
                'content_length': len(content)
            })
        
        logger.info(f"✅ Document {file_id} processed successfully: {pages_processed} pages, {chunks_created} chunks in {processing_time:.2f}s")
        
    except Exception as e:
        # Handle failure
        processing_time = time.time() - start_time
        error_message = str(e)
        
        logger.error(f"❌ Error processing document {file_id}: {error_message}", exc_info=True)
        
        failed_status = {
            'status': 'failed',
            'progress': 0,
            'message': f'Processing failed: {error_message}',
            'error': error_message,
            'failed_at': datetime.utcnow().isoformat(),
            'processing_time': processing_time
        }
        
        # Update both storage systems
        redis_adapter.set_status(file_id, failed_status)
        document_processing_status[file_id] = failed_status
        
        # Update uploaded_files for compatibility
        if file_id in uploaded_files:
            uploaded_files[file_id]['status'] = 'failed'

# --- Updated Status Endpoint for API Compatibility ---

def get_document_processing_status(file_id: str) -> Dict[str, Any]:
    """
    Get document processing status - compatible with existing API
    """
    redis_adapter = get_redis_adapter()
    
    # Try Redis first, fallback to in-memory
    status = redis_adapter.get_status(file_id)
    if status:
        return {
            'file_id': file_id,
            'status': status.get('status', 'unknown'),
            'progress': status.get('progress', 0),
            'message': status.get('message', ''),
            'pages_processed': status.get('details', {}).get('pages', 0),
            'chunks_created': status.get('details', {}).get('chunks_created', 0),
            'processing_time': status.get('processing_time', 0),
            'errors': [status.get('error')] if status.get('error') else []
        }
    
    # Fallback to existing in-memory storage
    if file_id in document_processing_status:
        return document_processing_status[file_id]
    
    return {
        'file_id': file_id,
        'status': 'not_found',
        'progress': 0,
        'message': 'Document not found in processing queue'
    }

# --- Easy Migration Path ---

class BackwardCompatibleStatusManager:
    """
    Drop-in replacement that provides Redis benefits while maintaining compatibility
    """
    
    def __init__(self):
        self.redis_adapter = get_redis_adapter()
    
    def update_status(self, file_id: str, status: str, progress: int = 0, 
                     message: str = "", details: Dict = None):
        """Update status - compatible with existing code"""
        
        status_data = {
            'status': status,
            'progress': progress,
            'message': message,
            'updated_at': datetime.utcnow().isoformat(),
            'details': details or {}
        }
        
        # Update both systems for compatibility
        self.redis_adapter.set_status(file_id, status_data)
        document_processing_status[file_id] = status_data
    
    def get_status(self, file_id: str) -> Optional[Dict]:
        """Get status - compatible with existing code"""
        return get_document_processing_status(file_id)

# Global compatible instance
_compatible_status_manager = None

def get_compatible_status_manager() -> BackwardCompatibleStatusManager:
    """Get backward-compatible status manager"""
    global _compatible_status_manager
    if _compatible_status_manager is None:
        _compatible_status_manager = BackwardCompatibleStatusManager()
    return _compatible_status_manager
