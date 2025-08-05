# legal_assistant/tasks/document_tasks.py - COMPATIBLE VERSION
"""
Enhanced background tasks for document processing with Redis support but full backward compatibility.
Works with or without Redis, maintains existing API compatibility.
"""
import logging
import time
import json
from enum import Enum
from datetime import datetime
from typing import Dict, Any, Optional, Callable

from pydantic import BaseModel, Field

from ..services.document_processor import SafeDocumentProcessor
from ..services.container_manager import get_container_manager
from ..storage.managers import uploaded_files, document_processing_status

logger = logging.getLogger(__name__)

# --- Configuration Constants ---
PROGRESS_EXTRACTING = 10
PROGRESS_CHUNKING_START = 30
PROGRESS_CHUNKING_END = 90
PROGRESS_COMPLETE = 100
MIN_CONTENT_LENGTH = 50
STATUS_TTL_SECONDS = 86400

# --- Enums and Models ---

class ProcessingStatus(str, Enum):
    """Document processing status enumeration"""
    QUEUED = "queued"
    EXTRACTING = "extracting" 
    CHUNKING = "chunking"
    CHUNKING_AND_EMBEDDING = "chunking_and_embedding"
    COMPLETED = "completed"
    FAILED = "failed"

class StatusData(BaseModel):
    """Structured status data"""
    file_id: str
    status: ProcessingStatus
    progress: int = Field(..., ge=0, le=100)
    message: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    failed_at: Optional[str] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)

# --- Hybrid Status Manager (Redis + In-Memory) ---

class HybridStatusManager:
    """
    Manages status using both Redis (if available) and in-memory storage for compatibility.
    Gracefully falls back to in-memory if Redis is unavailable.
    """
    
    def __init__(self):
        self.redis_client = None
        self.redis_available = False
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis with graceful fallback"""
        try:
            import redis
            self.redis_client = redis.Redis.from_url(
                "redis://localhost:6379/0", 
                decode_responses=True,
                socket_timeout=2,
                socket_connect_timeout=2
            )
            # Test connection
            self.redis_client.ping()
            self.redis_available = True
            logger.info("✅ Redis connected for enhanced document processing")
        except Exception as e:
            logger.info(f"ℹ️ Redis not available, using in-memory storage: {e}")
            self.redis_available = False
    
    def update_status(self, file_id: str, status: str, progress: int = 0, 
                     message: str = "", details: Dict = None, error: str = None):
        """
        Update status in both Redis and in-memory for full compatibility
        """
        details = details or {}
        
        # Create status data
        status_dict = {
            'status': status,
            'progress': progress,
            'message': message,
            'updated_at': datetime.utcnow().isoformat(),
            'details': details
        }
        
        if error:
            status_dict['error'] = error
            status_dict['failed_at'] = datetime.utcnow().isoformat()
        
        if status == 'completed':
            status_dict['completed_at'] = datetime.utcnow().isoformat()
        
        # Always update in-memory for API compatibility
        document_processing_status[file_id] = status_dict
        
        # Also update Redis if available
        if self.redis_available:
            try:
                self.redis_client.set(
                    f"doc_status:{file_id}", 
                    json.dumps(status_dict), 
                    ex=STATUS_TTL_SECONDS
                )
            except Exception as e:
                logger.warning(f"Redis update failed, continuing with in-memory: {e}")
    
    def get_status(self, file_id: str) -> Optional[Dict]:
        """Get status from Redis or in-memory"""
        # Try Redis first if available
        if self.redis_available:
            try:
                data = self.redis_client.get(f"doc_status:{file_id}")
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.warning(f"Redis read failed: {e}")
        
        # Fallback to in-memory
        return document_processing_status.get(file_id)

# Global status manager instance
_status_manager = None

def get_status_manager() -> HybridStatusManager:
    """Get or create status manager instance"""
    global _status_manager
    if _status_manager is None:
        _status_manager = HybridStatusManager()
    return _status_manager

# --- Enhanced Background Task ---

async def process_document_background(
    file_id: str,
    file_content: bytes,
    file_ext: str,
    filename: str,
    user_id: str
):
    """
    ENHANCED: Process document with better progress tracking and error handling
    COMPATIBLE: Works with existing API endpoints and status checking
    """
    status_manager = get_status_manager()
    start_time = time.time()
    
    try:
        logger.info(f"🚀 Starting background processing for {filename} (user: {user_id})")
        
        # Step 1: Initial Status - Extracting
        status_manager.update_status(
            file_id=file_id,
            status=ProcessingStatus.EXTRACTING.value,
            progress=PROGRESS_EXTRACTING,
            message="Extracting text from document...",
            details={'started_at': datetime.utcnow().isoformat()}
        )
        
        # Step 2: Process Document with enhanced error handling
        try:
            content, pages_processed, warnings = SafeDocumentProcessor.process_document_from_bytes(
                file_content, filename, file_ext
            )
            
            logger.info(f"📄 Extracted {len(content)} characters from {pages_processed} pages")
            
        except Exception as extraction_error:
            logger.error(f"Document extraction failed for {filename}: {extraction_error}")
            status_manager.update_status(
                file_id=file_id,
                status=ProcessingStatus.FAILED.value,
                progress=0,
                message="Document extraction failed",
                error=f"Extraction error: {str(extraction_error)}"
            )
            
            # Also update uploaded_files for compatibility
            if file_id in uploaded_files:
                uploaded_files[file_id]['status'] = 'failed'
            
            return  # Exit early on extraction failure
        
        # Validate content quality
        if not content or len(content.strip()) < MIN_CONTENT_LENGTH:
            error_msg = f"Document content too short ({len(content.strip())} chars). Minimum required: {MIN_CONTENT_LENGTH} chars."
            logger.error(error_msg)
            
            status_manager.update_status(
                file_id=file_id,
                status=ProcessingStatus.FAILED.value,
                progress=0,
                message="Document content validation failed",
                error=error_msg
            )
            
            if file_id in uploaded_files:
                uploaded_files[file_id]['status'] = 'failed'
            
            return
        
        # Step 3: Update Status - Chunking & Embedding
        status_manager.update_status(
            file_id=file_id,
            status=ProcessingStatus.CHUNKING_AND_EMBEDDING.value,
            progress=PROGRESS_CHUNKING_START,
            message=f"Creating searchable chunks from {pages_processed} pages...",
            details={
                'pages': pages_processed,
                'content_length': len(content),
                'warnings_count': len(warnings)
            }
        )
        
        # Step 4: Prepare Enhanced Metadata
        metadata = {
            'source': filename,
            'file_id': file_id,
            'upload_date': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'file_type': file_ext,
            'pages': pages_processed,
            'file_size': len(file_content),
            'content_length': len(content),
            'processing_warnings': warnings,
            'extraction_quality': _assess_content_quality(content)
        }
        
        # Step 5: Add to Container with Progress Tracking
        container_manager = get_container_manager()
        
        def progress_callback(embedding_progress: int):
            """Update progress during chunking and embedding"""
            # Scale embedding progress (0-100) to our defined range (30-90)
            scaled_progress = PROGRESS_CHUNKING_START + int(
                (embedding_progress / 100.0) * (PROGRESS_CHUNKING_END - PROGRESS_CHUNKING_START)
            )
            
            status_manager.update_status(
                file_id=file_id,
                status=ProcessingStatus.CHUNKING_AND_EMBEDDING.value,
                progress=min(scaled_progress, PROGRESS_CHUNKING_END),
                message=f"Creating embeddings... ({embedding_progress}%)",
                details={'embedding_progress': embedding_progress}
            )
        
        # Process with async chunking and progress updates
        try:
            chunks_created = await container_manager.add_document_to_container_async(
                user_id,
                content,
                metadata,
                file_id,
                progress_callback=progress_callback
            )
            
            logger.info(f"✅ Created {chunks_created} chunks for document {filename}")
            
        except Exception as chunking_error:
            logger.error(f"Chunking failed for {filename}: {chunking_error}")
            
            status_manager.update_status(
                file_id=file_id,
                status=ProcessingStatus.FAILED.value,
                progress=0,
                message="Document chunking failed",
                error=f"Chunking error: {str(chunking_error)}"
            )
            
            if file_id in uploaded_files:
                uploaded_files[file_id]['status'] = 'failed'
            
            return
        
        # Step 6: Final Status - Completed
        processing_time = time.time() - start_time
        
        status_manager.update_status(
            file_id=file_id,
            status=ProcessingStatus.COMPLETED.value,
            progress=PROGRESS_COMPLETE,
            message=f"Successfully processed {pages_processed} pages into {chunks_created} searchable chunks",
            details={
                'pages_processed': pages_processed,
                'chunks_created': chunks_created,
                'processing_time': processing_time,
                'content_length': len(content),
                'extraction_quality': metadata['extraction_quality']
            }
        )
        
        # Update uploaded_files for full compatibility
        if file_id in uploaded_files:
            uploaded_files[file_id].update({
                'status': 'completed',
                'pages_processed': pages_processed,
                'chunks_created': chunks_created,
                'processing_time': processing_time,
                'content_length': len(content)
            })
        
        logger.info(f"🎉 Document {file_id} processing completed successfully in {processing_time:.2f}s: "
                   f"{pages_processed} pages → {chunks_created} chunks")
        
    except Exception as e:
        # Handle any unexpected errors
        processing_time = time.time() - start_time
        error_message = str(e)
        
        logger.error(f"💥 Unexpected error processing document {file_id}: {error_message}", exc_info=True)
        
        status_manager.update_status(
            file_id=file_id,
            status=ProcessingStatus.FAILED.value,
            progress=0,
            message="Processing failed due to unexpected error",
            error=error_message,
            details={'processing_time': processing_time}
        )
        
        # Update uploaded_files for compatibility
        if file_id in uploaded_files:
            uploaded_files[file_id]['status'] = 'failed'

# --- Utility Functions ---

def _assess_content_quality(content: str) -> float:
    """Assess the quality of extracted content (0.0 to 1.0)"""
    if not content or len(content.strip()) < 50:
        return 0.0
    
    quality_score = 1.0
    
    # Check for unicode replacement characters
    unicode_error_ratio = content.count('�') / len(content)
    if unicode_error_ratio > 0.01:  # More than 1% errors
        quality_score -= min(0.5, unicode_error_ratio * 10)
    
    # Check content-to-whitespace ratio
    content_ratio = len(content.strip()) / len(content)
    if content_ratio < 0.3:  # More than 70% whitespace
        quality_score -= 0.3
    
    # Check for reasonable word structure
    import re
    word_count = len(re.findall(r'\b\w+\b', content))
    if word_count == 0:
        quality_score = 0.0
    elif word_count < 10:
        quality_score -= 0.3
    
    return max(0.0, min(1.0, quality_score))

def update_progress(file_id: str, progress: int):
    """
    LEGACY COMPATIBLE: Update processing progress for existing code
    """
    status_manager = get_status_manager()
    
    # Get current status to preserve other fields
    current_status = document_processing_status.get(file_id, {})
    current_message = current_status.get('message', 'Processing...')
    current_status_type = current_status.get('status', 'processing')
    
    # Update with new progress
    status_manager.update_status(
        file_id=file_id,
        status=current_status_type,
        progress=min(PROGRESS_CHUNKING_START + int(progress * 0.5), PROGRESS_CHUNKING_END),
        message=current_message,
        details={'legacy_progress_update': progress}
    )

# --- API Compatibility Functions ---

def get_enhanced_document_status(file_id: str) -> Dict[str, Any]:
    """
    Get enhanced document status that works with existing API endpoints
    """
    status_manager = get_status_manager()
    
    # Get status from hybrid manager
    status_data = status_manager.get_status(file_id)
    
    if not status_data:
        # Check uploaded_files as final fallback
        if file_id in uploaded_files:
            file_data = uploaded_files[file_id]
            return {
                'file_id': file_id,
                'filename': file_data.get('filename', ''),
                'status': file_data.get('status', 'unknown'),
                'progress': 100 if file_data.get('status') == 'completed' else 0,
                'message': 'Processing completed' if file_data.get('status') == 'completed' else 'Status unknown',
                'pages_processed': file_data.get('pages_processed', 0),
                'chunks_created': file_data.get('chunks_created', 0),
                'processing_time': file_data.get('processing_time', 0),
                'errors': []
            }
        
        return {
            'file_id': file_id,
            'status': 'not_found',
            'progress': 0,
            'message': 'Document not found in processing queue',
            'errors': ['Document not found']
        }
    
    # Format for API compatibility
    return {
        'file_id': file_id,
        'status': status_data.get('status', 'unknown'),
        'progress': status_data.get('progress', 0),
        'message': status_data.get('message', ''),
        'pages_processed': status_data.get('details', {}).get('pages_processed', 0),
        'chunks_created': status_data.get('details', {}).get('chunks_created', 0),
        'processing_time': status_data.get('details', {}).get('processing_time', 0),
        'errors': [status_data.get('error')] if status_data.get('error') else [],
        'extraction_quality': status_data.get('details', {}).get('extraction_quality', 1.0),
        'processing_method': status_data.get('details', {}).get('processing_method', 'unknown')
    }

# --- Background Processing with Enhanced Features ---

async def process_document_background_enhanced(
    file_id: str,
    file_content: bytes,
    file_ext: str,
    filename: str,
    user_id: str,
    processing_options: Optional[Dict[str, Any]] = None
):
    """
    ENHANCED VERSION: Process document with additional features while maintaining compatibility
    """
    processing_options = processing_options or {}
    status_manager = get_status_manager()
    container_manager = get_container_manager()
    start_time = time.time()
    
    try:
        logger.info(f"🚀 Enhanced processing started for {filename} (size: {len(file_content)} bytes)")
        
        # Step 1: Initial Status
        status_manager.update_status(
            file_id=file_id,
            status=ProcessingStatus.EXTRACTING.value,
            progress=PROGRESS_EXTRACTING,
            message="Extracting text from document...",
            details={
                'started_at': datetime.utcnow().isoformat(),
                'filename': filename,
                'file_size': len(file_content),
                'user_id': user_id
            }
        )
        
        # Step 2: Enhanced Document Processing
        processing_start = time.time()
        
        try:
            content, pages_processed, warnings = SafeDocumentProcessor.process_document_from_bytes(
                file_content, filename, file_ext
            )
            
            extraction_time = time.time() - processing_start
            logger.info(f"📄 Extraction completed in {extraction_time:.2f}s: "
                       f"{len(content)} characters, {pages_processed} pages")
            
        except Exception as extraction_error:
            logger.error(f"❌ Document extraction failed for {filename}: {extraction_error}")
            
            status_manager.update_status(
                file_id=file_id,
                status=ProcessingStatus.FAILED.value,
                progress=0,
                message="Document extraction failed - file may be corrupted or unsupported",
                error=str(extraction_error),
                details={'extraction_time': time.time() - processing_start}
            )
            
            # Update uploaded_files for API compatibility
            if file_id in uploaded_files:
                uploaded_files[file_id]['status'] = 'failed'
            
            return
        
        # Step 3: Validate Content Quality
        content_quality = _assess_content_quality(content)
        
        if content_quality < 0.3:
            logger.warning(f"⚠️ Low content quality ({content_quality:.2f}) for {filename}")
            warnings.append(f"Low extraction quality detected: {content_quality:.2f}/1.0")
        
        if len(content.strip()) < MIN_CONTENT_LENGTH:
            error_msg = f"Content too short: {len(content.strip())} chars (minimum: {MIN_CONTENT_LENGTH})"
            logger.error(error_msg)
            
            status_manager.update_status(
                file_id=file_id,
                status=ProcessingStatus.FAILED.value,
                progress=0,
                message="Document content validation failed",
                error=error_msg,
                details={'content_length': len(content.strip())}
            )
            
            if file_id in uploaded_files:
                uploaded_files[file_id]['status'] = 'failed'
            
            return
        
        # Step 4: Chunking & Embedding Status
        status_manager.update_status(
            file_id=file_id,
            status=ProcessingStatus.CHUNKING_AND_EMBEDDING.value,
            progress=PROGRESS_CHUNKING_START,
            message=f"Creating searchable chunks from {pages_processed} pages...",
            details={
                'pages': pages_processed,
                'content_length': len(content),
                'extraction_quality': content_quality,
                'warnings_count': len(warnings)
            }
        )
        
        # Step 5: Enhanced Metadata Preparation
        metadata = {
            'source': filename,
            'file_id': file_id,
            'upload_date': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'file_type': file_ext,
            'pages': pages_processed,
            'file_size': len(file_content),
            'content_length': len(content),
            'processing_warnings': warnings,
            'extraction_quality': content_quality,
            'processing_options': processing_options
        }
        
        # Step 6: Chunking with Enhanced Progress Tracking
        def enhanced_progress_callback(embedding_progress: int):
            """Enhanced progress callback with better status updates"""
            scaled_progress = PROGRESS_CHUNKING_START + int(
                (embedding_progress / 100.0) * (PROGRESS_CHUNKING_END - PROGRESS_CHUNKING_START)
            )
            
            # More descriptive messages based on progress
            if embedding_progress < 25:
                message = f"Analyzing document structure... ({embedding_progress}%)"
            elif embedding_progress < 50:
                message = f"Creating text chunks... ({embedding_progress}%)"
            elif embedding_progress < 75:
                message = f"Generating embeddings... ({embedding_progress}%)"
            else:
                message = f"Finalizing searchable index... ({embedding_progress}%)"
            
            status_manager.update_status(
                file_id=file_id,
                status=ProcessingStatus.CHUNKING_AND_EMBEDDING.value,
                progress=min(scaled_progress, PROGRESS_CHUNKING_END),
                message=message,
                details={'embedding_progress': embedding_progress}
            )
        
        # Process document asynchronously
        chunking_start = time.time()
        
        try:
            chunks_created = await container_manager.add_document_to_container_async(
                user_id,
                content,
                metadata,
                file_id,
                progress_callback=enhanced_progress_callback
            )
            
            chunking_time = time.time() - chunking_start
            logger.info(f"📊 Chunking completed in {chunking_time:.2f}s: {chunks_created} chunks created")
            
        except Exception as chunking_error:
            logger.error(f"❌ Chunking failed for {filename}: {chunking_error}")
            
            status_manager.update_status(
                file_id=file_id,
                status=ProcessingStatus.FAILED.value,
                progress=0,
                message="Document chunking and embedding failed",
                error=str(chunking_error),
                details={'chunking_time': time.time() - chunking_start}
            )
            
            if file_id in uploaded_files:
                uploaded_files[file_id]['status'] = 'failed'
            
            return
        
        # Step 7: Final Success Status
        total_processing_time = time.time() - start_time
        
        final_details = {
            'pages_processed': pages_processed,
            'chunks_created': chunks_created,
            'processing_time': total_processing_time,
            'extraction_time': extraction_time,
            'chunking_time': chunking_time,
            'content_length': len(content),
            'extraction_quality': content_quality,
            'warnings_count': len(warnings)
        }
        
        status_manager.update_status(
            file_id=file_id,
            status=ProcessingStatus.COMPLETED.value,
            progress=PROGRESS_COMPLETE,
            message=f"Successfully processed {pages_processed} pages into {chunks_created} searchable chunks",
            details=final_details
        )
        
        # Update uploaded_files for full API compatibility
        if file_id in uploaded_files:
            uploaded_files[file_id].update({
                'status': 'completed',
                'pages_processed': pages_processed,
                'chunks_created': chunks_created,
                'processing_time': total_processing_time,
                'content_length': len(content),
                'extraction_quality': content_quality
            })
        
        logger.info(f"🎉 Document {file_id} ({filename}) processing completed successfully!")
        logger.info(f"   📊 Stats: {pages_processed} pages → {chunks_created} chunks in {total_processing_time:.2f}s")
        logger.info(f"   🎯 Quality: {content_quality:.2f}/1.0, Warnings: {len(warnings)}")
        
    except Exception as e:
        # Handle any truly unexpected errors
        processing_time = time.time() - start_time
        error_message = str(e)
        
        logger.error(f"💥 CRITICAL ERROR processing document {file_id}: {error_message}", exc_info=True)
        
        status_manager.update_status(
            file_id=file_id,
            status=ProcessingStatus.FAILED.value,
            progress=0,
            message="Critical processing error occurred",
            error=f"Critical error: {error_message}",
            details={
                'processing_time': processing_time,
                'error_type': type(e).__name__
            }
        )
        
        # Ensure uploaded_files is updated
        if file_id in uploaded_files:
            uploaded_files[file_id]['status'] = 'failed'

# --- BACKWARD COMPATIBILITY ---

# Keep the original function name for existing code
process_document_background = process_document_background_enhanced

# Add enhanced status checking for existing API endpoints
def get_processing_status_enhanced(file_id: str) -> Dict[str, Any]:
    """
    Enhanced status checking that works with existing API endpoints
    """
    return get_enhanced_document_status(file_id)

# --- INTEGRATION NOTES ---

"""
INTEGRATION STEPS:

1. REPLACE your existing tasks/document_tasks.py with this file
2. ADD to requirements.txt: redis>=4.5.0  
3. NO OTHER CHANGES NEEDED - all existing code continues to work

FEATURES ADDED:
✅ Redis support (optional - graceful fallback)
✅ Better progress tracking with descriptive messages
✅ Content quality assessment (0.0-1.0 score)
✅ Enhanced error handling with specific error types
✅ Better logging and monitoring
✅ Backward compatibility with all existing APIs
✅ Processing time breakdown (extraction vs chunking)
✅ Enhanced metadata with quality scores

REDIS SETUP (OPTIONAL):
- Install: sudo apt install redis-server (Linux) or brew install redis (Mac)  
- Start: redis-server
- If not available, app automatically uses in-memory storage

BENEFITS:
- 🚀 Better processing reliability
- 📊 Quality scoring for documents
- ⏱️ Detailed timing metrics
- 🔄 Graceful fallback when Redis unavailable
- 🎯 Better error messages for users
- 📋 Enhanced progress tracking
- 🔧 Easier debugging with detailed logging

The app will work immediately with or without Redis!
"""
