"""
Background tasks for document processing with enhanced progress tracking and error handling.
"""
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

from ..services.document_processor import SafeDocumentProcessor
from ..services.container_manager import get_container_manager
from ..storage.managers import uploaded_files, document_processing_status

logger = logging.getLogger(__name__)

# Processing status constants
PROGRESS_EXTRACTING = 10
PROGRESS_CHUNKING_START = 30
PROGRESS_CHUNKING_END = 90
PROGRESS_COMPLETE = 100
MIN_CONTENT_LENGTH = 50

async def process_document_background(
    file_id: str,
    file_content: bytes,
    file_ext: str,
    filename: str,
    user_id: str
):
    """
    Background document processing task with enhanced status tracking
    """
    start_time = time.time()
    
    try:
        logger.info(f"🚀 Starting background processing for {filename} (user: {user_id})")
        
        # Step 1: Initial Status - Extracting
        document_processing_status[file_id] = {
            'status': 'extracting',
            'progress': PROGRESS_EXTRACTING,
            'message': "Extracting text from document...",
            'started_at': datetime.utcnow().isoformat(),
            'details': {
                'filename': filename,
                'file_size': len(file_content),
                'user_id': user_id
            }
        }
        
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
            
            document_processing_status[file_id] = {
                'status': 'failed',
                'progress': 0,
                'message': "Document extraction failed - file may be corrupted or unsupported",
                'error': str(extraction_error),
                'details': {'extraction_time': time.time() - processing_start}
            }
            
            # Update uploaded_files for API compatibility
            if file_id in uploaded_files:
                uploaded_files[file_id]['status'] = 'failed'
            
            return  # Exit early on extraction failure
        
        # Step 3: Validate Content Quality
        content_quality = _assess_content_quality(content)
        
        if content_quality < 0.3:
            logger.warning(f"⚠️ Low content quality ({content_quality:.2f}) for {filename}")
            warnings.append(f"Low extraction quality detected: {content_quality:.2f}/1.0")
        
        if len(content.strip()) < MIN_CONTENT_LENGTH:
            error_msg = f"Content too short: {len(content.strip())} chars (minimum: {MIN_CONTENT_LENGTH})"
            logger.error(error_msg)
            
            document_processing_status[file_id] = {
                'status': 'failed',
                'progress': 0,
                'message': "Document content validation failed",
                'error': error_msg,
                'details': {'content_length': len(content.strip())}
            }
            
            if file_id in uploaded_files:
                uploaded_files[file_id]['status'] = 'failed'
            
            return
        
        # Step 4: Chunking & Embedding Status
        document_processing_status[file_id] = {
            'status': 'chunking_and_embedding',
            'progress': PROGRESS_CHUNKING_START,
            'message': f"Creating searchable chunks from {pages_processed} pages...",
            'details': {
                'pages': pages_processed,
                'content_length': len(content),
                'extraction_quality': content_quality,
                'warnings_count': len(warnings)
            }
        }
        
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
            'extraction_quality': content_quality
        }
        
        # Step 6: Chunking with Progress Tracking
        def progress_callback(embedding_progress: int):
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
            
            document_processing_status[file_id] = {
                'status': 'chunking_and_embedding',
                'progress': min(scaled_progress, PROGRESS_CHUNKING_END),
                'message': message,
                'details': {'embedding_progress': embedding_progress}
            }
        
        # Process document asynchronously
        chunking_start = time.time()
        
        try:
            container_manager = get_container_manager()
            chunks_created = await container_manager.add_document_to_container_async(
                user_id,
                content,
                metadata,
                file_id,
                progress_callback=progress_callback
            )
            
            chunking_time = time.time() - chunking_start
            logger.info(f"📊 Chunking completed in {chunking_time:.2f}s: {chunks_created} chunks created")
            
        except Exception as chunking_error:
            logger.error(f"❌ Chunking failed for {filename}: {chunking_error}")
            
            document_processing_status[file_id] = {
                'status': 'failed',
                'progress': 0,
                'message': "Document chunking and embedding failed",
                'error': str(chunking_error),
                'details': {'chunking_time': time.time() - chunking_start}
            }
            
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
        
        document_processing_status[file_id] = {
            'status': 'completed',
            'progress': PROGRESS_COMPLETE,
            'message': f"Successfully processed {pages_processed} pages into {chunks_created} searchable chunks",
            'details': final_details
        }
        
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
        
        document_processing_status[file_id] = {
            'status': 'failed',
            'progress': 0,
            'message': "Critical processing error occurred",
            'error': f"Critical error: {error_message}",
            'details': {
                'processing_time': processing_time,
                'error_type': type(e).__name__
            }
        }
        
        # Ensure uploaded_files is updated
        if file_id in uploaded_files:
            uploaded_files[file_id]['status'] = 'failed'

# --- ENHANCED BACKGROUND PROCESSING VARIANT ---

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
    # This is the same as process_document_background but with enhanced options support
    processing_options = processing_options or {}
    
    # For now, just call the standard version
    # In the future, you could add special processing based on options
    await process_document_background(
        file_id, file_content, file_ext, filename, user_id
    )

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
    # Get current status to preserve other fields
    current_status = document_processing_status.get(file_id, {})
    current_message = current_status.get('message', 'Processing...')
    current_status_type = current_status.get('status', 'processing')
    
    # Update with new progress
    document_processing_status[file_id] = {
        **current_status,
        'status': current_status_type,
        'progress': min(PROGRESS_CHUNKING_START + int(progress * 0.5), PROGRESS_CHUNKING_END),
        'message': current_message,
        'updated_at': datetime.utcnow().isoformat()
    }

# --- API Compatibility Functions ---

def get_enhanced_document_status(file_id: str) -> Dict[str, Any]:
    """
    Get enhanced document status that works with existing API endpoints
    """
    # Get status from processing status
    status_data = document_processing_status.get(file_id)
    
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
