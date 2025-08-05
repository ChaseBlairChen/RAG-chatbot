"""
Background tasks for document processing using a robust, scalable status manager.
"""
import logging
import time
import json
from enum import Enum
from datetime import datetime
from typing import Dict, Any, Optional

import redis
from pydantic import BaseModel, Field

from ..services.document_processor import SafeDocumentProcessor
from ..services.container_manager import get_container_manager
# Note: We are no longer importing the in-memory managers
# from ..storage.managers import uploaded_files, document_processing_status

logger = logging.getLogger(__name__)

# --- Configuration Constants ---
# Progress percentages are defined in one place for easy management
PROGRESS_EXTRACTING = 10
PROGRESS_CHUNKING_START = 30
PROGRESS_CHUNKING_END = 90
PROGRESS_COMPLETE = 100
MIN_CONTENT_LENGTH = 50
STATUS_TTL_SECONDS = 86400  # Status expires from Redis after 24 hours

# --- Enums and Pydantic Models for Robustness ---

class ProcessingStatus(str, Enum):
    """Enumeration for document processing statuses to avoid magic strings."""
    QUEUED = "queued"
    EXTRACTING = "extracting"
    CHUNKING_AND_EMBEDDING = "chunking_and_embedding"
    COMPLETED = "completed"
    FAILED = "failed"

class Status(BaseModel):
    """Pydantic model for a structured and validated status object."""
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


# --- Redis-backed Status Manager for Scalability ---

class StatusManager:
    """Manages processing status in Redis for persistence and scalability."""
    def __init__(self, redis_client: redis.Redis, ttl: int = STATUS_TTL_SECONDS):
        self.redis = redis_client
        self.ttl = ttl

    def _get_key(self, file_id: str) -> str:
        return f"doc_status:{file_id}"

    def update_status(self, status: Status):
        """Sets or updates the status for a given file_id in Redis."""
        key = self._get_key(status.file_id)
        # Use .model_dump_json() for Pydantic v2+
        self.redis.set(key, status.model_dump_json(), ex=self.ttl)
        logger.debug(f"Updated status for {status.file_id}: {status.status.value}")
    
    def get_status(self, file_id: str) -> Optional[Status]:
        """Retrieves the status for a given file_id from Redis."""
        key = self._get_key(file_id)
        data = self.redis.get(key)
        if data:
            # Use Status.model_validate_json for Pydantic v2+
            return Status.model_validate_json(data)
        return None

# --- Improved Background Task ---

async def process_document_background(
    file_id: str,
    file_content: bytes,
    file_ext: str,
    filename: str,
    user_id: str
):
    """
    Process a document in the background with robust, scalable progress updates via Redis.
    """
    # Initialize dependencies
    redis_client = redis.Redis.from_url("redis://localhost:6379/0", decode_responses=True)
    status_manager = StatusManager(redis_client)
    container_manager = get_container_manager()
    start_time = time.time()

    try:
        # 1. Initial Status: Extracting
        status_manager.update_status(Status(
            file_id=file_id,
            status=ProcessingStatus.EXTRACTING,
            progress=PROGRESS_EXTRACTING,
            message="Extracting text from document...",
            started_at=datetime.utcnow().isoformat()
        ))

        # 2. Process Document (CPU-bound, could be run in a thread pool executor)
        content, pages_processed, warnings = SafeDocumentProcessor.process_document_from_bytes(
            file_content, filename, file_ext
        )

        if not content or len(content.strip()) < MIN_CONTENT_LENGTH:
            raise ValueError(f"Document content is too short or could not be extracted. Min length: {MIN_CONTENT_LENGTH} chars.")

        # 3. Status Update: Chunking & Embedding
        status_manager.update_status(Status(
            file_id=file_id,
            status=ProcessingStatus.CHUNKING_AND_EMBEDDING,
            progress=PROGRESS_CHUNKING_START,
            message=f"Creating searchable chunks from {pages_processed} pages...",
            details={'pages': pages_processed}
        ))

        # 4. Prepare Metadata and Add to Vector DB
        metadata = {
            'source': filename, 'file_id': file_id,
            'upload_date': datetime.utcnow().isoformat(),
            'user_id': user_id, 'file_type': file_ext,
            'pages': pages_processed, 'file_size': len(file_content),
            'content_length': len(content), 'processing_warnings': warnings
        }

        # Define a callback to update progress during the embedding stage
        def progress_callback(embedding_progress: float):
            # Scale embedding progress (0.0 to 1.0) to our defined range
            progress = PROGRESS_CHUNKING_START + int(embedding_progress * (PROGRESS_CHUNKING_END - PROGRESS_CHUNKING_START))
            status_manager.update_status(Status(
                file_id=file_id,
                status=ProcessingStatus.CHUNKING_AND_EMBEDDING,
                progress=progress,
                message=f"Embedding document... ({int(embedding_progress*100)}%)"
            ))

        chunks_created = await container_manager.add_document_to_container_async(
            user_id,
            content,
            metadata,
            file_id,
            progress_callback=progress_callback
        )

        # 5. Final Status: Completed
        processing_time = time.time() - start_time
        status_manager.update_status(Status(
            file_id=file_id,
            status=ProcessingStatus.COMPLETED,
            progress=PROGRESS_COMPLETE,
            message=f"Successfully processed {pages_processed} pages into {chunks_created} searchable chunks.",
            completed_at=datetime.utcnow().isoformat(),
            processing_time=processing_time,
            details={
                'pages_processed': pages_processed,
                'chunks_created': chunks_created
            }
        ))
        logger.info(f"Document {file_id} processed successfully in {processing_time:.2f}s")

    except Exception as e:
        # Final Status: Failed
        processing_time = time.time() - start_time
        logger.error(f"Error processing document {file_id}: {e}", exc_info=True)
        status_manager.update_status(Status(
            file_id=file_id,
            status=ProcessingStatus.FAILED,
            progress=0, # Reset progress on failure
            message=f"An unexpected error occurred.",
            error=str(e),
            failed_at=datetime.utcnow().isoformat(),
            processing_time=processing_time
        ))
