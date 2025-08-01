# 1. Update api/routers/documents.py for async processing
"""Document management endpoints with async processing"""
import os
import uuid
import logging
import traceback
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from ...models import User, DocumentUploadResponse
from ...config import MAX_FILE_SIZE, LEGAL_EXTENSIONS
from ...core.security import get_current_user
from ...services.document_processor import SafeDocumentProcessor
from ...services.container_manager import get_container_manager
from ...storage.managers import uploaded_files, document_processing_status
from ...tasks.document_tasks import process_document_background

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/user/upload", response_model=DocumentUploadResponse)
async def upload_user_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Enhanced upload endpoint with background processing"""
    start_time = datetime.utcnow()
    
    try:
        # Check file size first
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE//1024//1024}MB"
            )
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in LEGAL_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type")
        
        # Generate file_id immediately
        file_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        
        # Read file content
        file_content = file.file.read()
        
        # Quick validation - just check if we can extract text
        try:
            content_preview, _, _ = SafeDocumentProcessor.quick_validate(file_content, file_ext)
            if len(content_preview) < 50:
                raise HTTPException(status_code=422, detail="Document appears to be empty")
        except Exception as e:
            logger.error(f"Document validation failed: {e}")
            raise HTTPException(status_code=422, detail=f"Invalid document: {str(e)}")
        
        # Store initial metadata
        uploaded_files[file_id] = {
            'filename': file.filename,
            'user_id': current_user.user_id,
            'container_id': current_user.container_id,
            'uploaded_at': datetime.utcnow(),
            'session_id': session_id,
            'file_size': file_size,
            'status': 'processing',
            'progress': 0
        }
        
        # Initialize processing status
        document_processing_status[file_id] = {
            'status': 'queued',
            'progress': 0,
            'message': 'Document queued for processing',
            'started_at': datetime.utcnow().isoformat()
        }
        
        # Process document in background
        background_tasks.add_task(
            process_document_background,
            file_id=file_id,
            file_content=file_content,
            file_ext=file_ext,
            filename=file.filename,
            user_id=current_user.user_id
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"Document {file.filename} queued for processing in {processing_time:.2f}s")
        
        return DocumentUploadResponse(
            message=f"Document {file.filename} is being processed",
            file_id=file_id,
            pages_processed=0,  # Will be updated async
            processing_time=processing_time,
            warnings=[],
            session_id=session_id,
            user_id=current_user.user_id,
            container_id=current_user.container_id or "",
            status="processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/user/documents/{file_id}/status")
async def get_document_status(
    file_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get document processing status"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Document not found")
    
    file_data = uploaded_files[file_id]
    if file_data.get('user_id') != current_user.user_id:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    status = document_processing_status.get(file_id, {
        'status': 'unknown',
        'progress': 0,
        'message': 'Status unavailable'
    })
    
    return {
        'file_id': file_id,
        'filename': file_data['filename'],
        'status': status['status'],
        'progress': status['progress'],
        'message': status.get('message', ''),
        'pages_processed': file_data.get('pages_processed', 0),
        'chunks_created': file_data.get('chunks_created', 0),
        'processing_time': file_data.get('processing_time', 0),
        'errors': status.get('errors', [])
    }
