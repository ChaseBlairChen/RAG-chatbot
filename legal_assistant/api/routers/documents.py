"""Document management endpoints"""
import os
import uuid
import logging
import traceback
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException

from ...models import User, DocumentUploadResponse
from ...config import MAX_FILE_SIZE, LEGAL_EXTENSIONS
from ...core.security import get_current_user
from ...services.document_processor import SafeDocumentProcessor
from ...services.container_manager import get_container_manager
from ...storage.managers import uploaded_files

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/user/upload", response_model=DocumentUploadResponse)
async def upload_user_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Enhanced upload endpoint with file_id tracking and timeout handling"""
    start_time = datetime.utcnow()
    
    try:
        # Check file size first (before reading)
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE//1024//1024}MB. Your file: {file_size//1024//1024}MB"
            )
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in LEGAL_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type. Supported: {LEGAL_EXTENSIONS}")
        
        logger.info(f"Processing upload: {file.filename} ({file_size//1024}KB) for user {current_user.user_id}")
        
        # Process document with timeout protection
        try:
            content, pages_processed, warnings = SafeDocumentProcessor.process_document_safe(file)
        except Exception as doc_error:
            logger.error(f"Document processing failed: {doc_error}")
            raise HTTPException(
                status_code=422, 
                detail=f"Failed to process document: {str(doc_error)}"
            )
        
        if not content or len(content.strip()) < 50:
            raise HTTPException(
                status_code=422,
                detail="Document appears to be empty or could not be processed properly"
            )
        
        file_id = str(uuid.uuid4())
        metadata = {
            'source': file.filename,
            'file_id': file_id,
            'upload_date': datetime.utcnow().isoformat(),
            'user_id': current_user.user_id,
            'file_type': file_ext,
            'pages': pages_processed,
            'file_size': file_size,
            'content_length': len(content),
            'processing_warnings': warnings
        }
        
        logger.info(f"Adding document to container: {len(content)} chars, {pages_processed} pages")
        
        # Add to container with timeout protection
        container_manager = get_container_manager()
        try:
            success = container_manager.add_document_to_container(
                current_user.user_id,
                content,
                metadata,
                file_id
            )
        except Exception as container_error:
            logger.error(f"Container operation failed: {container_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to store document: {str(container_error)}"
            )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to add document to user container")
        
        session_id = str(uuid.uuid4())
        uploaded_files[file_id] = {
            'filename': file.filename,
            'user_id': current_user.user_id,
            'container_id': current_user.container_id,
            'pages_processed': pages_processed,
            'uploaded_at': datetime.utcnow(),
            'session_id': session_id,
            'file_size': file_size,
            'content_length': len(content)
        }
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"Upload successful: {file.filename} processed in {processing_time:.2f}s")
        
        return DocumentUploadResponse(
            message=f"Document {file.filename} uploaded successfully ({pages_processed} pages, {len(content)} chars)",
            file_id=file_id,
            pages_processed=pages_processed,
            processing_time=processing_time,
            warnings=warnings,
            session_id=session_id,
            user_id=current_user.user_id,
            container_id=current_user.container_id or ""
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        logger.error(f"Error uploading user document after {processing_time:.2f}s: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Upload failed after {processing_time:.2f}s: {str(e)}"
        )

@router.get("/user/documents")
async def list_user_documents(current_user: User = Depends(get_current_user)):
    """ENHANCED: List all documents in user's container with better error handling"""
    try:
        user_documents = []
        
        # Add timeout and better error handling
        for file_id, file_data in uploaded_files.items():
            try:
                if file_data.get('user_id') == current_user.user_id:
                    # Handle both datetime objects and strings
                    uploaded_at_str = file_data['uploaded_at']
                    if hasattr(uploaded_at_str, 'isoformat'):
                        uploaded_at_str = uploaded_at_str.isoformat()
                    elif not isinstance(uploaded_at_str, str):
                        uploaded_at_str = str(uploaded_at_str)
                    
                    user_documents.append({
                        'file_id': file_id,
                        'filename': file_data['filename'],
                        'uploaded_at': uploaded_at_str,
                        'pages_processed': file_data.get('pages_processed', 0),
                        'file_size': file_data.get('file_size', 0)
                    })
            except Exception as e:
                logger.warning(f"Error processing file {file_id}: {e}")
                continue
        
        logger.info(f"Retrieved {len(user_documents)} documents for user {current_user.user_id}")
        
        return {
            'user_id': current_user.user_id,
            'container_id': current_user.container_id,
            'documents': user_documents,
            'total_documents': len(user_documents)
        }
        
    except Exception as e:
        logger.error(f"Error listing user documents: {e}")
        # Return empty list instead of failing completely
        return {
            'user_id': current_user.user_id,
            'container_id': current_user.container_id or "unknown",
            'documents': [],
            'total_documents': 0,
            'error': str(e)
        }

@router.delete("/user/documents/{file_id}")
async def delete_user_document(
    file_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a document from user's container"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Document not found")
    
    file_data = uploaded_files[file_id]
    if file_data.get('user_id') != current_user.user_id:
        raise HTTPException(status_code=403, detail="Unauthorized to delete this document")
    
    del uploaded_files[file_id]
    return {"message": "Document deleted successfully", "file_id": file_id}
