"""Admin endpoints"""
import os
import logging
from datetime import datetime
from fastapi import APIRouter, Form, Depends

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ...models import User
from ...config import USER_CONTAINERS_PATH
from ...core.security import get_current_user
from ...services.container_manager import get_container_manager
from ...storage.managers import uploaded_files
from ...utils.formatting import format_context_for_llm
from ...utils.text_processing import extract_bill_information, extract_universal_information

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/cleanup-containers")
async def cleanup_orphaned_containers():
    """Clean up orphaned files in containers that are no longer tracked"""
    cleanup_results = {
        "containers_checked": 0,
        "orphaned_documents_found": 0,
        "cleanup_performed": False,
        "errors": []
    }
    
    try:
        if not os.path.exists(USER_CONTAINERS_PATH):
            return cleanup_results
        
        container_dirs = [d for d in os.listdir(USER_CONTAINERS_PATH) 
                         if os.path.isdir(os.path.join(USER_CONTAINERS_PATH, d))]
        
        cleanup_results["containers_checked"] = len(container_dirs)
        tracked_file_ids = set(uploaded_files.keys())
        
        logger.info(f"Checking {len(container_dirs)} containers against {len(tracked_file_ids)} tracked files")
        
        for container_dir in container_dirs:
            try:
                container_path = os.path.join(USER_CONTAINERS_PATH, container_dir)
                
                try:
                    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    db = Chroma(
                        collection_name=f"user_{container_dir}",
                        embedding_function=embedding_function,
                        persist_directory=container_path
                    )
                    
                    logger.info(f"Container {container_dir} loaded successfully")
                    
                except Exception as e:
                    logger.warning(f"Could not load container {container_dir}: {e}")
                    cleanup_results["errors"].append(f"Container {container_dir}: {str(e)}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error processing container {container_dir}: {e}")
                cleanup_results["errors"].append(f"Container {container_dir}: {str(e)}")
        
        return cleanup_results
        
    except Exception as e:
        logger.error(f"Error during container cleanup: {e}")
        cleanup_results["errors"].append(str(e))
        return cleanup_results

@router.post("/sync-document-tracking")
async def sync_document_tracking():
    """Sync the uploaded_files tracking with what's actually in the containers"""
    sync_results = {
        "tracked_files": len(uploaded_files),
        "containers_found": 0,
        "sync_performed": False,
        "recovered_files": 0,
        "errors": []
    }
    
    try:
        if not os.path.exists(USER_CONTAINERS_PATH):
            return sync_results
        
        container_dirs = [d for d in os.listdir(USER_CONTAINERS_PATH) 
                         if os.path.isdir(os.path.join(USER_CONTAINERS_PATH, d))]
        
        sync_results["containers_found"] = len(container_dirs)
        
        logger.info(f"Syncing document tracking: {len(uploaded_files)} tracked files, {len(container_dirs)} containers")
        
        return sync_results
        
    except Exception as e:
        logger.error(f"Error during document tracking sync: {e}")
        sync_results["errors"].append(str(e))
        return sync_results

@router.get("/document-health")
async def check_document_health():
    """Check the health of document tracking and containers"""
    health_info = {
        "timestamp": datetime.utcnow().isoformat(),
        "uploaded_files_count": len(uploaded_files),
        "container_directories": 0,
        "users_with_containers": 0,
        "orphaned_files": [],
        "container_errors": [],
        "recommendations": []
    }
    
    try:
        # Check container directories
        if os.path.exists(USER_CONTAINERS_PATH):
            container_dirs = [d for d in os.listdir(USER_CONTAINERS_PATH) 
                             if os.path.isdir(os.path.join(USER_CONTAINERS_PATH, d))]
            health_info["container_directories"] = len(container_dirs)
            
            # Check which users have containers
            user_ids_with_files = set()
            for file_data in uploaded_files.values():
                if 'user_id' in file_data:
                    user_ids_with_files.add(file_data['user_id'])
            
            health_info["users_with_containers"] = len(user_ids_with_files)
            
            # Check for potential issues
            if len(container_dirs) > len(user_ids_with_files):
                health_info["recommendations"].append("Some containers may be orphaned - consider running cleanup")
            
            if len(uploaded_files) == 0 and len(container_dirs) > 0:
                health_info["recommendations"].append("Containers exist but no files are tracked - may need sync")
        
        # Check for files with missing metadata
        for file_id, file_data in uploaded_files.items():
            if not file_data.get('user_id'):
                health_info["orphaned_files"].append(file_id)
        
        if health_info["orphaned_files"]:
            health_info["recommendations"].append(f"{len(health_info['orphaned_files'])} files have missing user_id")
        
        logger.info(f"Document health check: {health_info['uploaded_files_count']} files, {health_info['container_directories']} containers")
        
        return health_info
        
    except Exception as e:
        logger.error(f"Error during document health check: {e}")
        health_info["container_errors"].append(str(e))
        return health_info

@router.post("/emergency-clear-tracking")
async def emergency_clear_document_tracking():
    """EMERGENCY: Clear all document tracking"""
    try:
        global uploaded_files
        backup_count = len(uploaded_files)
        uploaded_files.clear()
        
        logger.warning(f"EMERGENCY: Cleared tracking for {backup_count} files")
        
        return {
            "status": "completed",
            "cleared_files": backup_count,
            "warning": "All document tracking has been cleared. Users will need to re-upload documents.",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during emergency clear: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }

@router.get("/debug/test-bill-search")
async def debug_bill_search_get(
    bill_number: str,
    user_id: str
):
    """Debug bill-specific search functionality (GET version for browser testing)"""
    
    try:
        container_manager = get_container_manager()
        # Get user database
        user_db = container_manager.get_user_database_safe(user_id)
        if not user_db:
            return {"error": "No user database found"}
        
        # Get all documents and check metadata
        all_docs = user_db.get()
        found_chunks = []
        
        logger.info(f"Debugging search for bill: {bill_number}")
        logger.info(f"Total documents in database: {len(all_docs.get('ids', []))}")
        
        for i, (doc_id, metadata, content) in enumerate(zip(
            all_docs.get('ids', []), 
            all_docs.get('metadatas', []), 
            all_docs.get('documents', [])
        )):
            if metadata:
                chunk_index = metadata.get('chunk_index', 'unknown')
                contains_bills = metadata.get('contains_bills', '')
                
                if bill_number in contains_bills:
                    found_chunks.append({
                        'chunk_index': chunk_index,
                        'contains_bills': contains_bills,
                        'content_preview': content[:200] + "..." if len(content) > 200 else content
                    })
                    logger.info(f"Found {bill_number} in chunk {chunk_index}")
        
        # Also test direct text search
        direct_search = [content for content in all_docs.get('documents', []) if bill_number in content]
        
        return {
            "bill_number": bill_number,
            "user_id": user_id,
            "total_chunks": len(all_docs.get('ids', [])),
            "chunks_with_bill_metadata": found_chunks,
            "chunks_with_bill_in_text": len(direct_search),
            "text_search_preview": direct_search[0][:300] + "..." if direct_search else "Not found in text",
            "sample_metadata": all_docs.get('metadatas', [])[:2] if all_docs.get('metadatas') else []
        }
        
    except Exception as e:
        logger.error(f"Debug bill search failed: {e}")
        return {"error": str(e)}

@router.post("/debug/test-bill-search")
async def debug_bill_search(
    bill_number: str = Form(...),
    user_id: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    """Debug bill-specific search functionality"""
    
    try:
        container_manager = get_container_manager()
        # Get user database
        user_db = container_manager.get_user_database_safe(user_id)
        if not user_db:
            return {"error": "No user database found"}
        
        # Get all documents and check metadata
        all_docs = user_db.get()
        found_chunks = []
        
        logger.info(f"Debugging search for bill: {bill_number}")
        logger.info(f"Total documents in database: {len(all_docs.get('ids', []))}")
        
        for i, (doc_id, metadata, content) in enumerate(zip(
            all_docs.get('ids', []), 
            all_docs.get('metadatas', []), 
            all_docs.get('documents', [])
        )):
            if metadata:
                chunk_index = metadata.get('chunk_index', 'unknown')
                contains_bills = metadata.get('contains_bills', '')
                
                if bill_number in contains_bills:
                    found_chunks.append({
                        'chunk_index': chunk_index,
                        'contains_bills': contains_bills,
                        'content_preview': content[:200] + "..." if len(content) > 200 else content
                    })
                    logger.info(f"Found {bill_number} in chunk {chunk_index}")
        
        # Also test direct text search
        direct_search = [content for content in all_docs.get('documents', []) if bill_number in content]
        
        return {
            "bill_number": bill_number,
            "total_chunks": len(all_docs.get('ids', [])),
            "chunks_with_bill_metadata": found_chunks,
            "chunks_with_bill_in_text": len(direct_search),
            "text_search_preview": direct_search[0][:300] + "..." if direct_search else "Not found in text"
        }
        
    except Exception as e:
        logger.error(f"Debug bill search failed: {e}")
        return {"error": str(e)}

@router.post("/debug/test-extraction")
async def debug_test_extraction(
    question: str = Form(...),
    user_id: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    """Test information extraction for any question"""
    
    try:
        container_manager = get_container_manager()
        # Search user's documents
        user_results = container_manager.enhanced_search_user_container(user_id, question, "", k=5)
        
        if user_results:
            # Get context
            context_text, source_info = format_context_for_llm(user_results, max_length=3000)
            
            # Test extraction
            import re
            bill_match = re.search(r"(HB|SB|SSB|ESSB|SHB|ESHB)\s*(\d+)", question, re.IGNORECASE)
            if bill_match:
                bill_number = f"{bill_match.group(1)} {bill_match.group(2)}"
                extracted_info = extract_bill_information(context_text, bill_number)
            else:
                extracted_info = extract_universal_information(context_text, question)
            
            return {
                "question": question,
                "context_preview": context_text[:500] + "...",
                "extracted_info": extracted_info,
                "sources_found": len(user_results)
            }
        else:
            return {
                "question": question,
                "error": "No relevant documents found"
            }
            
    except Exception as e:
        return {"error": str(e)}
