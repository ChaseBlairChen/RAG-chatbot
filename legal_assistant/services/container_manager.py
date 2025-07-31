"""User container management service"""
import os
import hashlib
import logging
import re
import traceback
from typing import Optional, List, Tuple, Dict
from datetime import datetime

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from ..config import USER_CONTAINERS_PATH, FAST_EMBEDDING_MODELS
from ..core.exceptions import ContainerError
from ..core.dependencies import get_embeddings, get_nlp
from ..utils.text_processing import remove_duplicate_documents

logger = logging.getLogger(__name__)

class UserContainerManager:
    """Manages user-specific document containers with powerful embeddings"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.embeddings = None
        self._initialize_embeddings()
        logger.info(f"UserContainerManager initialized with base path: {base_path}")
    
    def _initialize_embeddings(self):
        """Initialize embeddings with the best available model"""
        # Try to use the global embeddings if available
        global_embeddings = get_embeddings()
        
        if global_embeddings:
            self.embeddings = global_embeddings
            logger.info(f"Using global embeddings model")
            return
        
        # TEMPORARY: Use faster embeddings for large document processing
        for model_name in FAST_EMBEDDING_MODELS:
            try:
                self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
                logger.info(f"✅ UserContainerManager using FAST embeddings: {model_name}")
                return
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        # Last resort fallback
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            logger.warning("⚠️ Using fallback embeddings: all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"❌ Failed to load any embeddings model: {e}")
            self.embeddings = None
    
    def create_user_container(self, user_id: str) -> str:
        """Create a new container for a user"""
        container_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        container_path = os.path.join(self.base_path, container_id)
        os.makedirs(container_path, exist_ok=True)
        
        # Ensure embeddings are available
        if not self.embeddings:
            self._initialize_embeddings()
        
        if not self.embeddings:
            raise ContainerError("No embeddings model available for container creation")
        
        user_db = Chroma(
            collection_name=f"user_{container_id}",
            embedding_function=self.embeddings,
            persist_directory=container_path
        )
        
        logger.info(f"Created container for user {user_id}: {container_id}")
        return container_id
    
    def get_container_id(self, user_id: str) -> str:
        """Get container ID for a user"""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    
    def get_user_database(self, user_id: str) -> Optional[Chroma]:
        """Get user's database"""
        container_id = self.get_container_id(user_id)
        container_path = os.path.join(self.base_path, container_id)
        
        if not os.path.exists(container_path):
            logger.warning(f"Container not found for user {user_id}")
            return None
        
        # Ensure embeddings are available
        if not self.embeddings:
            self._initialize_embeddings()
        
        if not self.embeddings:
            logger.error("No embeddings model available for database access")
            return None
        
        return Chroma(
            collection_name=f"user_{container_id}",
            embedding_function=self.embeddings,
            persist_directory=container_path
        )
    
    def get_user_database_safe(self, user_id: str) -> Optional[Chroma]:
        """Get user database with enhanced error handling and recovery"""
        try:
            container_id = self.get_container_id(user_id)
            container_path = os.path.join(self.base_path, container_id)
            
            if not os.path.exists(container_path):
                logger.warning(f"Container not found for user {user_id}, creating new one")
                self.create_user_container(user_id)
            
            # Ensure embeddings are available
            if not self.embeddings:
                self._initialize_embeddings()
            
            if not self.embeddings:
                logger.error("No embeddings model available for safe database access")
                return None
            
            return Chroma(
                collection_name=f"user_{container_id}",
                embedding_function=self.embeddings,
                persist_directory=container_path
            )
            
        except Exception as e:
            logger.error(f"Error getting user database for {user_id}: {e}")
            try:
                logger.info(f"Attempting to recover by creating new container for {user_id}")
                self.create_user_container(user_id)
                container_id = self.get_container_id(user_id)
                container_path = os.path.join(self.base_path, container_id)
                
                if not self.embeddings:
                    self._initialize_embeddings()
                
                if not self.embeddings:
                    logger.error("No embeddings model available for recovery")
                    return None
                
                return Chroma(
                    collection_name=f"user_{container_id}",
                    embedding_function=self.embeddings,
                    persist_directory=container_path
                )
            except Exception as recovery_error:
                logger.error(f"Recovery failed for user {user_id}: {recovery_error}")
                return None
    
    def add_document_to_container(self, user_id: str, document_text: str, metadata: Dict, file_id: str = None) -> bool:
        """Add document to user's container"""
        try:
            user_db = self.get_user_database_safe(user_id)
            if not user_db:
                container_id = self.create_user_container(user_id)
                user_db = self.get_user_database_safe(user_id)
            
            # Smart document type detection - lowered threshold
            bill_count = len(re.findall(r'\b(?:HB|SB|SHB|SSB|ESHB|ESSB)\s+\d+', document_text))
            is_legislative = bill_count > 1  # Lowered from 2 to 1 - even 1-2 bills = legislative
            
            if is_legislative:
                # Legislative document: Bill-aware chunking
                logger.info(f"Detected legislative document with {bill_count} bills - using bill-aware chunking")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=500,
                    length_function=len,
                    separators=["\n\n", "\nHB ", "\nSB ", "\nSHB ", "\nSSB ", "\nESHB ", "\nESSB ", "\n", " "]
                )
                chunking_method = 'bill_aware_chunking'
            else:
                # Regular document: Standard semantic chunking
                logger.info(f"Detected regular document ({bill_count} bills found) - using standard chunking")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=300,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]  # Natural breakpoints
                )
                chunking_method = 'semantic_chunking'
            
            chunks = text_splitter.split_text(document_text)
            logger.info(f"Created {len(chunks)} chunks using {chunking_method}")
            
            # Adjust batch size based on chunk size
            batch_size = 25 if is_legislative else 50
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(chunks))
                batch_chunks = chunks[start_idx:end_idx]
                
                logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_chunks)} chunks)")
                
                documents = []
                for i, chunk in enumerate(batch_chunks):
                    doc_metadata = metadata.copy()
                    doc_metadata['chunk_index'] = start_idx + i
                    doc_metadata['total_chunks'] = len(chunks)
                    doc_metadata['user_id'] = user_id
                    doc_metadata['upload_timestamp'] = datetime.utcnow().isoformat()
                    doc_metadata['chunk_size'] = len(chunk)
                    doc_metadata['chunking_method'] = chunking_method
                    doc_metadata['document_type'] = 'legislative' if is_legislative else 'general'
                    
                    # Extract bill numbers for legislative docs only
                    if is_legislative:
                        bill_numbers = re.findall(r'\b(?:HB|SB|SHB|SSB|ESHB|ESSB)\s+\d+', chunk)
                        if bill_numbers:
                            doc_metadata['contains_bills'] = ', '.join(bill_numbers)
                            logger.info(f"Chunk {start_idx + i} contains bills: {bill_numbers}")
                    
                    if file_id:
                        doc_metadata['file_id'] = file_id
                    
                    # Clean metadata for ChromaDB
                    clean_metadata = {}
                    for key, value in doc_metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            clean_metadata[key] = value
                        elif isinstance(value, list):
                            clean_metadata[key] = str(value)
                        elif value is None:
                            clean_metadata[key] = ""
                        else:
                            clean_metadata[key] = str(value)
                    
                    documents.append(Document(
                        page_content=chunk,
                        metadata=clean_metadata
                    ))
                
                # Add batch to ChromaDB
                try:
                    user_db.add_documents(documents)
                    logger.info(f"✅ Added batch {batch_num + 1} ({len(documents)} chunks)")
                except Exception as batch_error:
                    logger.error(f"❌ Batch {batch_num + 1} failed: {batch_error}")
                    return False
            
            logger.info(f"✅ Successfully added ALL {len(chunks)} chunks for document {file_id or 'unknown'}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in add_document_to_container: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def search_user_container(self, user_id: str, query: str, k: int = 5, document_id: str = None) -> List[Tuple]:
        """Search with timeout protection"""
        return self.search_user_container_safe(user_id, query, k, document_id)
    
    def search_user_container_safe(self, user_id: str, query: str, k: int = 5, document_id: str = None) -> List[Tuple]:
        """Search with enhanced error handling and timeout protection"""
        try:
            user_db = self.get_user_database_safe(user_id)
            if not user_db:
                logger.warning(f"No database available for user {user_id}")
                return []
            
            filter_dict = None
            if document_id:
                filter_dict = {"file_id": document_id}
            
            try:
                results = user_db.similarity_search_with_score(query, k=k, filter=filter_dict)
                return results
            except Exception as search_error:
                logger.warning(f"Search failed for user {user_id}: {search_error}")
                return []
                
        except Exception as e:
            logger.error(f"Error in safe container search for user {user_id}: {e}")
            return []
    
    def enhanced_search_user_container(self, user_id: str, query: str, conversation_context: str, k: int = 12, document_id: str = None) -> List[Tuple]:
        """Enhanced search with timeout protection and bill-specific optimization"""
        try:
            user_db = self.get_user_database_safe(user_id)
            if not user_db:
                return []
            
            filter_dict = None
            if document_id:
                filter_dict = {"file_id": document_id}
            
            try:
                # Check if this is a bill-specific query
                bill_match = re.search(r"\b(HB|SB|SSB|ESSB|SHB|ESHB)\s+(\d+)\b", query, re.IGNORECASE)
                
                if bill_match:
                    bill_number = f"{bill_match.group(1)} {bill_match.group(2)}"
                    logger.info(f"Bill-specific search for: {bill_number}")
                    
                    # First, try to find chunks that contain this specific bill
                    try:
                        all_docs = user_db.get()
                        bill_specific_chunks = []
                        
                        for i, (doc_id, metadata, content) in enumerate(zip(all_docs['ids'], all_docs['metadatas'], all_docs['documents'])):
                            if metadata and 'contains_bills' in metadata:
                                if bill_number in metadata['contains_bills']:
                                    # Create a document object for this chunk
                                    doc_obj = Document(page_content=content, metadata=metadata)
                                    # Use a high relevance score since we found exact bill match
                                    bill_specific_chunks.append((doc_obj, 0.95))  # High relevance for exact matches
                                    logger.info(f"Found {bill_number} in chunk {metadata.get('chunk_index')} with boosted score")
                        
                        if bill_specific_chunks:
                            logger.info(f"Using {len(bill_specific_chunks)} bill-specific chunks with high relevance")
                            # Get additional context chunks with lower threshold
                            regular_results = user_db.similarity_search_with_score(query, k=k, filter=filter_dict)
                            
                            # Combine bill-specific (high score) with regular results
                            all_results = bill_specific_chunks + regular_results
                            return remove_duplicate_documents(all_results)[:k]
                    except Exception as bill_search_error:
                        logger.warning(f"Bill-specific search failed, falling back to regular search: {bill_search_error}")
                        # Fall through to regular search
                
                # Fallback to regular search
                direct_results = user_db.similarity_search_with_score(query, k=k, filter=filter_dict)
                expanded_query = f"{query} {conversation_context}"
                expanded_results = user_db.similarity_search_with_score(expanded_query, k=k, filter=filter_dict)
                
                sub_query_results = []
                nlp = get_nlp()
                if nlp:
                    doc = nlp(query)
                    for ent in doc.ents:
                        if ent.label_ in ["ORG", "PERSON", "LAW", "DATE"]:
                            sub_results = user_db.similarity_search_with_score(f"What is {ent.text}?", k=3, filter=filter_dict)
                            sub_query_results.extend(sub_results)
                
                all_results = direct_results + expanded_results + sub_query_results
                return remove_duplicate_documents(all_results)[:k]
                
            except Exception as search_error:
                logger.warning(f"Enhanced search failed for user {user_id}: {search_error}")
                return []
            
        except Exception as e:
            logger.error(f"Error in enhanced user container search: {e}")
            return []

# Global instance
_container_manager = None

def initialize_container_manager():
    """Initialize the global container manager"""
    global _container_manager
    _container_manager = UserContainerManager(USER_CONTAINERS_PATH)
    return _container_manager

def get_container_manager() -> UserContainerManager:
    """Get the global container manager instance"""
    if _container_manager is None:
        return initialize_container_manager()
    return _container_manager
