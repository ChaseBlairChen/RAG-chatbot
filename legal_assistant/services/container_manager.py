"""User container management service"""
import os
import hashlib
import logging
import re
import traceback
import asyncio  # ADD this import
from typing import Optional, List, Tuple, Dict, Callable  # UPDATE this line to include Callable
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
        """Add document to user's container with intelligent chunking"""
        try:
            user_db = self.get_user_database_safe(user_id)
            if not user_db:
                container_id = self.create_user_container(user_id)
                user_db = self.get_user_database_safe(user_id)
            
            # Use intelligent chunking
            chunks_with_metadata = self.intelligent_chunking(document_text, metadata)
            
            logger.info(f"Created {len(chunks_with_metadata)} chunks using intelligent chunking")
            
            # Process in batches
            batch_size = 25
            total_batches = (len(chunks_with_metadata) + batch_size - 1) // batch_size
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(chunks_with_metadata))
                batch_chunks = chunks_with_metadata[start_idx:end_idx]
                
                logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_chunks)} chunks)")
                
                documents = []
                for chunk_data in batch_chunks:
                    # Merge metadata
                    doc_metadata = chunk_data['metadata']
                    doc_metadata['user_id'] = user_id
                    doc_metadata['upload_timestamp'] = datetime.utcnow().isoformat()
                    doc_metadata['chunk_size'] = len(chunk_data['text'])
                    
                    if file_id:
                        doc_metadata['file_id'] = file_id
                    
                    # Generate chunk hash for deduplication
                    doc_metadata['content_hash'] = hashlib.md5(chunk_data['text'].encode()).hexdigest()[:16]
                    
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
                        page_content=chunk_data['text'],
                        metadata=clean_metadata
                    ))
                
                # Add batch to ChromaDB
                try:
                    user_db.add_documents(documents)
                    logger.info(f"✅ Added batch {batch_num + 1} ({len(documents)} chunks)")
                except Exception as batch_error:
                    logger.error(f"❌ Batch {batch_num + 1} failed: {batch_error}")
                    return False
            
            logger.info(f"✅ Successfully added ALL {len(chunks_with_metadata)} chunks for document {file_id or 'unknown'}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in add_document_to_container: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    # ADD THE NEW ASYNC METHOD HERE
    async def add_document_to_container_async(
        self, 
        user_id: str, 
        document_text: str, 
        metadata: Dict, 
        file_id: str = None,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> int:
        """Add document to container asynchronously with progress tracking"""
        try:
            user_db = self.get_user_database_safe(user_id)
            if not user_db:
                container_id = self.create_user_container(user_id)
                user_db = self.get_user_database_safe(user_id)
            
            # Use improved chunking
            chunks_with_metadata = self.intelligent_chunking_v2(document_text, metadata)
            
            logger.info(f"Created {len(chunks_with_metadata)} chunks using intelligent chunking v2")
            
            # Process in smaller batches for better performance
            batch_size = 10  # Smaller batches for faster processing
            total_batches = (len(chunks_with_metadata) + batch_size - 1) // batch_size
            chunks_added = 0
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(chunks_with_metadata))
                batch_chunks = chunks_with_metadata[start_idx:end_idx]
                
                # Process batch
                documents = []
                for chunk_data in batch_chunks:
                    doc_metadata = chunk_data['metadata'].copy()
                    doc_metadata.update({
                        'user_id': user_id,
                        'file_id': file_id,
                        'chunk_index': chunk_data['metadata']['chunk_index'],
                        'total_chunks': len(chunks_with_metadata),
                        'upload_timestamp': datetime.utcnow().isoformat(),
                        'chunk_size': len(chunk_data['text'])
                    })
                    
                    # Generate chunk hash for deduplication
                    doc_metadata['content_hash'] = hashlib.md5(chunk_data['text'].encode()).hexdigest()[:16]
                    
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
                        page_content=chunk_data['text'],
                        metadata=clean_metadata
                    ))
                
                # Add batch
                user_db.add_documents(documents)
                chunks_added += len(documents)
                
                # Update progress
                if progress_callback:
                    progress = int((chunks_added / len(chunks_with_metadata)) * 100)
                    progress_callback(progress)
                
                # Yield control to prevent blocking
                await asyncio.sleep(0.1)
            
            return chunks_added
            
        except Exception as e:
            logger.error(f"Error in async document addition: {e}")
            raise
    
    def intelligent_chunking(self, text: str, metadata: Dict, doc_type: str = "general") -> List[Dict]:
        """Intelligent chunking based on document structure with metadata preservation"""
        
        # Detect document type if not specified
        if doc_type == "general":
            doc_type = self._detect_document_type(text)
        
        logger.info(f"Using {doc_type} chunking strategy")
        
        if doc_type == "legal":
            return self._chunk_legal_document(text, metadata)
        elif doc_type == "legislative":
            return self._chunk_legislative_document(text, metadata)
        else:
            return self._chunk_general_document(text, metadata)
    
    # ADD THE NEW INTELLIGENT CHUNKING V2 METHOD HERE
    def intelligent_chunking_v2(self, text: str, metadata: Dict) -> List[Dict]:
        """Improved chunking with better context preservation"""
        
        # Detect document type
        doc_type = self._detect_document_type(text)
        logger.info(f"Document type detected: {doc_type}")
        
        # For legislative documents, use bill-aware chunking
        if doc_type == "legislative":
            return self._chunk_legislative_smart(text, metadata)
        elif doc_type == "legal":
            return self._chunk_legal_smart(text, metadata)
        else:
            return self._chunk_with_sliding_window(text, metadata)
    
    # ADD ALL THE OTHER NEW METHODS HERE (inside the class)
    def _chunk_legislative_smart(self, text: str, base_metadata: Dict) -> List[Dict]:
        """Smart chunking for legislative documents that keeps bills together"""
        chunks = []
        
        # Find all bill boundaries
        bill_pattern = r'(?=(?:HB|SB|SHB|SSB|ESHB|ESSB)\s+\d+)'
        bill_positions = [match.start() for match in re.finditer(bill_pattern, text)]
        
        if not bill_positions:
            # No bills found, use regular chunking
            return self._chunk_with_sliding_window(text, base_metadata)
        
        # Add end position
        bill_positions.append(len(text))
        
        # Process each bill as a unit
        for i in range(len(bill_positions) - 1):
            start = bill_positions[i]
            end = bill_positions[i + 1]
            bill_text = text[start:end].strip()
            
            # Extract bill number
            bill_match = re.match(r'((?:HB|SB|SHB|SSB|ESHB|ESSB)\s+\d+)', bill_text)
            bill_number = bill_match.group(1) if bill_match else f"Section_{i}"
            
            # If bill is too large, chunk it intelligently
            if len(bill_text) > 2000:
                # Find natural break points within the bill
                sub_chunks = self._chunk_bill_intelligently(bill_text, bill_number, base_metadata)
                chunks.extend(sub_chunks)
            else:
                # Keep entire bill as one chunk
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_type': 'legislative_bill',
                    'bill_number': bill_number,
                    'contains_bills': bill_number,
                    'chunk_index': len(chunks),
                    'is_complete_bill': True
                })
                
                chunks.append({
                    'text': bill_text,
                    'metadata': chunk_metadata
                })
        
        return chunks
    
    def _chunk_bill_intelligently(self, bill_text: str, bill_number: str, base_metadata: Dict) -> List[Dict]:
        """Intelligently chunk a single bill while preserving context"""
        chunks = []
        
        # Look for section breaks within the bill
        section_pattern = r'\n\s*(?:Section|SECTION|Sec\.)\s+\d+'
        sections = re.split(section_pattern, bill_text)
        
        # Always include bill header in each chunk
        bill_header = bill_text[:200] if len(bill_text) > 200 else bill_text[:50]
        if '\n' in bill_header:
            bill_header = bill_header[:bill_header.find('\n')]
        
        current_chunk = f"[Bill {bill_number}]\n{bill_header}\n...\n\n"
        current_size = len(current_chunk)
        chunk_parts = []
        
        for section in sections:
            section_size = len(section)
            
            if current_size + section_size > 1500 and chunk_parts:
                # Save current chunk
                chunk_text = current_chunk + '\n\n'.join(chunk_parts)
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_type': 'legislative_bill_part',
                    'bill_number': bill_number,
                    'contains_bills': bill_number,
                    'chunk_index': len(chunks),
                    'part_number': len(chunks) + 1
                })
                
                chunks.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
                
                # Start new chunk with bill reference
                chunk_parts = [section]
                current_size = len(current_chunk) + section_size
            else:
                chunk_parts.append(section)
                current_size += section_size
        
        # Add remaining parts
        if chunk_parts:
            chunk_text = current_chunk + '\n\n'.join(chunk_parts)
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_type': 'legislative_bill_part',
                'bill_number': bill_number,
                'contains_bills': bill_number,
                'chunk_index': len(chunks),
                'part_number': len(chunks) + 1,
                'is_final_part': True
            })
            
            chunks.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
        
        return chunks
    
    def _chunk_legal_smart(self, text: str, base_metadata: Dict) -> List[Dict]:
        """Smart chunking for legal documents"""
        # Use your existing _chunk_legal_document method
        return self._chunk_legal_document(text, base_metadata)
    
    def _chunk_with_sliding_window(self, text: str, base_metadata: Dict, 
                                  chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Improved sliding window with sentence boundaries"""
        chunks = []
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if not sentences:
            return [{
                'text': text,
                'metadata': base_metadata.copy()
            }]
        
        current_chunk = []
        current_size = 0
        
        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence)
            
            # If adding this sentence exceeds chunk size
            if current_size + sentence_size > chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_type': 'sliding_window',
                    'chunk_index': len(chunks),
                    'sentence_start': i - len(current_chunk),
                    'sentence_end': i
                })
                
                chunks.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
                
                # Calculate overlap sentences
                overlap_size = 0
                overlap_sentences = []
                
                for sent in reversed(current_chunk):
                    if overlap_size + len(sent) <= overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_size += len(sent)
                    else:
                        break
                
                # Start new chunk with overlap
                current_chunk = overlap_sentences + [sentence]
                current_size = overlap_size + sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_type': 'sliding_window',
                'chunk_index': len(chunks),
                'is_final_chunk': True
            })
            
            chunks.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
        
        return chunks
    
    # KEEP ALL YOUR EXISTING METHODS BELOW (don't change anything)
    def _detect_document_type(self, text: str) -> str:
        # ... your existing code ...
    
    def _chunk_legal_document(self, text: str, base_metadata: Dict) -> List[Dict]:
        # ... your existing code ...
    
    # ... rest of your existing methods ...

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
