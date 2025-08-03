"""User container management service"""
import os
import hashlib
import logging
import re
import traceback
import asyncio
from typing import Optional, List, Tuple, Dict, Callable
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
        self._db_cache = {}  # Cache database connections
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
        
        # Cache the new database
        self._db_cache[user_id] = user_db
        
        logger.info(f"Created container for user {user_id}: {container_id}")
        return container_id
    
    def get_container_id(self, user_id: str) -> str:
        """Get container ID for a user"""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    
    def _create_database(self, user_id: str) -> Optional[Chroma]:
        """Create database connection for a user"""
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
    
    def get_user_database(self, user_id: str) -> Optional[Chroma]:
        """Get user's database with caching"""
        # Check cache first
        if user_id in self._db_cache:
            logger.debug(f"Using cached database for user {user_id}")
            return self._db_cache[user_id]
        
        # Create and cache connection
        db = self._create_database(user_id)
        if db:
            self._db_cache[user_id] = db
            logger.info(f"Created and cached database connection for user {user_id}")
        return db
    
    def get_user_database_safe(self, user_id: str) -> Optional[Chroma]:
        """Get user database with enhanced error handling and recovery"""
        try:
            # Try to get from cache or create
            db = self.get_user_database(user_id)
            if db:
                return db
            
            # Database not found, try to create container
            container_id = self.get_container_id(user_id)
            container_path = os.path.join(self.base_path, container_id)
            
            if not os.path.exists(container_path):
                logger.warning(f"Container not found for user {user_id}, creating new one")
                self.create_user_container(user_id)
                return self._db_cache.get(user_id)  # Should be cached after creation
            
            # Ensure embeddings are available
            if not self.embeddings:
                self._initialize_embeddings()
            
            if not self.embeddings:
                logger.error("No embeddings model available for safe database access")
                return None
            
            # Try to create database connection
            db = Chroma(
                collection_name=f"user_{container_id}",
                embedding_function=self.embeddings,
                persist_directory=container_path
            )
            
            # Cache the connection
            self._db_cache[user_id] = db
            return db
            
        except Exception as e:
            logger.error(f"Error getting user database for {user_id}: {e}")
            try:
                logger.info(f"Attempting to recover by creating new container for {user_id}")
                # Clear any corrupted cache entry
                if user_id in self._db_cache:
                    del self._db_cache[user_id]
                
                self.create_user_container(user_id)
                return self._db_cache.get(user_id)
                
            except Exception as recovery_error:
                logger.error(f"Recovery failed for user {user_id}: {recovery_error}")
                return None
    
    def clear_cache(self, user_id: str = None):
        """Clear database cache for specific user or all users"""
        if user_id:
            if user_id in self._db_cache:
                del self._db_cache[user_id]
                logger.info(f"Cleared cache for user {user_id}")
        else:
            self._db_cache.clear()
            logger.info("Cleared all database cache")
    
    def get_cache_info(self) -> Dict:
        """Get information about current cache state"""
        return {
            'cached_users': list(self._db_cache.keys()),
            'cache_size': len(self._db_cache),
            'memory_usage': sum(1 for _ in self._db_cache.values())  # Simple count
        }
    
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
    
    def _detect_document_type(self, text: str) -> str:
        """Detect document type based on content patterns"""
        
        # Legislative patterns
        bill_count = len(re.findall(r'\b(?:HB|SB|SHB|SSB|ESHB|ESSB)\s+\d+', text))
        if bill_count > 1:
            return "legislative"
        
        # Legal document patterns
        legal_patterns = [
            r'\b(?:WHEREAS|NOW THEREFORE|AGREEMENT|CONTRACT|PARTY|PARTIES)\b',
            r'\b(?:Section|Article|Clause)\s+\d+',
            r'\b(?:shall|hereby|herein|thereof|whereof)\b'
        ]
        
        legal_score = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in legal_patterns)
        if legal_score > 10:
            return "legal"
        
        return "general"
    
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
                    'contains_bills': [bill_number],  # FIX: Make it a list for ChromaDB
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
                    'contains_bills': [bill_number],  # FIX: Make it a list for ChromaDB
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
                'contains_bills': [bill_number],  # FIX: Make it a list for ChromaDB
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
                                  chunk_size: int = 1500, overlap: int = 300) -> List[Dict]:
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
    
    def _chunk_legal_document(self, text: str, base_metadata: Dict) -> List[Dict]:
        """Chunk legal documents preserving structure"""
        chunks = []
        
        # Legal section patterns
        section_patterns = [
            (r'\n\s*(?:ARTICLE|Article)\s+([IVX\d]+)[^\n]*', 'article'),
            (r'\n\s*(?:SECTION|Section)\s+(\d+(?:\.\d+)?)[^\n]*', 'section'),
            (r'\n\s*(\d+)\.\s+[A-Z][^\n]*', 'numbered_section'),
            (r'\n\s*\(([a-z])\)\s*', 'subsection'),
        ]
        
        # Find all section boundaries
        boundaries = []
        for pattern, section_type in section_patterns:
            for match in re.finditer(pattern, text):
                boundaries.append({
                    'pos': match.start(),
                    'type': section_type,
                    'label': match.group(1),
                    'full_match': match.group(0)
                })
        
        # Sort boundaries by position
        boundaries.sort(key=lambda x: x['pos'])
        
        # Create chunks based on boundaries
        current_pos = 0
        current_section_hierarchy = []
        
        for i, boundary in enumerate(boundaries):
            # Extract text before this boundary
            chunk_text = text[current_pos:boundary['pos']].strip()
            
            if chunk_text and len(chunk_text) > 50:  # Minimum chunk size
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_type': 'legal_section',
                    'section_hierarchy': '/'.join(current_section_hierarchy),
                    'section_type': boundaries[i-1]['type'] if i > 0 else 'preamble',
                    'section_label': boundaries[i-1]['label'] if i > 0 else 'start',
                })
                
                # Split if too large
                if len(chunk_text) > 2000:
                    sub_chunks = self._split_large_section(chunk_text, chunk_metadata)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append({
                        'text': chunk_text,
                        'metadata': chunk_metadata
                    })
            
            # Update hierarchy
            if boundary['type'] in ['article', 'section']:
                current_section_hierarchy = [f"{boundary['type']}_{boundary['label']}"]
            elif boundary['type'] == 'subsection':
                if current_section_hierarchy:
                    current_section_hierarchy.append(f"subsection_{boundary['label']}")
            
            current_pos = boundary['pos']
        
        # Don't forget the last chunk
        if current_pos < len(text):
            chunk_text = text[current_pos:].strip()
            if chunk_text:
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_type': 'legal_section',
                    'section_hierarchy': '/'.join(current_section_hierarchy),
                    'section_type': 'final',
                })
                chunks.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
        
        return self._add_chunk_relationships(chunks)
    
    def _chunk_legislative_document(self, text: str, base_metadata: Dict) -> List[Dict]:
        """Chunk legislative documents preserving bill structure"""
        chunks = []
        
        # Find all bills
        bill_pattern = r'\b((?:HB|SB|SHB|SSB|ESHB|ESSB)\s+\d+)[^\n]*\n'
        bill_matches = list(re.finditer(bill_pattern, text))
        
        if not bill_matches:
            # No clear bill structure, fall back to general chunking
            return self._chunk_general_document(text, base_metadata)
        
        # Chunk by bills
        for i, match in enumerate(bill_matches):
            start_pos = match.start()
            end_pos = bill_matches[i + 1].start() if i + 1 < len(bill_matches) else len(text)
            
            bill_text = text[start_pos:end_pos].strip()
            bill_number = match.group(1)
            
            # Extract bill metadata
            bill_metadata = base_metadata.copy()
            bill_metadata.update({
                'chunk_type': 'legislative_bill',
                'bill_number': bill_number,
                'contains_bills': [bill_number],  # FIX: Make it a list for ChromaDB
            })
            
            # Extract additional bill info
            sponsor_match = re.search(r'Sponsors?\s*:\s*([^\n]+)', bill_text)
            if sponsor_match:
                bill_metadata['bill_sponsors'] = sponsor_match.group(1).strip()
            
            status_match = re.search(r'Final Status\s*:\s*([^\n]+)', bill_text)
            if status_match:
                bill_metadata['bill_status'] = status_match.group(1).strip()
            
            # Split if too large
            if len(bill_text) > 2000:
                sub_chunks = self._split_large_section(bill_text, bill_metadata)
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    'text': bill_text,
                    'metadata': bill_metadata
                })
        
        return self._add_chunk_relationships(chunks)
    
    def _chunk_general_document(self, text: str, base_metadata: Dict) -> List[Dict]:
        """Chunk general documents with sliding window and sentence boundaries"""
        chunks = []
        
        # First try paragraph-based chunking
        paragraphs = text.split('\n\n')
        
        if len(paragraphs) > 1:
            current_chunk = []
            current_size = 0
            
            for para in paragraphs:
                para_size = len(para)
                
                if current_size + para_size > 1000 and current_chunk:
                    # Save current chunk
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata['chunk_type'] = 'paragraph_based'
                    
                    chunks.append({
                        'text': chunk_text,
                        'metadata': chunk_metadata
                    })
                    
                    # Start new chunk with overlap
                    if len(current_chunk) > 1:
                        current_chunk = [current_chunk[-1], para]
                        current_size = len(current_chunk[-1]) + para_size
                    else:
                        current_chunk = [para]
                        current_size = para_size
                else:
                    current_chunk.append(para)
                    current_size += para_size
            
            # Add remaining
            if current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunk_metadata = base_metadata.copy()
                chunk_metadata['chunk_type'] = 'paragraph_based'
                chunks.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
        else:
            # Fall back to sliding window
            chunks = self._sliding_window_chunk(text, base_metadata, 1000, 200)
        
        return self._add_chunk_relationships(chunks)
    
    def _sliding_window_chunk(self, text: str, base_metadata: Dict, chunk_size: int, overlap: int) -> List[Dict]:
        """Sliding window chunking with sentence boundaries"""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunk_metadata = base_metadata.copy()
                chunk_metadata['chunk_type'] = 'sliding_window'
                
                chunks.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
                
                # Calculate overlap
                overlap_sentences = []
                overlap_size = 0
                
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_size = overlap_size + sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add remaining
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_metadata = base_metadata.copy()
            chunk_metadata['chunk_type'] = 'sliding_window'
            chunks.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
        
        return chunks
    
    def _split_large_section(self, text: str, metadata: Dict) -> List[Dict]:
        """Split large sections while preserving context"""
        return self._sliding_window_chunk(text, metadata, 1000, 300)
    
    def _add_chunk_relationships(self, chunks: List[Dict]) -> List[Dict]:
        """Add relationship metadata between chunks"""
        for i, chunk in enumerate(chunks):
            chunk['metadata']['chunk_index'] = i
            chunk['metadata']['total_chunks'] = len(chunks)
            
            if i > 0:
                chunk['metadata']['previous_chunk_index'] = i - 1
                # Add context from previous chunk
                prev_text = chunks[i-1]['text']
                chunk['metadata']['previous_context'] = prev_text[-200:] if len(prev_text) > 200 else prev_text
            
            if i < len(chunks) - 1:
                chunk['metadata']['next_chunk_index'] = i + 1
                # Add context from next chunk
                next_text = chunks[i+1]['text']
                chunk['metadata']['next_context'] = next_text[:200] if len(next_text) > 200 else next_text
        
        return chunks
    
    def hybrid_search_user_container(self, user_id: str, query: str, k: int = 10, document_id: str = None) -> List[Tuple]:
        """Perform hybrid search combining keyword and semantic search"""
        try:
            user_db = self.get_user_database_safe(user_id)
            if not user_db:
                logger.warning(f"No database available for user {user_id}")
                return []
            
            # Import hybrid searcher
            from .hybrid_search import get_hybrid_searcher
            searcher = get_hybrid_searcher()
            
            # Prepare filter
            filter_dict = None
            if document_id:
                filter_dict = {"file_id": document_id}
            
            # Perform hybrid search
            results = searcher.hybrid_search(
                query=query,
                vector_store=user_db,
                k=k,
                keyword_weight=0.3,
                semantic_weight=0.7,
                rerank=True,
                filter_dict=filter_dict
            )
            
            logger.info(f"Hybrid search returned {len(results)} results for query: '{query}'")
            
            # Log top results for debugging
            for i, (doc, score) in enumerate(results[:3]):
                logger.debug(f"Result {i+1}: Score={score:.3f}, Content preview: {doc.page_content[:100]}...")
            
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed, falling back to semantic search: {e}")
            # Fall back to enhanced semantic search
            return self.enhanced_search_user_container(user_id, query, "", k, document_id)
    
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
                                # Check if contains_bills contains our bill (handle both string and list formats)
                                contains_bills = metadata['contains_bills']
                                if isinstance(contains_bills, str):
                                    # Handle string format (old data)
                                    if bill_number in contains_bills:
                                        doc_obj = Document(page_content=content, metadata=metadata)
                                        bill_specific_chunks.append((doc_obj, 0.95))
                                        logger.info(f"Found {bill_number} in chunk {metadata.get('chunk_index')} with boosted score")
                                else:
                                    # Handle list format (new data) - convert to string for searching
                                    contains_bills_str = str(contains_bills)
                                    if bill_number in contains_bills_str:
                                        doc_obj = Document(page_content=content, metadata=metadata)
                                        bill_specific_chunks.append((doc_obj, 0.95))
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
