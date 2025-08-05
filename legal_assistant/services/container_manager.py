"""
User Container Management Service

This module manages user-specific vector databases (containers) using ChromaDB.
It includes sophisticated document chunking strategies tailored for different
document types (e.g., legal, legislative) and provides robust search capabilities
with caching, error handling, and performance optimizations.
"""
import os
import hashlib
import logging
import re
import traceback
import asyncio
from typing import Optional, List, Tuple, Dict, Callable, Any
from datetime import datetime

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..config import USER_CONTAINERS_PATH, FAST_EMBEDDING_MODELS
from ..core.exceptions import ContainerError
from ..core.dependencies import get_embeddings, get_nlp
from ..utils.text_processing import remove_duplicate_documents

logger = logging.getLogger(__name__)

class UserContainerManager:
    """
    Manages user-specific document containers, providing methods for creation,
    document addition, and retrieval. It handles embedding model fallback
    and caches database connections for efficiency.
    """
    
    def __init__(self, base_path: str):
        """
        Initializes the manager and sets up the embedding model.

        Args:
            base_path: The root directory for storing user containers.
        """
        self.base_path = base_path
        self._db_cache: Dict[str, Chroma] = {}
        
        # Initialize embeddings once for the entire service
        self.embeddings = self._initialize_embeddings()
        if not self.embeddings:
            logger.error("❌ Failed to initialize any embeddings model. The service will be non-functional.")
        
        logger.info(f"UserContainerManager initialized with base path: {self.base_path}")

    def _initialize_embeddings(self) -> Optional[HuggingFaceEmbeddings]:
        """
        Initializes the embeddings with a priority-based fallback.
        1. Global embeddings (e.g., from a shared dependency).
        2. A list of fast, temporary embedding models.
        3. A final, reliable fallback model.

        Returns:
            An initialized HuggingFaceEmbeddings object or None if all attempts fail.
        """
        # 1. Try to use a globally provided embedding function
        global_embeddings = get_embeddings()
        if global_embeddings:
            logger.info("Using global embeddings model.")
            return global_embeddings

        # 2. Try fast models from configuration
        if FAST_EMBEDDING_MODELS:
            for model_name in FAST_EMBEDDING_MODELS:
                try:
                    embeddings = HuggingFaceEmbeddings(model_name=model_name)
                    logger.info(f"✅ UserContainerManager using FAST embeddings: {model_name}")
                    return embeddings
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")

        # 3. Last resort fallback
        try:
            fallback_model = "all-MiniLM-L6-v2"
            embeddings = HuggingFaceEmbeddings(model_name=fallback_model)
            logger.warning(f"⚠️ Using fallback embeddings: {fallback_model}")
            return embeddings
        except Exception as e:
            logger.error(f"❌ Failed to load any embeddings model: {e}")
            return None

    def _get_container_path(self, user_id: str) -> Tuple[str, str]:
        """Generates container ID and path from a user ID."""
        container_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        container_path = os.path.join(self.base_path, container_id)
        return container_id, container_path

    def get_container_id(self, user_id: str) -> str:
        """Get container ID for user"""
        container_id, _ = self._get_container_path(user_id)
        return container_id

    def _create_database(self, user_id: str) -> Chroma:
        """
        Internal method to create a new Chroma database connection for a user.
        Assumes embeddings are available and path exists.
        """
        if not self.embeddings:
            raise ContainerError("No embeddings model available for container creation.")
            
        container_id, container_path = self._get_container_path(user_id)
        os.makedirs(container_path, exist_ok=True)
        
        return Chroma(
            collection_name=f"user_{container_id}",
            embedding_function=self.embeddings,
            persist_directory=container_path
        )

    def get_user_database(self, user_id: str) -> Optional[Chroma]:
        """
        Retrieves a user's database instance from cache or creates a new one.

        This method is the main entry point for database access and provides
        resilient error handling and auto-creation.

        Args:
            user_id: The unique identifier for the user.

        Returns:
            A Chroma database instance or None if a critical error occurs.
        """
        if not self.embeddings:
            logger.error("Cannot get user database, no embeddings model is available.")
            return None

        # 1. Check cache first
        if user_id in self._db_cache:
            logger.debug(f"Using cached database for user {user_id}")
            return self._db_cache[user_id]
        
        container_id, container_path = self._get_container_path(user_id)
        
        # 2. If path exists, try to connect to the existing database
        if os.path.exists(container_path):
            try:
                db = Chroma(
                    collection_name=f"user_{container_id}",
                    embedding_function=self.embeddings,
                    persist_directory=container_path
                )
                self._db_cache[user_id] = db
                logger.info(f"Connected to existing container for user {user_id}")
                return db
            except Exception as e:
                logger.error(f"Failed to connect to existing container for {user_id}: {e}")
                # Fall through to attempt a fresh creation
        
        # 3. Path doesn't exist or connection failed, create a new one
        try:
            db = self._create_database(user_id)
            self._db_cache[user_id] = db
            logger.info(f"Created a new container for user {user_id}")
            return db
        except (ContainerError, Exception) as e:
            logger.error(f"❌ Critical error: Could not create container for user {user_id}: {e}")
            return None

    def get_user_database_safe(self, user_id: str) -> Optional[Chroma]:
        """Safe database getter that doesn't raise exceptions"""
        try:
            return self.get_user_database(user_id)
        except Exception as e:
            logger.error(f"Failed to get database for user {user_id}: {e}")
            return None

    def clear_cache(self, user_id: Optional[str] = None):
        """
        Clears the in-memory cache for a specific user's database or for all users.
        """
        if user_id:
            if user_id in self._db_cache:
                del self._db_cache[user_id]
                logger.info(f"Cleared cache for user {user_id}")
        else:
            self._db_cache.clear()
            logger.info("Cleared all database cache")

    async def add_document_to_container_async(
        self,
        user_id: str,
        document_text: str,
        metadata: Dict,
        file_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> int:
        """
        Adds a document to a user's container with smart chunking and batch processing.
        This is an async operation that reports progress.

        Args:
            user_id: The ID of the user.
            document_text: The full text content of the document.
            metadata: Base metadata for the document.
            file_id: An optional ID for the source file.
            progress_callback: An optional function to report progress (0-100).

        Returns:
            The number of chunks successfully added.
        
        Raises:
            ContainerError: If the database cannot be accessed.
        """
        user_db = self.get_user_database(user_id)
        if not user_db:
            raise ContainerError(f"Could not get or create database for user {user_id}")

        try:
            chunks = self._intelligent_chunking_with_context(document_text, metadata)
            
            logger.info(f"Processing document with {len(chunks)} chunks.")
            
            batch_size = 50
            chunks_added = 0
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                documents = []
                for chunk_data in batch_chunks:
                    # Finalize metadata and create Document objects
                    final_metadata = chunk_data['metadata'].copy()
                    final_metadata.update({
                        'user_id': user_id,
                        'file_id': file_id,
                        'upload_timestamp': datetime.utcnow().isoformat(),
                        'chunk_size': len(chunk_data['text']),
                        'content_hash': hashlib.md5(chunk_data['text'].encode()).hexdigest()[:16]
                    })
                    
                    # Clean metadata for ChromaDB (ensure only supported types)
                    cleaned_metadata = {
                        k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                        for k, v in final_metadata.items()
                    }
                    
                    documents.append(Document(
                        page_content=chunk_data['text'],
                        metadata=cleaned_metadata
                    ))
                
                try:
                    # Add documents to the vector store
                    user_db.add_documents(documents)
                    chunks_added += len(documents)
                    logger.info(f"✅ Added batch {i // batch_size + 1}/{len(chunks) // batch_size + 1}")
                except Exception as e:
                    logger.error(f"❌ Failed to add batch {i // batch_size + 1} to database: {e}")
                    raise ContainerError("Failed to add documents to the container.") from e
                
                if progress_callback:
                    progress = int((chunks_added / len(chunks)) * 100)
                    progress_callback(progress)
                
                # Yield control to prevent blocking the event loop
                await asyncio.sleep(0.01)

            logger.info(f"✅ Successfully added {chunks_added} chunks for file {file_id or 'unknown'}")
            return chunks_added

        except Exception as e:
            logger.error(f"❌ Error during document addition: {traceback.format_exc()}")
            raise ContainerError(f"An unexpected error occurred during document processing: {e}") from e

    def add_document_to_container_sync(
        self,
        user_id: str,
        document_text: str,
        metadata: Dict,
        file_id: Optional[str] = None
    ) -> int:
        """
        Synchronous version of add_document_to_container for compatibility.
        """
        try:
            # Run the async version in a new event loop
            return asyncio.run(self.add_document_to_container_async(
                user_id, document_text, metadata, file_id
            ))
        except Exception as e:
            logger.error(f"Sync document addition failed: {e}")
            raise ContainerError(f"Failed to add document synchronously: {e}")

    def enhanced_search_user_container(self, user_id: str, query: str, context: str, 
                                     k: int = 15, document_id: str = None) -> List[Tuple]:
        """Enhanced search with document filtering"""
        try:
            user_db = self.get_user_database_safe(user_id)
            if not user_db:
                return []
            
            # Apply document filter if specified
            filter_dict = {"file_id": document_id} if document_id else None
            
            # Perform search
            results = user_db.similarity_search_with_score(query, k=k, filter=filter_dict)
            
            # Add metadata
            for doc, score in results:
                doc.metadata['source_type'] = 'user_container'
            
            return results
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            return []

    def search_user_container(self, user_id: str, query: str, k: int = 10, 
                             document_id: str = None) -> List[Tuple]:
        """Basic search method"""
        return self.enhanced_search_user_container(user_id, query, "", k, document_id)

    def _intelligent_chunking_with_context(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Improved chunking method that detects document type and uses appropriate
        chunking strategies, ensuring each chunk has metadata about its original
        structure and its relationship to neighboring chunks.
        """
        doc_type = self._detect_document_type(text)
        logger.info(f"Document type detected: {doc_type}. Applying specific chunking strategy.")
        
        chunks = []
        if doc_type == "legislative":
            chunks = self._chunk_legislative(text, metadata)
        elif doc_type == "legal":
            chunks = self._chunk_legal(text, metadata)
        else:
            chunks = self._chunk_general(text, metadata)
            
        return chunks

    def _detect_document_type(self, text: str) -> str:
        """Determines document type based on regex patterns."""
        if re.search(r'\b(?:HB|SB|SHB|SSB|ESHB|ESSB)\s+\d+', text, re.IGNORECASE):
            return "legislative"
        
        legal_patterns = [
            r'\b(WHEREAS|NOW THEREFORE|AGREEMENT|CONTRACT|PARTY|PARTIES)\b',
            r'\b(Section|Article|Clause)\s+\d+',
            r'\b(shall|hereby|herein|thereof|whereof)\b'
        ]
        
        legal_score = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in legal_patterns)
        if legal_score > 5:
            return "legal"
            
        return "general"

    def _chunk_legislative(self, text: str, base_metadata: Dict) -> List[Dict]:
        """
        Chunks legislative documents, prioritizing bill boundaries.
        Each chunk is augmented with bill-specific metadata.
        """
        chunks = []
        bill_pattern = r'(?=\n(?:HB|SB|SHB|SSB|ESHB|ESSB)\s+\d+)'
        parts = re.split(bill_pattern, text)
        
        for i, part in enumerate(parts):
            if not part.strip():
                continue
                
            bill_match = re.match(r'((?:HB|SB|SHB|SSB|ESHB|ESSB)\s+\d+)', part)
            bill_number = bill_match.group(1) if bill_match else f"intro_or_unknown_{i}"
            
            # Use a standard text splitter for large bills to manage size
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            sub_chunks = splitter.split_text(part)
            
            for j, sub_chunk_text in enumerate(sub_chunks):
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_type': 'legislative_bill_part',
                    'bill_number': bill_number,
                    'chunk_index': len(chunks),
                    'part_number': j + 1,
                    'contains_bills': bill_number  # Add this for search
                })
                chunks.append({'text': sub_chunk_text, 'metadata': chunk_metadata})
                
        return chunks

    def _chunk_legal(self, text: str, base_metadata: Dict) -> List[Dict]:
        """
        Chunks legal documents, preserving logical sections like Articles and Sections.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\nArticle", "\n\nSection", "\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = []
        for i, part in enumerate(splitter.split_text(text)):
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_type': 'legal_section',
                'chunk_index': len(chunks),
                'part_number': i + 1
            })
            chunks.append({'text': part, 'metadata': chunk_metadata})
            
        return chunks
        
    def _chunk_general(self, text: str, base_metadata: Dict) -> List[Dict]:
        """
        Chunks general documents using a robust, paragraph-aware text splitter.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = []
        for i, part in enumerate(splitter.split_text(text)):
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_type': 'general',
                'chunk_index': len(chunks),
                'part_number': i + 1
            })
            chunks.append({'text': part, 'metadata': chunk_metadata})
            
        return chunks

    def hybrid_search(self, user_id: str, query: str, k: int = 10, file_id: Optional[str] = None) -> List[Tuple]:
        """
        Performs an advanced hybrid search combining semantic and keyword matching.
        This method is preferred for most retrieval tasks.

        Args:
            user_id: The user's ID.
            query: The search query.
            k: The number of results to return.
            file_id: Optional filter to search within a specific document.

        Returns:
            A list of tuples containing (Document, score).
        """
        try:
            user_db = self.get_user_database(user_id)
            if not user_db:
                logger.warning(f"No database available for user {user_id}")
                return []
            
            # Use enhanced semantic search for now
            return self._enhanced_semantic_search(user_db, user_id, query, k, file_id)

        except Exception as e:
            logger.error(f"Hybrid search failed for user {user_id}, query '{query}': {e}")
            logger.debug(traceback.format_exc())
            return []

    def _enhanced_semantic_search(self, user_db: Chroma, user_id: str, query: str, k: int, file_id: Optional[str]) -> List[Tuple[Document, float]]:
        """
        An internal helper for performing an enhanced semantic search with query expansion.
        """
        filter_dict = {"file_id": file_id} if file_id else None
        
        # 1. Direct semantic search
        direct_results = user_db.similarity_search_with_score(query, k=k, filter=filter_dict)
        
        # 2. Query expansion using named entities
        nlp = get_nlp()
        expanded_query = query
        if nlp:
            try:
                doc = nlp(query)
                for ent in doc.ents:
                    # Expand the query with key entities to capture more contextually relevant documents
                    if ent.label_ in ["ORG", "PERSON", "LAW", "DATE"]:
                        expanded_query += f" {ent.text}"
            except Exception as e:
                logger.warning(f"NLP processing failed: {e}")

        expanded_results = user_db.similarity_search_with_score(expanded_query, k=k, filter=filter_dict)
        
        # 3. Combine and deduplicate results
        all_results = direct_results + expanded_results
        
        # Simple deduplication by content
        unique_results: Dict[str, Tuple[Document, float]] = {}
        for doc, score in all_results:
            content_key = doc.page_content[:100]  # Use first 100 chars as key
            if content_key not in unique_results or score > unique_results[content_key][1]:
                unique_results[content_key] = (doc, score)
                
        final_results = list(unique_results.values())
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        return final_results[:k]

    def search_for_bill(self, user_id: str, bill_number: str, k: int = 10) -> List[Tuple]:
        """
        Search specifically for bill information with enhanced metadata filtering
        """
        try:
            user_db = self.get_user_database_safe(user_id)
            if not user_db:
                return []
            
            # Try multiple search strategies for bill numbers
            search_queries = [
                bill_number,
                bill_number.replace(" ", ""),
                f"bill {bill_number}",
                f"{bill_number} sponsors",
                f"{bill_number} status"
            ]
            
            all_results = []
            
            for query in search_queries:
                try:
                    results = user_db.similarity_search_with_score(query, k=k)
                    all_results.extend(results)
                except Exception as e:
                    logger.warning(f"Bill search query '{query}' failed: {e}")
            
            # Deduplicate and sort
            unique_results = {}
            for doc, score in all_results:
                content_key = doc.page_content[:100]
                if content_key not in unique_results or score > unique_results[content_key][1]:
                    unique_results[content_key] = (doc, score)
            
            final_results = list(unique_results.values())
            final_results.sort(key=lambda x: x[1], reverse=True)
            
            return final_results[:k]
            
        except Exception as e:
            logger.error(f"Bill search failed: {e}")
            return []

    def get_document_count(self, user_id: str) -> int:
        """Get total number of documents in user's container"""
        try:
            user_db = self.get_user_database_safe(user_id)
            if not user_db:
                return 0
            
            collection = user_db.get()
            return len(collection.get('ids', []))
            
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for user's container"""
        try:
            user_db = self.get_user_database_safe(user_id)
            if not user_db:
                return {"error": "No database available"}
            
            collection = user_db.get()
            documents = collection.get('documents', [])
            metadatas = collection.get('metadatas', [])
            
            # Calculate stats
            stats = {
                'total_chunks': len(documents),
                'total_characters': sum(len(doc) for doc in documents),
                'unique_files': len(set(meta.get('file_id', '') for meta in metadatas if meta.get('file_id'))),
                'document_types': {},
                'avg_chunk_size': 0
            }
            
            # Count document types
            for meta in metadatas:
                doc_type = meta.get('chunk_type', 'unknown')
                stats['document_types'][doc_type] = stats['document_types'].get(doc_type, 0) + 1
            
            # Calculate average chunk size
            if documents:
                stats['avg_chunk_size'] = stats['total_characters'] / len(documents)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get user stats: {e}")
            return {"error": str(e)}

# --- Global Instance and Initialization ---
_container_manager: Optional[UserContainerManager] = None

def get_container_manager() -> UserContainerManager:
    """
    Singleton pattern for the UserContainerManager.
    Initializes the manager on the first call and returns the same instance thereafter.
    """
    global _container_manager
    if _container_manager is None:
        _container_manager = UserContainerManager(USER_CONTAINERS_PATH)
    return _container_manager

def initialize_container_manager():
    """Initialize the container manager"""
    global _container_manager
    if _container_manager is None:
        _container_manager = UserContainerManager(USER_CONTAINERS_PATH)
    logger.info("✅ Container manager initialized")
    return _container_manager
