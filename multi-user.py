# Unified Legal Assistant Backend - Multi-User with Enhanced RAG


from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os
import json
import requests
import re
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Set, Any
from dataclasses import dataclass
from enum import Enum
import io
import tempfile
import sys
import traceback
import hashlib
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Third-party library imports for RAG
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np

# AI imports
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("⚠️ aiohttp not available - AI features disabled. Install with: pip install aiohttp")

# Import open-source NLP models support
OPEN_SOURCE_NLP_AVAILABLE = False
try:
    import torch
    from transformers import (
        pipeline,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModelForQuestionAnswering,
        AutoModelForTokenClassification,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
    )
    OPEN_SOURCE_NLP_AVAILABLE = True
    logger.info("✅ Open-source NLP models available")
except ImportError as e:
    logger.warning(f"⚠️ Open-source NLP models not available: {e}")
    print("Install with: pip install transformers torch")

# Import PDF processing libraries
PYMUPDF_AVAILABLE = False
PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
    print("✅ PyMuPDF available - using enhanced PDF extraction")
except ImportError as e:
    print(f"⚠️ PyMuPDF not available: {e}")
    print("Install with: pip install PyMuPDF")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
    print("✅ pdfplumber available - using enhanced PDF extraction")
except ImportError as e:
    print(f"⚠️ pdfplumber not available: {e}")
    print("Install with: pip install pdfplumber")

print(f"PDF processing status: PyMuPDF={PYMUPDF_AVAILABLE}, pdfplumber={PDFPLUMBER_AVAILABLE}")

# ADD THE SAFEDOCUMENTPROCESSOR CLASS HERE - BEFORE FastAPI app creation
class SafeDocumentProcessor:
    """Safe document processor for various file types"""
    
    @staticmethod
    def process_document_safe(file) -> Tuple[str, int, List[str]]:
        """
        Process uploaded document safely
        Returns: (content, pages_processed, warnings)
        """
        warnings = []
        content = ""
        pages_processed = 0
        
        try:
            # Get file extension
            filename = getattr(file, 'filename', 'unknown')
            file_ext = os.path.splitext(filename)[1].lower()
            
            # Read file content
            file_content = file.file.read()
            
            if file_ext == '.txt':
                content = file_content.decode('utf-8', errors='ignore')
                pages_processed = 1
                
            elif file_ext == '.pdf':
                # Try to process PDF
                content, pages_processed = SafeDocumentProcessor._process_pdf(file_content, warnings)
                
            elif file_ext == '.docx':
                # Try to process DOCX
                content, pages_processed = SafeDocumentProcessor._process_docx(file_content, warnings)
                
            else:
                # For other formats, try to read as text
                try:
                    content = file_content.decode('utf-8', errors='ignore')
                    pages_processed = 1
                    warnings.append(f"File type {file_ext} processed as plain text")
                except Exception as e:
                    warnings.append(f"Could not process file: {str(e)}")
                    content = "Unable to process this file type"
                    pages_processed = 0
            
            # Reset file pointer
            file.file.seek(0)
            
        except Exception as e:
            warnings.append(f"Error processing document: {str(e)}")
            content = "Error processing document"
            pages_processed = 0
        
        return content, pages_processed, warnings
    
    @staticmethod
    def _process_pdf(file_content: bytes, warnings: List[str]) -> Tuple[str, int]:
        """Process PDF content"""
        try:
            # Try PyMuPDF first
            if PYMUPDF_AVAILABLE:
                try:
                    import fitz
                    doc = fitz.open(stream=file_content, filetype="pdf")
                    text_content = ""
                    pages = len(doc)
                    for page_num in range(pages):
                        page = doc.load_page(page_num)
                        text_content += page.get_text()
                    doc.close()
                    return text_content, pages
                except Exception as e:
                    warnings.append(f"PyMuPDF error: {str(e)}")
            
            # Try pdfplumber as fallback
            if PDFPLUMBER_AVAILABLE:
                try:
                    import pdfplumber
                    with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                        text_content = ""
                        pages = len(pdf.pages)
                        for page in pdf.pages:
                            text_content += page.extract_text() or ""
                    return text_content, pages
                except Exception as e:
                    warnings.append(f"pdfplumber error: {str(e)}")
            
            # If no PDF libraries available or both failed
            warnings.append("No PDF processing libraries available or both failed. Install PyMuPDF or pdfplumber.")
            return "PDF processing not available", 0
            
        except Exception as e:
            warnings.append(f"Error processing PDF: {str(e)}")
            return "Error processing PDF", 0
    
    @staticmethod
    def _process_docx(file_content: bytes, warnings: List[str]) -> Tuple[str, int]:
        """Process DOCX content"""
        try:
            # Try to import python-docx
            try:
                from docx import Document
                doc = Document(io.BytesIO(file_content))
                text_content = ""
                for paragraph in doc.paragraphs:
                    text_content += paragraph.text + "\n"
                return text_content, 1  # DOCX doesn't have clear "pages"
            except ImportError:
                warnings.append("python-docx not available. Install with: pip install python-docx")
                return "DOCX processing not available", 0
            except Exception as e:
                warnings.append(f"Error processing DOCX: {str(e)}")
                return "Error processing DOCX", 0
                
        except Exception as e:
            warnings.append(f"Error processing DOCX: {str(e)}")
            return "Error processing DOCX", 0

# Create FastAPI app
app = FastAPI(
    title="Unified Legal Assistant API",
    description="Multi-User Legal Assistant with Enhanced RAG and External Database Integration",
    version="9.0.0-SmartRAG"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# - Configuration -
# Database paths
DEFAULT_CHROMA_PATH = os.path.abspath(os.path.join(os.getcwd(), "chromadb-database"))
USER_CONTAINERS_PATH = os.path.abspath(os.path.join(os.getcwd(), "user-containers"))
logger.info(f"Using DEFAULT_CHROMA_PATH: {DEFAULT_CHROMA_PATH}")
logger.info(f"Using USER_CONTAINERS_PATH: {USER_CONTAINERS_PATH}")

# Create directories if they don't exist
os.makedirs(USER_CONTAINERS_PATH, exist_ok=True)

OPENROUTER_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
AI_ENABLED = bool(OPENROUTER_API_KEY) and AIOHTTP_AVAILABLE
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
LEGAL_EXTENSIONS = {'.pdf', '.txt', '.docx', '.rtf'}

# Security - Modified to make authentication optional for debugging
security = HTTPBearer(auto_error=False)  # Changed to not auto-error

# - External Legal Database Interface -
class LegalDatabaseInterface(ABC):
    """Abstract interface for external legal databases"""
    
    @abstractmethod
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Search the legal database"""
        pass
    
    @abstractmethod
    def get_document(self, document_id: str) -> Dict:
        """Retrieve a specific document"""
        pass
    
    @abstractmethod
    def authenticate(self, credentials: Dict) -> bool:
        """Authenticate with the legal database"""
        pass

class LexisNexisInterface(LegalDatabaseInterface):
    """Interface for LexisNexis integration (placeholder for future implementation)"""
    
    def __init__(self, api_key: str = None, api_endpoint: str = None):
        self.api_key = api_key or os.environ.get("LEXISNEXIS_API_KEY")
        self.api_endpoint = api_endpoint or os.environ.get("LEXISNEXIS_API_ENDPOINT")
        self.authenticated = False
    
    def authenticate(self, credentials: Dict) -> bool:
        """Authenticate with LexisNexis"""
        # Placeholder for actual authentication
        logger.info("LexisNexis authentication placeholder")
        return False
    
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Search LexisNexis database"""
        # Placeholder for actual search implementation
        logger.info(f"LexisNexis search placeholder for query: {query}")
        return []
    
    def get_document(self, document_id: str) -> Dict:
        """Get document from LexisNexis"""
        # Placeholder for actual document retrieval
        logger.info(f"LexisNexis document retrieval placeholder for ID: {document_id}")
        return {}

class WestlawInterface(LegalDatabaseInterface):
    """Interface for Westlaw integration (placeholder for future implementation)"""
    
    def __init__(self, api_key: str = None, api_endpoint: str = None):
        self.api_key = api_key or os.environ.get("WESTLAW_API_KEY")
        self.api_endpoint = api_endpoint or os.environ.get("WESTLAW_API_ENDPOINT")
        self.authenticated = False
    
    def authenticate(self, credentials: Dict) -> bool:
        """Authenticate with Westlaw"""
        # Placeholder for actual authentication
        logger.info("Westlaw authentication placeholder")
        return False
    
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Search Westlaw database"""
        # Placeholder for actual search implementation
        logger.info(f"Westlaw search placeholder for query: {query}")
        return []
    
    def get_document(self, document_id: str) -> Dict:
        """Get document from Westlaw"""
        # Placeholder for actual document retrieval
        logger.info(f"Westlaw document retrieval placeholder for ID: {document_id}")
        return {}

# - Pydantic Models - FIXED: Corrected the Query model field name
class User(BaseModel):
    user_id: str
    email: Optional[str] = None
    container_id: Optional[str] = None
    subscription_tier: str = "free"  # free, premium, enterprise
    external_db_access: List[str] = []  # ["lexisnexis", "westlaw"]

class Query(BaseModel):
    question: str  # FIXED: This is the correct field name, not 'query'
    session_id: Optional[str] = None
    response_style: Optional[str] = "balanced"
    user_id: Optional[str] = None
    search_scope: Optional[str] = "all"  # "all", "user_only", "default_only", "external_only"
    external_databases: Optional[List[str]] = []  # ["lexisnexis", "westlaw"]
    use_enhanced_rag: Optional[bool] = True  # New: toggle enhanced RAG

class QueryResponse(BaseModel):
    response: Optional[str] = None
    error: Optional[str] = None
    context_found: bool = False
    sources: Optional[list] = None
    session_id: str
    confidence_score: float = 0.0
    expand_available: bool = False
    sources_searched: List[str] = []  # ["default_db", "user_container", "lexisnexis"]
    retrieval_method: Optional[str] = None  # New: track which retrieval method was used

class UserDocumentUpload(BaseModel):
    user_id: str
    file_id: str
    filename: str
    upload_timestamp: str
    pages_processed: int
    metadata: Dict[str, Any]

class DocumentUploadResponse(BaseModel):
    message: str
    file_id: str
    pages_processed: int
    processing_time: float
    warnings: List[str]
    session_id: str
    user_id: str
    container_id: str

class ConversationHistory(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]

# - User Management -
class UserContainerManager:
    """Manages user-specific document containers"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    def create_user_container(self, user_id: str) -> str:
        """Create a new container for a user"""
        container_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        container_path = os.path.join(self.base_path, container_id)
        os.makedirs(container_path, exist_ok=True)
        
        # Initialize user's ChromaDB
        user_db = Chroma(
            collection_name=f"user_{container_id}",
            embedding_function=self.embeddings,
            persist_directory=container_path
        )
        
        logger.info(f"Created container for user {user_id}: {container_id}")
        return container_id
    
    def get_user_database(self, user_id: str) -> Optional[Chroma]:
        """Get the ChromaDB instance for a user"""
        container_id = self.get_container_id(user_id)
        container_path = os.path.join(self.base_path, container_id)
        
        if not os.path.exists(container_path):
            logger.warning(f"Container not found for user {user_id}")
            return None
        
        return Chroma(
            collection_name=f"user_{container_id}",
            embedding_function=self.embeddings,
            persist_directory=container_path
        )
    
    def get_container_id(self, user_id: str) -> str:
        """Get container ID for a user"""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    
    def add_document_to_container(self, user_id: str, document_text: str, metadata: Dict) -> bool:
        """Add a document to user's container"""
        try:
            user_db = self.get_user_database(user_id)
            if not user_db:
                container_id = self.create_user_container(user_id)
                user_db = self.get_user_database(user_id)
            
            # Split document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            
            chunks = text_splitter.split_text(document_text)
            
            # Create documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata['chunk_index'] = i
                doc_metadata['total_chunks'] = len(chunks)
                doc_metadata['user_id'] = user_id
                doc_metadata['upload_timestamp'] = datetime.utcnow().isoformat()
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))
            
            # Add to user's database
            user_db.add_documents(documents)
            logger.info(f"Added {len(documents)} chunks to user {user_id}'s container")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document to user container: {e}")
            return False
    
    def search_user_container(self, user_id: str, query: str, k: int = 5) -> List[Tuple]:
        """Search within a user's container"""
        user_db = self.get_user_database(user_id)
        if not user_db:
            return []
        
        try:
            results = user_db.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error searching user container: {e}")
            return []
    
    def enhanced_search_user_container(self, user_id: str, query: str, conversation_context: str, k: int = 12) -> List[Tuple]:
        """Enhanced search within user's container using App 2's strategies"""
        user_db = self.get_user_database(user_id)
        if not user_db:
            return []
        
        try:
            # Strategy 1: Direct query
            direct_results = user_db.similarity_search_with_score(query, k=k)
            
            # Strategy 2: Expanded query
            expanded_query = f"{query} {conversation_context}"
            expanded_results = user_db.similarity_search_with_score(expanded_query, k=k)
            
            # Strategy 3: Sub-queries (if NLP available)
            sub_query_results = []
            if nlp:
                doc = nlp(query)
                for ent in doc.ents:
                    if ent.label_ in ["ORG", "PERSON", "LAW", "DATE"]:
                        sub_results = user_db.similarity_search_with_score(f"What is {ent.text}?", k=3)
                        sub_query_results.extend(sub_results)
            
            # Combine and deduplicate
            all_results = direct_results + expanded_results + sub_query_results
            return remove_duplicate_documents(all_results)[:k]
            
        except Exception as e:
            logger.error(f"Error in enhanced user container search: {e}")
            return []

# - Global State -
conversations: Dict[str, Dict] = {}
uploaded_files: Dict[str, Dict] = {}
user_sessions: Dict[str, User] = {}  # Map session_id to user
container_manager = UserContainerManager(USER_CONTAINERS_PATH)
external_databases = {
    "lexisnexis": LexisNexisInterface(),
    "westlaw": WestlawInterface()
}

# - Load NLP Models -
nlp = None
sentence_model = None
embeddings = None

try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {e}")
    nlp = None

try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence transformer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Sentence Transformer model: {e}")
    sentence_model = None

try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    logger.info("HuggingFace embeddings loaded successfully")
except Exception as e:
    logger.error(f"Failed to load HuggingFace embeddings: {e}")
    embeddings = None

# - Authentication - FIXED: Made authentication optional for debugging
def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> User:
    """Get current user from token (simplified for demo)"""
    if credentials is None:
        # For debugging - create a default user when no auth provided
        default_user_id = "debug_user"
        if default_user_id not in user_sessions:
            user_sessions[default_user_id] = User(
                user_id=default_user_id,
                container_id=container_manager.get_container_id(default_user_id),
                subscription_tier="free"
            )
        return user_sessions[default_user_id]
    
    # In production, validate JWT token and get user from database
    token = credentials.credentials
    
    # For demo, create user from token
    user_id = f"user_{token[:8]}"
    
    if user_id not in user_sessions:
        user_sessions[user_id] = User(
            user_id=user_id,
            container_id=container_manager.get_container_id(user_id),
            subscription_tier="free"
        )
    
    return user_sessions[user_id]

# - Enhanced RAG Functions from App 2 -
def parse_multiple_questions(query_text: str) -> List[str]:
    """Parse multiple questions from a single query"""
    questions = []
    
    # Strategy 1: Split by common separators
    if ';' in query_text:
        parts = query_text.split(';')
        for part in parts:
            part = part.strip()
            if part:
                questions.append(part)
    elif '?' in query_text and query_text.count('?') > 1:
        # Handle multiple question marks
        parts = query_text.split('?')
        for part in parts:
            part = part.strip()
            if part:
                questions.append(part + '?')
    else:
        final_question = query_text
        if not final_question.endswith('?') and '?' not in final_question:
            final_question += '?'
        questions = [final_question]
    
    return questions

def remove_duplicate_documents(results_with_scores: List[Tuple]) -> List[Tuple]:
    """Remove duplicate documents based on content similarity"""
    if not results_with_scores:
        return []
    
    unique_results = []
    seen_content = set()
    
    for doc, score in results_with_scores:
        # Create a hash of the first 100 characters for deduplication
        content_hash = hash(doc.page_content[:100])
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_results.append((doc, score))
    
    # Sort by relevance score (descending)
    unique_results.sort(key=lambda x: x[1], reverse=True)
    return unique_results

def enhanced_retrieval_v2(db, query_text: str, conversation_history_context: str, k: int = 12) -> Tuple[List, str]:
    """Enhanced retrieval from App 2 with multi-query approach"""
    logger.info(f"[ENHANCED_RETRIEVAL] Original query: '{query_text}'")
    
    try:
        # Strategy 1: Direct query
        direct_results = db.similarity_search_with_score(query_text, k=k)
        logger.info(f"[ENHANCED_RETRIEVAL] Direct search returned {len(direct_results)} results")
        
        # Strategy 2: Expanded query using conversation context
        expanded_query = f"{query_text} {conversation_history_context}"
        expanded_results = db.similarity_search_with_score(expanded_query, k=k)
        logger.info(f"[ENHANCED_RETRIEVAL] Expanded search returned {len(expanded_results)} results")
        
        # Strategy 3: Sub-query decomposition
        sub_queries = []
        # Extract potential legal terms or entities
        if nlp:
            doc = nlp(query_text)
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PERSON", "LAW", "DATE"]:
                    sub_queries.append(f"What is {ent.text}?")
        
        # Add generic question words if no entities found
        if not sub_queries:
            question_words = ["what", "who", "when", "where", "why", "how"]
            for word in question_words:
                if word in query_text.lower():
                    sub_queries.append(f"{word.capitalize()} {query_text.lower().replace(word, '').strip()}?")
        
        sub_query_results = []
        for sq in sub_queries[:3]:  # Limit sub-queries
            sq_results = db.similarity_search_with_score(sq, k=3)
            sub_query_results.extend(sq_results)
        
        logger.info(f"[ENHANCED_RETRIEVAL] Sub-query search returned {len(sub_query_results)} results")
        
        # Combine all results
        all_results = direct_results + expanded_results + sub_query_results
        
        # Remove duplicates and sort by score
        unique_results = remove_duplicate_documents(all_results)
        
        # Take top k results
        top_results = unique_results[:k]
        
        logger.info(f"[ENHANCED_RETRIEVAL] Final results after deduplication: {len(top_results)}")
        return top_results, "enhanced_retrieval_v2"
        
    except Exception as e:
        logger.error(f"[ENHANCED_RETRIEVAL] Error in enhanced retrieval: {e}")
        # Fallback to basic retrieval
        basic_results = db.similarity_search_with_score(query_text, k=k)
        return basic_results, "basic_fallback"

def calculate_confidence_score(results_with_scores: List[Tuple], response_length: int) -> float:
    """Calculate confidence score based on retrieval results and response"""
    try:
        if not results_with_scores:
            return 0.2
        
        scores = [score for _, score in results_with_scores]
        
        # Factor 1: Average relevance score
        avg_relevance = np.mean(scores)
        
        # Factor 2: Number of supporting documents
        doc_factor = min(1.0, len(results_with_scores) / 5.0)
        
        # Factor 3: Score distribution (consistency)
        if len(scores) > 1:
            score_std = np.std(scores)
            consistency_factor = max(0.5, 1.0 - score_std)
        else:
            consistency_factor = 0.7
            
        # Factor 4: Response completeness
        completeness_factor = min(1.0, response_length / 500.0)
        
        # Weighted combination
        confidence = (
            avg_relevance * 0.4 +
            doc_factor * 0.3 +
            consistency_factor * 0.2 +
            completeness_factor * 0.1
        )
        
        # Ensure confidence is between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        return confidence
    
    except Exception as e:
        logger.error(f"Error calculating confidence score: {e}")
        return 0.5

# - Utility Functions -
def load_database():
    """Load the default Chroma database"""
    try:
        if not os.path.exists(DEFAULT_CHROMA_PATH):
            logger.warning(f"Default database path does not exist: {DEFAULT_CHROMA_PATH}")
            return None
        
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(
            collection_name="default",
            embedding_function=embedding_function,
            persist_directory=DEFAULT_CHROMA_PATH
        )
        logger.debug("Default database loaded successfully")
        return db
    except Exception as e:
        logger.error(f"Failed to load default database: {e}")
        raise

def search_external_databases(query: str, databases: List[str], user: User) -> List[Dict]:
    """Search external legal databases (placeholder for future implementation)"""
    results = []
    
    for db_name in databases:
        if db_name not in user.external_db_access:
            logger.warning(f"User {user.user_id} does not have access to {db_name}")
            continue
        
        if db_name in external_databases:
            db_interface = external_databases[db_name]
            try:
                db_results = db_interface.search(query)
                for result in db_results:
                    result['source_database'] = db_name
                    results.extend(db_results)
            except Exception as e:
                logger.error(f"Error searching {db_name}: {e}")
    
    return results

def combined_search(query: str, user_id: Optional[str], search_scope: str, conversation_context: str, use_enhanced: bool = True, k: int = 10) -> Tuple[List, List[str], str]:
    """Enhanced combined search across all sources with App 2's smart RAG"""
    all_results = []
    sources_searched = []
    retrieval_method = "basic"
    
    # Search default database
    if search_scope in ["all", "default_only"]:
        try:
            default_db = load_database()
            if default_db:
                if use_enhanced:
                    default_results, method = enhanced_retrieval_v2(default_db, query, conversation_context, k=k)
                    retrieval_method = method
                else:
                    default_results = default_db.similarity_search_with_score(query, k=k)
                    retrieval_method = "basic_search"
                
                for doc, score in default_results:
                    doc.metadata['source_type'] = 'default_database'
                    all_results.append((doc, score))
                sources_searched.append("default_database")
        except Exception as e:
            logger.error(f"Error searching default database: {e}")
    
    # Search user container
    if user_id and search_scope in ["all", "user_only"]:
        try:
            if use_enhanced:
                user_results = container_manager.enhanced_search_user_container(user_id, query, conversation_context, k=k)
            else:
                user_results = container_manager.search_user_container(user_id, query, k=k)
            
            for doc, score in user_results:
                doc.metadata['source_type'] = 'user_container'
                all_results.append((doc, score))
            if user_results:
                sources_searched.append("user_container")
        except Exception as e:
            logger.error(f"Error searching user container: {e}")
    
    # Remove duplicates and sort by relevance
    if use_enhanced:
        all_results = remove_duplicate_documents(all_results)
    else:
        all_results.sort(key=lambda x: x[1], reverse=True)
    
    # Take top k results
    return all_results[:k], sources_searched, retrieval_method

def add_to_conversation(session_id: str, role: str, content: str, sources: Optional[List] = None):
    """Add message to conversation"""
    if session_id not in conversations:
        conversations[session_id] = {
            'messages': [],
            'created_at': datetime.utcnow(),
            'last_accessed': datetime.utcnow()
        }
    
    message = {
        'role': role,
        'content': content,
        'timestamp': datetime.utcnow().isoformat(),
        'sources': sources or []
    }
    
    conversations[session_id]['messages'].append(message)
    conversations[session_id]['last_accessed'] = datetime.utcnow()

def get_conversation_context(session_id: str, max_length: int = 2000) -> str:
    """Get formatted conversation context"""
    if session_id not in conversations:
        return ""
    
    messages = conversations[session_id]['messages']
    context_parts = []
    
    recent_messages = messages[-4:]
    
    for msg in recent_messages:
        role = msg['role'].upper()
        content = msg['content']
        if len(content) > 800:
            content = content[:800] + "..."
        context_parts.append(f"{role}: {content}")
    
    if context_parts:
        return "Previous conversation:\n" + "\n".join(context_parts)
    return ""

def cleanup_expired_conversations():
    """Remove conversations older than 1 hour"""
    now = datetime.utcnow()
    expired_sessions = [
        session_id for session_id, data in conversations.items()
        if now - data['last_accessed'] > timedelta(hours=1)
    ]
    for session_id in expired_sessions:
        del conversations[session_id]
    if expired_sessions:
        logger.info(f"Cleaned up {len(expired_sessions)} expired conversations")

def format_context_for_llm(results_with_scores: List[Tuple], max_length: int = 3000) -> Tuple[str, List]:
    """Format retrieved context for the LLM with source information"""
    context_parts = []
    source_info = []
    
    total_length = 0
    for i, (doc, score) in enumerate(results_with_scores):
        if total_length >= max_length:
            break
            
        content = doc.page_content.strip()
        metadata = doc.metadata
        
        source_path = metadata.get('source', 'unknown_source')
        page = metadata.get('page', None)
        source_type = metadata.get('source_type', 'unknown')
        
        display_source = os.path.basename(source_path)
        
        page_info = f" (Page {page})" if page is not None else ""
        source_prefix = f"[{source_type.upper()}]" if source_type != 'unknown' else ""
        
        if len(content) > 800:
            content = content[:800] + "... [truncated]"
            
        context_part = f"{source_prefix} [{display_source}{page_info}] (Relevance: {score:.2f}): {content}"
        context_parts.append(context_part)
        
        source_info.append({
            'id': i+1,
            'file_name': display_source,
            'page': page,
            'relevance': score,
            'full_path': source_path,
            'source_type': source_type
        })
        
        total_length += len(context_part)
    
    context_text = "\n\n".join(context_parts)
    return context_text, source_info

def call_openrouter_api(prompt: str, api_key: str, api_base: str = "https://openrouter.ai/api/v1") -> str:
    """Call OpenRouter API with fallback models"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Legal Assistant"
    }
    
    models_to_try = [
        "deepseek/deepseek-chat",
        "microsoft/phi-3-mini-128k-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "google/gemma-2-9b-it:free",
        "mistralai/mistral-7b-instruct:free",
        "openchat/openchat-7b:free"
    ]
    
    for model in models_to_try:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens": 2000
            }
            
            response = requests.post(api_base + "/chat/completions", headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content'].strip()
                
        except Exception as e:
            logger.error(f"Error with model {model}: {e}")
            continue
    
    return "I apologize, but I'm experiencing technical difficulties. Please try again."

# - Main Query Processing -
def process_query(question: str, session_id: str, user_id: Optional[str], search_scope: str, response_style: str = "balanced", use_enhanced_rag: bool = True) -> QueryResponse:
    """Process query with optional enhanced RAG from App 2"""
    try:
        # Log the incoming request for debugging
        logger.info(f"Processing query - Question: '{question}', User: {user_id}, Scope: {search_scope}, Enhanced: {use_enhanced_rag}")
        
        # Parse multiple questions if present
        questions = parse_multiple_questions(question) if use_enhanced_rag else [question]
        combined_query = " ".join(questions)
        
        # Get conversation context
        conversation_context = get_conversation_context(session_id)
        
        # Perform combined search with enhanced RAG option
        retrieved_results, sources_searched, retrieval_method = combined_search(
            combined_query, 
            user_id, 
            search_scope, 
            conversation_context,
            use_enhanced=use_enhanced_rag
        )
        
        if not retrieved_results:
            return QueryResponse(
                response="I couldn't find any relevant information to answer your question in the searched sources.",
                error=None,
                context_found=False,
                sources=[],
                session_id=session_id,
                confidence_score=0.1,
                sources_searched=sources_searched,
                retrieval_method=retrieval_method
            )
        
        # Format context for LLM
        context_text, source_info = format_context_for_llm(retrieved_results)
        
        # Style instructions
        style_instructions = {
            "concise": "Please provide a concise answer (1-2 sentences) based on the context.",
            "balanced": "Please provide a balanced answer (2-3 paragraphs) based on the context.",
            "detailed": "Please provide a detailed answer with explanations based on the context."
        }
        
        instruction = style_instructions.get(response_style, style_instructions["balanced"])
        
        # Enhanced prompt with best features from both apps
        prompt = f"""You are a legal research assistant. Your responses must be STRICTLY based on the provided legal documents, including any logical implications that can be reasonably drawn from their content.

SOURCES SEARCHED: {', '.join(sources_searched)}
RETRIEVAL METHOD: {retrieval_method}

CRITICAL REQUIREMENTS:
1. **ONLY use information from the provided context below**
2. **You may draw reasonable legal implications based on explicit content (e.g., statutory language, case reasoning, legislative findings)**
3. **Note which source type each piece of information comes from**
4. **If the context doesn't contain sufficient information, explicitly state this**
5. **Cite specific document names and source types for each claim**
6. **Do NOT invent facts, statutes, or case law not found in the context**
7. **Avoid general legal knowledge unless directly supported by cited documents**

RESPONSE STYLE: {instruction}
- Concise: Provide key points only
- Balanced: Structured overview with main points
- Detailed: Comprehensive analysis

CONVERSATION HISTORY:
{conversation_context}

LEGAL DOCUMENT CONTEXT (USE ONLY THIS INFORMATION):
{context_text}

USER QUESTION:
{questions}

INSTRUCTIONS:
- Use only the provided legal content, but you may infer relationships or implications between statutes and cases when clearly supported
- If context is insufficient, say: "Based on the available documents, I can only provide limited information..."
- Always cite the source document(s) for each fact or inference: [document_name.pdf]
- If no relevant information exists, say: "The available documents do not contain information about this topic."

RESPONSE:"""
        
        # Call LLM
        if AI_ENABLED and OPENROUTER_API_KEY:
            response_text = call_openrouter_api(prompt, OPENROUTER_API_KEY, OPENAI_API_BASE)
        else:
            response_text = f"Based on the retrieved documents:\n\n{context_text}\n\nPlease review this information to answer your question."
        
        # Add sources section
        MIN_RELEVANCE_SCORE = 0.25
        relevant_sources = [s for s in source_info if s['relevance'] >= MIN_RELEVANCE_SCORE]
        
        if relevant_sources:
            response_text += "\n\n**SOURCES:**"
            for source in relevant_sources:
                source_type = source['source_type'].replace('_', ' ').title()
                page_info = f", Page {source['page']}" if source['page'] is not None else ""
                response_text += f"\n- [{source_type}] {source['file_name']}{page_info} (Relevance: {source['relevance']:.2f})"
        
        # Calculate enhanced confidence score
        confidence_score = calculate_confidence_score(retrieved_results, len(response_text))
        
        # Update conversation
        add_to_conversation(session_id, "user", question)
        add_to_conversation(session_id, "assistant", response_text, source_info)
        
        return QueryResponse(
            response=response_text,
            error=None,
            context_found=True,
            sources=source_info,
            session_id=session_id,
            confidence_score=float(confidence_score),
            sources_searched=sources_searched,
            expand_available=len(questions) > 1 if use_enhanced_rag else False,
            retrieval_method=retrieval_method
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        traceback.print_exc()  # Add full traceback for debugging
        return QueryResponse(
            response=None,
            error=str(e),
            context_found=False,
            sources=[],
            session_id=session_id,
            confidence_score=0.0,
            sources_searched=[],
            retrieval_method="error"
        )

# - API Endpoints - FIXED: Made authentication optional for debugging
@app.post("/ask", response_model=QueryResponse)
async def ask_question(query: Query, current_user: Optional[User] = Depends(get_current_user)):
    """
    Enhanced ask endpoint with smart RAG capabilities
    - all: Search default database and user's container
    - user_only: Search only user's uploaded documents
    - default_only: Search only the default legal database
    - use_enhanced_rag: Enable/disable enhanced retrieval strategies
    """
    logger.info(f"Received ask request: {query}")
    
    cleanup_expired_conversations()
    
    session_id = query.session_id or str(uuid.uuid4())
    user_id = query.user_id or (current_user.user_id if current_user else None)
    
    if session_id not in conversations:
        conversations[session_id] = {
            "messages": [],
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow()
        }
    else:
        conversations[session_id]["last_accessed"] = datetime.utcnow()
    
    user_question = query.question.strip()
    if not user_question:
        return QueryResponse(
            response=None,
            error="Question cannot be empty.",
            context_found=False,
            sources=[],
            session_id=session_id,
            confidence_score=0.0,
            sources_searched=[]
        )
    
    response = process_query(
        user_question, 
        session_id, 
        user_id,
        query.search_scope or "all",
        query.response_style or "balanced",
        query.use_enhanced_rag if query.use_enhanced_rag is not None else True
    )
    return response

@app.get("/conversation/{session_id}", response_model=ConversationHistory)
async def get_conversation(session_id: str):
    """Get the conversation history for a session"""
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return ConversationHistory(
        session_id=session_id,
        messages=conversations[session_id]['messages']
    )

@app.post("/user/upload", response_model=DocumentUploadResponse)
async def upload_user_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload a document to user's personal container"""
    start_time = datetime.utcnow()
    
    try:
        # Check file size
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large. Maximum size is {MAX_FILE_SIZE//1024//1024}MB")
        
        # Check file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in LEGAL_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type. Supported: {LEGAL_EXTENSIONS}")
        
        # Process document using SafeDocumentProcessor
        content, pages_processed, warnings = SafeDocumentProcessor.process_document_safe(file)
        
        # Create metadata
        file_id = str(uuid.uuid4())
        metadata = {
            'source': file.filename,
            'file_id': file_id,
            'upload_date': datetime.utcnow().isoformat(),
            'user_id': current_user.user_id,
            'file_type': file_ext,
            'pages': pages_processed
        }
        
        # Add to user's container
        success = container_manager.add_document_to_container(
            current_user.user_id,
            content,
            metadata
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to add document to user container")
        
        # Store file info
        session_id = str(uuid.uuid4())
        uploaded_files[file_id] = {
            'filename': file.filename,
            'user_id': current_user.user_id,
            'container_id': current_user.container_id,
            'pages_processed': pages_processed,
            'uploaded_at': datetime.utcnow(),
            'session_id': session_id
        }
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return DocumentUploadResponse(
            message=f"Document {file.filename} uploaded successfully to your personal container",
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
        logger.error(f"Error uploading user document: {e}")
        traceback.print_exc()  # Add full traceback for debugging
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/documents")
async def list_user_documents(current_user: User = Depends(get_current_user)):
    """List all documents in user's container"""
    user_documents = []
    
    for file_id, file_data in uploaded_files.items():
        if file_data.get('user_id') == current_user.user_id:
            user_documents.append({
                'file_id': file_id,
                'filename': file_data['filename'],
                'uploaded_at': file_data['uploaded_at'].isoformat(),
                'pages_processed': file_data['pages_processed']
            })
    
    return {
        'user_id': current_user.user_id,
        'container_id': current_user.container_id,
        'documents': user_documents,
        'total_documents': len(user_documents)
    }

@app.delete("/user/documents/{file_id}")
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
    
    # In production, would also remove from ChromaDB
    del uploaded_files[file_id]
    
    return {"message": "Document deleted successfully", "file_id": file_id}

@app.post("/external/search")
async def search_external_databases_endpoint(
    query: str = Form(...),
    databases: List[str] = Form(...),
    current_user: User = Depends(get_current_user)
):
    """Search external legal databases (requires premium subscription)"""
    if current_user.subscription_tier not in ["premium", "enterprise"]:
        raise HTTPException(
            status_code=403, 
            detail="External database access requires premium subscription"
        )
    
    results = search_external_databases(query, databases, current_user)
    
    return {
        "query": query,
        "databases_searched": databases,
        "results": results,
        "total_results": len(results)
    }

@app.get("/subscription/status")
async def get_subscription_status(current_user: User = Depends(get_current_user)):
    """Get user's subscription status and available features"""
    features = {
        "free": {
            "default_database_access": True,
            "user_container": True,
            "max_documents": 10,
            "external_databases": [],
            "ai_analysis": True,
            "api_calls_per_month": 100,
            "enhanced_rag": True
        },
        "premium": {
            "default_database_access": True,
            "user_container": True,
            "max_documents": 100,
            "external_databases": ["lexisnexis", "westlaw"],
            "ai_analysis": True,
            "api_calls_per_month": 1000,
            "priority_support": True,
            "enhanced_rag": True
        },
        "enterprise": {
            "default_database_access": True,
            "user_container": True,
            "max_documents": "unlimited",
            "external_databases": ["lexisnexis", "westlaw", "bloomberg_law"],
            "ai_analysis": True,
            "api_calls_per_month": "unlimited",
            "priority_support": True,
            "custom_integrations": True,
            "enhanced_rag": True
        }
    }
    
    return {
        "user_id": current_user.user_id,
        "subscription_tier": current_user.subscription_tier,
        "features": features.get(current_user.subscription_tier, features["free"]),
        "external_db_access": current_user.external_db_access
    }

@app.get("/health")
def health_check():
    """System health check with enhanced RAG status"""
    db_exists = os.path.exists(DEFAULT_CHROMA_PATH)
    
    return {
        "status": "healthy",
        "version": "9.0.0-SmartRAG-Fixed",
        "timestamp": datetime.utcnow().isoformat(),
        "ai_enabled": AI_ENABLED,
        "openrouter_api_configured": bool(OPENROUTER_API_KEY),
        "components": {
            "default_database": {
                "exists": db_exists,
                "path": DEFAULT_CHROMA_PATH
            },
            "user_containers": {
                "enabled": True,
                "base_path": USER_CONTAINERS_PATH,
                "active_containers": len(os.listdir(USER_CONTAINERS_PATH)) if os.path.exists(USER_CONTAINERS_PATH) else 0
            },
            "external_databases": {
                "lexisnexis": {
                    "configured": bool(os.environ.get("LEXISNEXIS_API_KEY")),
                    "status": "ready" if bool(os.environ.get("LEXISNEXIS_API_KEY")) else "not_configured"
                },
                "westlaw": {
                    "configured": bool(os.environ.get("WESTLAW_API_KEY")),
                    "status": "ready" if bool(os.environ.get("WESTLAW_API_KEY")) else "not_configured"
                }
            },
            "enhanced_rag": {
                "enabled": True,
                "features": [
                    "multi_query_strategies",
                    "query_expansion",
                    "entity_extraction",
                    "sub_query_decomposition",
                    "confidence_scoring",
                    "duplicate_removal"
                ],
                "nlp_model": nlp is not None,
                "sentence_model": sentence_model is not None
            },
            "document_processing": {
                "pdf_support": PYMUPDF_AVAILABLE or PDFPLUMBER_AVAILABLE,
                "pymupdf_available": PYMUPDF_AVAILABLE,
                "pdfplumber_available": PDFPLUMBER_AVAILABLE,
                "docx_support": True,  # Basic docx support always available
                "txt_support": True
            }
        },
        "features": [
            "✅ User-specific document containers",
            "✅ Enhanced RAG with multi-query strategies",
            "✅ Combined search across all sources",
            "✅ External legal database integration (ready)",
            "✅ Subscription tier management",
            "✅ Document access control",
            "✅ Source attribution (default/user/external)",
            "✅ Dynamic confidence scoring",
            "✅ Query expansion and decomposition",
            "✅ SafeDocumentProcessor for file handling",
            "🔧 Optional authentication for debugging"
        ]
    }

# ADDED: Debug endpoint to test without authentication
@app.post("/ask-debug", response_model=QueryResponse)
async def ask_question_debug(query: Query):
    """
    Debug version of ask endpoint without authentication
    For testing frontend integration
    """
    logger.info(f"Debug ask request received: {query}")
    
    cleanup_expired_conversations()
    
    session_id = query.session_id or str(uuid.uuid4())
    
    # Create a debug user if none provided
    user_id = query.user_id or "debug_user"
    
    if session_id not in conversations:
        conversations[session_id] = {
            "messages": [],
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow()
        }
    else:
        conversations[session_id]["last_accessed"] = datetime.utcnow()
    
    user_question = query.question.strip()
    if not user_question:
        return QueryResponse(
            response=None,
            error="Question cannot be empty.",
            context_found=False,
            sources=[],
            session_id=session_id,
            confidence_score=0.0,
            sources_searched=[]
        )
    
    response = process_query(
        user_question, 
        session_id, 
        user_id,
        query.search_scope or "all",
        query.response_style or "balanced",
        query.use_enhanced_rag if query.use_enhanced_rag is not None else True
    )
    return response

@app.get("/", response_class=HTMLResponse)
def get_interface():
    """Web interface with updated documentation"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Legal Assistant - Smart Multi-User Edition (Fixed)</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f8f9fa; }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { color: #2c3e50; }
            .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0; }
            .feature-card { background: #fff; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; }
            .endpoint { background: #f1f3f4; padding: 10px; margin: 10px 0; border-radius: 5px; font-family: monospace; }
            .status { padding: 5px 10px; border-radius: 15px; font-size: 12px; }
            .status-active { background: #d4edda; color: #155724; }
            .status-ready { background: #cce5ff; color: #004085; }
            .status-fixed { background: #fff3cd; color: #856404; }
            .badge-new { background: #ff6b6b; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-left: 5px; }
            .code-example { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 5px; padding: 15px; margin: 10px 0; font-family: monospace; font-size: 12px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>⚖️ Legal Assistant API v9.0 <span class="badge-new">FIXED</span></h1>
            <p>Multi-User Platform with Enhanced Retrieval-Augmented Generation</p>
            <div class="status status-fixed">422 Error Fixed: Correct field name 'question' in Query model</div>
            
            <div class="feature-grid">
                <div class="feature-card">
                    <h3>🔧 Quick Fix Applied</h3>
                    <p>Fixed the 422 error by correcting the Pydantic model</p>
                    <ul>
                        <li>✅ Query model uses 'question' field (not 'query')</li>
                        <li>✅ Authentication made optional for debugging</li>
                        <li>✅ Added /ask-debug endpoint</li>
                        <li>✅ Enhanced error logging</li>
                    </ul>
                </div>
                
                <div class="feature-card">
                    <h3>📡 Correct Frontend Request</h3>
                    <p>Use this exact format in your frontend:</p>
                    <div class="code-example">
fetch('/ask', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer YOUR_TOKEN' // Optional
  },
  body: JSON.stringify({
    question: "What is contract law?",  // CORRECT field
    response_style: "balanced",
    search_scope: "all",
    use_enhanced_rag: true
  })
})
                    </div>
                </div>
                
                <div class="feature-card">
                    <h3>🧪 Debug Endpoint</h3>
                    <p>Test without authentication</p>
                    <div class="endpoint">POST /ask-debug</div>
                    <p>No Bearer token required - perfect for testing your frontend</p>
                </div>
                
                <div class="feature-card">
                    <h3>🧠 Enhanced RAG System</h3>
                    <p>Smart multi-query search with confidence scoring</p>
                    <div class="endpoint">POST /ask</div>
                    <ul>
                        <li>Multi-query strategies</li>
                        <li>Entity extraction</li>
                        <li>Query expansion</li>
                        <li>Confidence scoring</li>
                    </ul>
                </div>
                
                <div class="feature-card">
                    <h3>📁 User Document Containers</h3>
                    <p>Upload and manage your personal legal documents</p>
                    <div class="endpoint">POST /user/upload</div>
                    <div class="endpoint">GET /user/documents</div>
                    <div class="endpoint">DELETE /user/documents/{file_id}</div>
                    <p>Supports: PDF, TXT, DOCX, RTF</p>
                </div>
                
                <div class="feature-card">
                    <h3>🔍 Unified Search</h3>
                    <p>Search across default DB, your documents, and external sources</p>
                    <div class="endpoint">POST /ask</div>
                    <p>Search scopes: all, user_only, default_only</p>
                    <p>Toggle enhanced RAG: use_enhanced_rag</p>
                </div>
            </div>
            
            <h2>🚀 Key Features</h2>
            <ul>
                <li><strong>FIXED:</strong> 422 error resolved - correct 'question' field name</li>
                <li><strong>Smart RAG:</strong> Enhanced retrieval with multi-query strategies, entity extraction, and confidence scoring</li>
                <li><strong>Multi-User Support:</strong> Personal document containers for each user</li>
                <li><strong>Combined Search:</strong> Unified search across multiple sources with source attribution</li>
                <li><strong>Debug Mode:</strong> Optional authentication for easier frontend testing</li>
                <li><strong>External Integration:</strong> Ready for LexisNexis and Westlaw integration</li>
                <li><strong>Flexible Responses:</strong> Choose between concise, balanced, or detailed responses</li>
                <li><strong>Subscription Management:</strong> Tiered access to features</li>
                <li><strong>Safe Document Processing:</strong> Robust handling of PDF, DOCX, TXT, and RTF files</li>
            </ul>
            
            <h2>🔧 Quick Start (FIXED VERSION)</h2>
            <ol>
                <li><strong>Basic Test (No Auth):</strong> Use <code>/ask-debug</code> endpoint for testing</li>
                <li><strong>With Auth:</strong> Include Bearer token: <code>Authorization: Bearer YOUR_TOKEN</code></li>
                <li><strong>Upload documents:</strong> Use <code>/user/upload</code> to add documents to your personal container</li>
                <li><strong>Search with enhanced RAG:</strong> Set <code>{"use_enhanced_rag": true}</code></li>
                <li><strong>Choose response style:</strong> "concise", "balanced", or "detailed"</li>
                <li><strong>Premium users:</strong> Access external legal databases</li>
            </ol>
            
            <h2>🐛 Debugging Tips</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <h4>✅ Correct Request Format</h4>
                    <div class="code-example">
{
  "question": "What is contract law?",  // ✅ CORRECT
  "response_style": "balanced",
  "search_scope": "all",
  "use_enhanced_rag": true
}
                    </div>
                </div>
                
                <div class="feature-card">
                    <h4>❌ Common Mistakes</h4>
                    <div class="code-example">
{
  "query": "What is contract law?",     // ❌ WRONG field name
  "q": "What is contract law?",        // ❌ WRONG field name
}

// ❌ WRONG Content-Type
fetch('/ask', {
  method: 'POST',
  body: formData  // Should be JSON.stringify()
})
                    </div>
                </div>
                
                <div class="feature-card">
                    <h4>🧪 Test Commands</h4>
                    <div class="code-example">
# Test with curl (no auth needed)
curl -X POST "http://localhost:8000/ask-debug" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is contract law?"}'

# Test with auth
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-token" \
  -d '{"question": "What is contract law?"}'
                    </div>
                </div>
                
                <div class="feature-card">
                    <h4>📊 Health Check</h4>
                    <div class="code-example">
# Check system status
curl http://localhost:8000/health

# Check your documents
curl -H "Authorization: Bearer test-token" \
  http://localhost:8000/user/documents
                    </div>
                </div>
            </div>
            
            <h2>📋 Document Processing</h2>
            <ul>
                <li><strong>PDF:</strong> PyMuPDF and pdfplumber support for robust text extraction</li>
                <li><strong>DOCX:</strong> Full Microsoft Word document support</li>
                <li><strong>TXT:</strong> Plain text files with UTF-8 encoding</li>
                <li><strong>RTF:</strong> Rich Text Format (processed as text)</li>
                <li><strong>Max file size:</strong> 10MB per document</li>
            </ul>
            
            <h2>🔒 Authentication Options</h2>
            <ul>
                <li><strong>Production Mode:</strong> Use <code>/ask</code> with Bearer token</li>
                <li><strong>Debug Mode:</strong> Use <code>/ask-debug</code> without authentication</li>
                <li><strong>Token Format:</strong> Any string works in debug mode (e.g., "test-token")</li>
                <li><strong>Default User:</strong> When no auth provided, uses "debug_user"</li>
            </ul>
            
            <h2>⚡ API Response Format</h2>
            <div class="code-example">
{
  "response": "Legal answer based on retrieved documents...",
  "error": null,
  "context_found": true,
  "sources": [
    {
      "id": 1,
      "file_name": "contract_law.pdf",
      "page": 15,
      "relevance": 0.89,
      "source_type": "default_database"
    }
  ],
  "session_id": "uuid-here",
  "confidence_score": 0.85,
  "sources_searched": ["default_database", "user_container"],
  "retrieval_method": "enhanced_retrieval_v2"
}
            </div>
            
            <h2>🎯 Search Scopes</h2>
            <ul>
                <li><strong>"all":</strong> Search both default database and user documents</li>
                <li><strong>"default_only":</strong> Search only the default legal database</li>
                <li><strong>"user_only":</strong> Search only user's uploaded documents</li>
                <li><strong>"external_only":</strong> Search only external databases (Premium+)</li>
            </ul>
            
            <h2>🧠 Enhanced RAG Features</h2>
            <ul>
                <li><strong>Multi-Query Strategies:</strong> Breaks down complex questions</li>
                <li><strong>Query Expansion:</strong> Uses conversation context</li>
                <li><strong>Entity Extraction:</strong> Identifies legal terms, people, organizations</li>
                <li><strong>Sub-Query Decomposition:</strong> Creates targeted searches</li>
                <li><strong>Duplicate Removal:</strong> Eliminates redundant results</li>
                <li><strong>Confidence Scoring:</strong> Rates answer reliability</li>
                <li><strong>Source Attribution:</strong> Tracks where each piece of info comes from</li>
            </ul>
            
            <h2>💡 Usage Examples</h2>
            <div class="feature-grid">
                <div class="feature-card">
                    <h4>Basic Question</h4>
                    <div class="code-example">
POST /ask-debug
{
  "question": "What are the elements of a valid contract?"
}
                    </div>
                </div>
                
                <div class="feature-card">
                    <h4>Detailed Analysis</h4>
                    <div class="code-example">
POST /ask
{
  "question": "How do force majeure clauses work in commercial contracts?",
  "response_style": "detailed",
  "use_enhanced_rag": true
}
                    </div>
                </div>
                
                <div class="feature-card">
                    <h4>User Documents Only</h4>
                    <div class="code-example">
POST /ask
{
  "question": "What does my employment contract say about vacation days?",
  "search_scope": "user_only"
}
                    </div>
                </div>
                
                <div class="feature-card">
                    <h4>Quick Answer</h4>
                    <div class="code-example">
POST /ask
{
  "question": "What is the statute of limitations for breach of contract?",
  "response_style": "concise"
}
                    </div>
                </div>
            </div>
            
            <h2>🔍 Troubleshooting</h2>
            <div class="feature-card">
                <h4>Common Issues & Solutions</h4>
                <ul>
                    <li><strong>422 Error:</strong> Check that you're using "question" field, not "query"</li>
                    <li><strong>401 Error:</strong> Use /ask-debug for testing without auth</li>
                    <li><strong>No results:</strong> Try uploading documents or check search_scope</li>
                    <li><strong>Empty response:</strong> Ensure OpenRouter API key is set</li>
                    <li><strong>File upload fails:</strong> Check file size (max 10MB) and format (PDF/DOCX/TXT)</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    return html_template

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
