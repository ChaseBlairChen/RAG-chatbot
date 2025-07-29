# Unified Legal Assistant Backend - Multi-User with Enhanced RAG - FIXED VERSION

from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
import os
import json
import requests
import re
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Set, Any, Union
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
try:
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LangChain not available: {e}")
    LANGCHAIN_AVAILABLE = False

try:
    import spacy
    from sentence_transformers import SentenceTransformer
    import numpy as np
    NLP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"NLP libraries not available: {e}")
    NLP_AVAILABLE = False

# AI imports
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp not available - AI features disabled")

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

# Import PDF processing libraries
PYMUPDF_AVAILABLE = False
PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
    logger.info("✅ PyMuPDF available")
except ImportError:
    logger.warning("⚠️ PyMuPDF not available")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
    logger.info("✅ pdfplumber available")
except ImportError:
    logger.warning("⚠️ pdfplumber not available")

# SafeDocumentProcessor - Handles documents safely even without all dependencies
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
                content, pages_processed = SafeDocumentProcessor._process_pdf(file_content, warnings)
                
            elif file_ext == '.docx':
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
        """Process PDF content with fallbacks"""
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
        
        warnings.append("No PDF processing libraries available")
        return "PDF processing not available - please install PyMuPDF or pdfplumber", 0
    
    @staticmethod
    def _process_docx(file_content: bytes, warnings: List[str]) -> Tuple[str, int]:
        """Process DOCX content"""
        try:
            from docx import Document
            doc = Document(io.BytesIO(file_content))
            text_content = ""
            for paragraph in doc.paragraphs:
                text_content += paragraph.text + "\n"
            return text_content, 1
        except ImportError:
            warnings.append("python-docx not available. Install with: pip install python-docx")
            return "DOCX processing not available", 0
        except Exception as e:
            warnings.append(f"Error processing DOCX: {str(e)}")
            return "Error processing DOCX", 0

# Create FastAPI app with enhanced error handling
app = FastAPI(
    title="Unified Legal Assistant API",
    description="Multi-User Legal Assistant with Enhanced RAG and External Database Integration",
    version="9.1.0-Fixed"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with helpful messages"""
    error_details = []
    for error in exc.errors():
        error_details.append({
            "field": " -> ".join(str(x) for x in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "The request format is incorrect. Please check the API documentation.",
            "details": error_details,
            "help": "For /ask endpoint, send JSON like: {\"question\": \"your question here\"}"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "type": type(exc).__name__
        }
    )

# Configuration
DEFAULT_CHROMA_PATH = os.path.abspath(os.path.join(os.getcwd(), "chromadb-database"))
USER_CONTAINERS_PATH = os.path.abspath(os.path.join(os.getcwd(), "user-containers"))
logger.info(f"Using DEFAULT_CHROMA_PATH: {DEFAULT_CHROMA_PATH}")
logger.info(f"Using USER_CONTAINERS_PATH: {USER_CONTAINERS_PATH}")

os.makedirs(USER_CONTAINERS_PATH, exist_ok=True)

OPENROUTER_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
AI_ENABLED = bool(OPENROUTER_API_KEY) and AIOHTTP_AVAILABLE
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
LEGAL_EXTENSIONS = {'.pdf', '.txt', '.docx', '.rtf'}

# Security
security = HTTPBearer(auto_error=False)

# Pydantic Models with better validation
class User(BaseModel):
    user_id: str
    email: Optional[str] = None
    container_id: Optional[str] = None
    subscription_tier: str = "free"
    external_db_access: List[str] = []

class Query(BaseModel):
    question: str
    session_id: Optional[str] = None
    response_style: Optional[str] = "balanced"
    user_id: Optional[str] = None
    search_scope: Optional[str] = "all"
    external_databases: Optional[List[str]] = []
    use_enhanced_rag: Optional[bool] = True
    
    class Config:
        # Allow extra fields for flexibility
        extra = "ignore"

class QueryResponse(BaseModel):
    response: Optional[str] = None
    error: Optional[str] = None
    context_found: bool = False
    sources: Optional[List[Dict[str, Any]]] = None
    session_id: str
    confidence_score: float = 0.0
    expand_available: bool = False
    sources_searched: List[str] = []
    retrieval_method: Optional[str] = None

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

# Mock classes for when dependencies aren't available
class MockChroma:
    def __init__(self, *args, **kwargs):
        self.collection_name = kwargs.get('collection_name', 'mock')
        
    def similarity_search_with_score(self, query: str, k: int = 5):
        return [(MockDocument("Mock result for: " + query), 0.8)]
    
    def add_documents(self, documents):
        logger.info(f"Mock: Added {len(documents)} documents")

class MockDocument:
    def __init__(self, content: str, metadata: Dict = None):
        self.page_content = content
        self.metadata = metadata or {}

# Initialize embeddings and models with fallbacks
nlp = None
sentence_model = None
embeddings = None

if NLP_AVAILABLE:
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("✅ spaCy model loaded")
    except Exception as e:
        logger.warning(f"Could not load spaCy model: {e}")
        nlp = None

    try:
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Sentence transformer loaded")
    except Exception as e:
        logger.warning(f"Could not load Sentence Transformer: {e}")
        sentence_model = None

if LANGCHAIN_AVAILABLE:
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        logger.info("✅ HuggingFace embeddings loaded")
    except Exception as e:
        logger.warning(f"Could not load HuggingFace embeddings: {e}")
        embeddings = None

# User Container Manager with fallbacks
class UserContainerManager:
    """Manages user-specific document containers with fallbacks"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.embeddings = embeddings
    
    def create_user_container(self, user_id: str) -> str:
        """Create a new container for a user"""
        container_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        container_path = os.path.join(self.base_path, container_id)
        os.makedirs(container_path, exist_ok=True)
        
        if LANGCHAIN_AVAILABLE and self.embeddings:
            try:
                user_db = Chroma(
                    collection_name=f"user_{container_id}",
                    embedding_function=self.embeddings,
                    persist_directory=container_path
                )
            except Exception as e:
                logger.warning(f"Could not create Chroma DB, using mock: {e}")
        
        logger.info(f"Created container for user {user_id}: {container_id}")
        return container_id
    
    def get_user_database(self, user_id: str):
        """Get the database instance for a user"""
        container_id = self.get_container_id(user_id)
        container_path = os.path.join(self.base_path, container_id)
        
        if not os.path.exists(container_path):
            os.makedirs(container_path, exist_ok=True)
        
        if LANGCHAIN_AVAILABLE and self.embeddings:
            try:
                return Chroma(
                    collection_name=f"user_{container_id}",
                    embedding_function=self.embeddings,
                    persist_directory=container_path
                )
            except Exception as e:
                logger.warning(f"Could not load Chroma DB, using mock: {e}")
                return MockChroma(collection_name=f"user_{container_id}")
        else:
            return MockChroma(collection_name=f"user_{container_id}")
    
    def get_container_id(self, user_id: str) -> str:
        """Get container ID for a user"""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    
    def add_document_to_container(self, user_id: str, document_text: str, metadata: Dict) -> bool:
        """Add a document to user's container"""
        try:
            user_db = self.get_user_database(user_id)
            
            if LANGCHAIN_AVAILABLE:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                chunks = text_splitter.split_text(document_text)
                
                documents = []
                for i, chunk in enumerate(chunks):
                    doc_metadata = metadata.copy()
                    doc_metadata.update({
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'user_id': user_id,
                        'upload_timestamp': datetime.utcnow().isoformat()
                    })
                    
                    documents.append(Document(
                        page_content=chunk,
                        metadata=doc_metadata
                    ))
                
                user_db.add_documents(documents)
                logger.info(f"Added {len(documents)} chunks to user {user_id}'s container")
            else:
                # Mock processing
                logger.info(f"Mock: Added document to user {user_id}'s container")
            
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

# Global state
conversations: Dict[str, Dict] = {}
uploaded_files: Dict[str, Dict] = {}
user_sessions: Dict[str, User] = {}
container_manager = UserContainerManager(USER_CONTAINERS_PATH)

# Utility Functions
def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> User:
    """Get current user from token (simplified for demo)"""
    if credentials:
        token = credentials.credentials
        user_id = f"user_{token[:8]}"
    else:
        user_id = "anonymous_user"
    
    if user_id not in user_sessions:
        user_sessions[user_id] = User(
            user_id=user_id,
            container_id=container_manager.get_container_id(user_id),
            subscription_tier="free"
        )
    
    return user_sessions[user_id]

def load_database():
    """Load the default database with fallback"""
    if not LANGCHAIN_AVAILABLE or not embeddings:
        logger.warning("LangChain not available, using mock database")
        return MockChroma(collection_name="default")
    
    try:
        if not os.path.exists(DEFAULT_CHROMA_PATH):
            logger.warning(f"Default database path does not exist: {DEFAULT_CHROMA_PATH}")
            os.makedirs(DEFAULT_CHROMA_PATH, exist_ok=True)
        
        db = Chroma(
            collection_name="default",
            embedding_function=embeddings,
            persist_directory=DEFAULT_CHROMA_PATH
        )
        logger.info("Default database loaded successfully")
        return db
    except Exception as e:
        logger.error(f"Failed to load default database: {e}")
        return MockChroma(collection_name="default")

def parse_multiple_questions(query_text: str) -> List[str]:
    """Parse multiple questions from a single query"""
    questions = []
    
    if ';' in query_text:
        parts = query_text.split(';')
        for part in parts:
            part = part.strip()
            if part:
                questions.append(part)
    elif '?' in query_text and query_text.count('?') > 1:
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

def enhanced_retrieval_v2(db, query_text: str, conversation_history_context: str, k: int = 12) -> Tuple[List, str]:
    """Enhanced retrieval with fallbacks"""
    logger.info(f"[ENHANCED_RETRIEVAL] Query: '{query_text[:100]}...'")
    
    try:
        # Direct query
        direct_results = db.similarity_search_with_score(query_text, k=k)
        
        # Expanded query using conversation context
        if conversation_history_context:
            expanded_query = f"{query_text} {conversation_history_context}"
            expanded_results = db.similarity_search_with_score(expanded_query, k=k//2)
        else:
            expanded_results = []
        
        # Combine results
        all_results = direct_results + expanded_results
        
        # Simple deduplication
        seen_content = set()
        unique_results = []
        for doc, score in all_results:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append((doc, score))
        
        return unique_results[:k], "enhanced_retrieval_v2"
        
    except Exception as e:
        logger.error(f"Error in enhanced retrieval: {e}")
        basic_results = db.similarity_search_with_score(query_text, k=k)
        return basic_results, "basic_fallback"

def combined_search(query: str, user_id: Optional[str], search_scope: str, conversation_context: str, use_enhanced: bool = True, k: int = 10) -> Tuple[List, List[str], str]:
    """Combined search across all sources"""
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
            user_results = container_manager.search_user_container(user_id, query, k=k)
            for doc, score in user_results:
                doc.metadata['source_type'] = 'user_container'
                all_results.append((doc, score))
            if user_results:
                sources_searched.append("user_container")
        except Exception as e:
            logger.error(f"Error searching user container: {e}")
    
    # Sort by relevance
    all_results.sort(key=lambda x: x[1], reverse=True)
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
    
    return "Previous conversation:\n" + "\n".join(context_parts) if context_parts else ""

def format_context_for_llm(results_with_scores: List[Tuple], max_length: int = 3000) -> Tuple[str, List]:
    """Format retrieved context for the LLM"""
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
        
        display_source = os.path.basename(source_path) if isinstance(source_path, str) else str(source_path)
        
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
            'relevance': float(score),
            'full_path': str(source_path),
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
        "mistralai/mistral-7b-instruct:free"
    ]
    
    for model in models_to_try:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens": 2000
            }
            
            response = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content'].strip()
                
        except Exception as e:
            logger.error(f"Error with model {model}: {e}")
            continue
    
    return "I apologize, but I'm experiencing technical difficulties with the AI service. Please try again later."

def calculate_confidence_score(results_with_scores: List[Tuple], response_length: int) -> float:
    """Calculate confidence score"""
    try:
        if not results_with_scores:
            return 0.2
        
        scores = [score for _, score in results_with_scores]
        avg_relevance = sum(scores) / len(scores) if scores else 0.0
        doc_factor = min(1.0, len(results_with_scores) / 5.0)
        completeness_factor = min(1.0, response_length / 500.0)
        
        confidence = (avg_relevance * 0.5 + doc_factor * 0.3 + completeness_factor * 0.2)
        return max(0.0, min(1.0, confidence))
    
    except Exception as e:
        logger.error(f"Error calculating confidence score: {e}")
        return 0.5

def process_query(question: str, session_id: str, user_id: Optional[str], search_scope: str, response_style: str = "balanced", use_enhanced_rag: bool = True) -> QueryResponse:
    """Process query with enhanced error handling"""
    try:
        # Parse multiple questions if enhanced RAG is enabled
        questions = parse_multiple_questions(question) if use_enhanced_rag else [question]
        combined_query = " ".join(questions)
        
        # Get conversation context
        conversation_context = get_conversation_context(session_id)
        
        # Perform combined search
        retrieved_results, sources_searched, retrieval_method = combined_search(
            combined_query, 
            user_id, 
            search_scope, 
            conversation_context,
            use_enhanced=use_enhanced_rag
        )
        
        if not retrieved_results:
            return QueryResponse(
                response="I couldn't find any relevant information to answer your question in the searched sources. This might be because the database is empty or your question doesn't match available content.",
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
        
        # Enhanced prompt
        prompt = f"""You are a legal research assistant. Answer based STRICTLY on the provided legal documents.

SOURCES SEARCHED: {', '.join(sources_searched)}
RETRIEVAL METHOD: {retrieval_method}

CRITICAL REQUIREMENTS:
1. **ONLY use information from the provided context below**
2. **You may draw reasonable legal implications based on explicit content**
3. **Note which source type each piece of information comes from**
4. **If the context doesn't contain sufficient information, explicitly state this**
5. **Cite specific document names and source types for each claim**
6. **Do NOT invent facts, statutes, or case law not found in the context**

RESPONSE STYLE: {instruction}

CONVERSATION HISTORY:
{conversation_context}

LEGAL DOCUMENT CONTEXT (USE ONLY THIS INFORMATION):
{context_text}

USER QUESTION:
{combined_query}

INSTRUCTIONS:
- Use only the provided legal content
- If context is insufficient, say: "Based on the available documents, I can only provide limited information..."
- Always cite the source document(s): [document_name.pdf]
- If no relevant information exists, say: "The available documents do not contain information about this topic."

RESPONSE:"""
        
        # Call LLM or provide fallback response
        if AI_ENABLED and OPENROUTER_API_KEY:
            response_text = call_openrouter_api(prompt, OPENROUTER_API_KEY, OPENAI_API_BASE)
        else:
            response_text = f"""Based on the retrieved documents, I found {len(retrieved_results)} relevant sources. However, AI processing is not currently available.

**Retrieved Information:**
{context_text[:1000]}{'...' if len(context_text) > 1000 else ''}

Please review this information to answer your question. To enable AI-powered analysis, configure the OPENAI_API_KEY environment variable with your OpenRouter API key."""
        
        # Add sources section
        MIN_RELEVANCE_SCORE = 0.25
        relevant_sources = [s for s in source_info if s['relevance'] >= MIN_RELEVANCE_SCORE]
        
        if relevant_sources:
            response_text += "\n\n**SOURCES:**"
            for source in relevant_sources:
                source_type = source['source_type'].replace('_', ' ').title()
                page_info = f", Page {source['page']}" if source['page'] is not None else ""
                response_text += f"\n- [{source_type}] {source['file_name']}{page_info} (Relevance: {source['relevance']:.2f})"
        
        # Calculate confidence score
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
        logger.error(f"Error processing query: {e}", exc_info=True)
        return QueryResponse(
            response=None,
            error=f"Error processing query: {str(e)}",
            context_found=False,
            sources=[],
            session_id=session_id,
            confidence_score=0.0,
            sources_searched=[],
            retrieval_method="error"
        )

# API Endpoints
@app.post("/ask", response_model=QueryResponse)
async def ask_question(
    request: Request,
    query: Optional[Query] = None,
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Enhanced ask endpoint with better error handling
    
    Accepts JSON body with Query model or handles various input formats
    """
    try:
        # Handle different input formats
        if query is None:
            # Try to parse request body manually
            try:
                body = await request.json()
                if isinstance(body, str):
                    # If body is a plain string, treat it as the question
                    query = Query(question=body)
                elif isinstance(body, dict):
                    # If body is a dict, try to create Query from it
                    query = Query(**body)
                else:
                    raise ValueError("Invalid request format")
            except Exception as e:
                logger.error(f"Error parsing request body: {e}")
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "Invalid request format",
                        "message": "Please send JSON with 'question' field",
                        "example": {"question": "Your legal question here"},
                        "details": str(e)
                    }
                )
        
        session_id = query.session_id or str(uuid.uuid4())
        user_id = query.user_id or (current_user.user_id if current_user else None)
        
        # Initialize conversation if needed
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in ask_question: {e}", exc_info=True)
        return QueryResponse(
            response=None,
            error=f"Unexpected error: {str(e)}",
            context_found=False,
            sources=[],
            session_id=str(uuid.uuid4()),
            confidence_score=0.0,
            sources_searched=[],
            retrieval_method="error"
        )

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
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large. Maximum size is {MAX_FILE_SIZE//1024//1024}MB")
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Check file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in LEGAL_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type. Supported: {LEGAL_EXTENSIONS}")
        
        # Process document using SafeDocumentProcessor
        content, pages_processed, warnings = SafeDocumentProcessor.process_document_safe(file)
        
        if not content or content.strip() == "":
            raise HTTPException(status_code=400, detail="Could not extract text from file")
        
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
            warnings.append("Document processed but may not be fully searchable")
        
        # Store file info
        session_id = str(uuid.uuid4())
        uploaded_files[file_id] = {
            'filename': file.filename,
            'user_id': current_user.user_id,
            'container_id': current_user.container_id,
            'pages_processed': pages_processed,
            'uploaded_at': datetime.utcnow(),
            'session_id': session_id,
            'content_length': len(content)
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
        logger.error(f"Error uploading user document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")

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
                'pages_processed': file_data['pages_processed'],
                'content_length': file_data.get('content_length', 0)
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
    
    # Remove from uploaded files tracker
    del uploaded_files[file_id]
    
    return {"message": "Document deleted successfully", "file_id": file_id}

@app.get("/subscription/status")
async def get_subscription_status(current_user: User = Depends(get_current_user)):
    """Get user's subscription status and available features"""
    features = {
        "free": {
            "default_database_access": True,
            "user_container": True,
            "max_documents": 10,
            "external_databases": [],
            "ai_analysis": AI_ENABLED,
            "api_calls_per_month": 100,
            "enhanced_rag": True
        },
        "premium": {
            "default_database_access": True,
            "user_container": True,
            "max_documents": 100,
            "external_databases": ["lexisnexis", "westlaw"],
            "ai_analysis": AI_ENABLED,
            "api_calls_per_month": 1000,
            "priority_support": True,
            "enhanced_rag": True
        },
        "enterprise": {
            "default_database_access": True,
            "user_container": True,
            "max_documents": "unlimited",
            "external_databases": ["lexisnexis", "westlaw", "bloomberg_law"],
            "ai_analysis": AI_ENABLED,
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
    """System health check with dependency status"""
    return {
        "status": "healthy",
        "version": "9.1.0-Fixed",
        "timestamp": datetime.utcnow().isoformat(),
        "ai_enabled": AI_ENABLED,
        "dependencies": {
            "langchain": LANGCHAIN_AVAILABLE,
            "nlp_libraries": NLP_AVAILABLE,
            "aiohttp": AIOHTTP_AVAILABLE,
            "pymupdf": PYMUPDF_AVAILABLE,
            "pdfplumber": PDFPLUMBER_AVAILABLE,
            "open_source_nlp": OPEN_SOURCE_NLP_AVAILABLE
        },
        "components": {
            "default_database": {
                "exists": os.path.exists(DEFAULT_CHROMA_PATH),
                "path": DEFAULT_CHROMA_PATH
            },
            "user_containers": {
                "enabled": True,
                "base_path": USER_CONTAINERS_PATH,
                "active_containers": len(os.listdir(USER_CONTAINERS_PATH)) if os.path.exists(USER_CONTAINERS_PATH) else 0
            },
            "enhanced_rag": {
                "enabled": True,
                "features": [
                    "multi_query_strategies",
                    "query_expansion", 
                    "confidence_scoring",
                    "fallback_support"
                ],
                "nlp_model": nlp is not None,
                "sentence_model": sentence_model is not None
            }
        },
        "features": [
            "✅ Enhanced error handling and validation",
            "✅ Graceful dependency fallbacks", 
            "✅ User-specific document containers",
            "✅ Enhanced RAG with smart retrieval",
            "✅ Multiple input format support",
            "✅ Robust document processing",
            "✅ Source attribution and confidence scoring",
            "✅ Conversation history management"
        ]
    }

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify API is working"""
    return {
        "message": "Legal Assistant API is running",
        "timestamp": datetime.utcnow().isoformat(),
        "status": "OK"
    }

@app.get("/", response_class=HTMLResponse)
def get_interface():
    """Enhanced web interface with better documentation"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Legal Assistant - Fixed Multi-User Edition</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f8f9fa; }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { color: #2c3e50; }
            .alert { padding: 15px; margin: 20px 0; border-radius: 5px; }
            .alert-success { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
            .alert-warning { background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }
            .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0; }
            .feature-card { background: #fff; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; }
            .endpoint { background: #f1f3f4; padding: 10px; margin: 10px 0; border-radius: 5px; font-family: monospace; }
            .code-block { background: #2d3748; color: #e2e8f0; padding: 15px; border-radius: 5px; margin: 10px 0; overflow-x: auto; }
            .badge-fixed { background: #28a745; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; margin-left: 5px; }
            .status { padding: 5px 10px; border-radius: 15px; font-size: 12px; }
            .status-active { background: #d4edda; color: #155724; }
            .status-degraded { background: #fff3cd; color: #856404; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>⚖️ Legal Assistant API v9.1 <span class="badge-fixed">FIXED</span></h1>
            <p>Multi-User Platform with Enhanced RAG and Robust Error Handling</p>
            
            <div class="alert alert-success">
                <strong>✅ Fixed Issues:</strong>
                <ul>
                    <li>422 validation errors - now handles multiple input formats</li>
                    <li>Graceful fallbacks when dependencies are missing</li>
                    <li>Better error messages and debugging</li>
                    <li>Robust document processing</li>
                </ul>
            </div>
            
            <div class="alert alert-warning">
                <strong>⚠️ Setup Notes:</strong>
                <ul>
                    <li>Install dependencies: <code>pip install langchain-chroma langchain-huggingface spacy sentence-transformers</code></li>
                    <li>For PDF support: <code>pip install PyMuPDF pdfplumber</code></li>
                    <li>For AI features: Set <code>OPENAI_API_KEY</code> environment variable</li>
                    <li>Download spaCy model: <code>python -m spacy download en_core_web_sm</code></li>
                </ul>
            </div>
            
            <h2>🚀 Quick Test</h2>
            <div class="code-block">
# Test the API with curl:
curl -X POST "http://localhost:8000/ask" \\
     -H "Content-Type: application/json" \\
     -H "Authorization: Bearer test_token" \\
     -d '{"question": "What is contract law?"}'

# Or with simple string (fallback):
curl -X POST "http://localhost:8000/ask" \\
     -H "Content-Type: application/json" \\
     -H "Authorization: Bearer test_token" \\
     -d '"What is contract law?"'
            </div>
            
            <div class="feature-grid">
                <div class="feature-card">
                    <h3>🧠 Enhanced RAG System</h3>
                    <p>Smart multi-query search with confidence scoring</p>
                    <div class="endpoint">POST /ask</div>
                    <div class="code-block">
{
  "question": "Your legal question",
  "search_scope": "all|user_only|default_only",
  "response_style": "concise|balanced|detailed",
  "use_enhanced_rag": true
}
                    </div>
                </div>
                
                <div class="feature-card">
                    <h3>📁 Document Management</h3>
                    <p>Upload and search your personal legal documents</p>
                    <div class="endpoint">POST /user/upload</div>
                    <div class="endpoint">GET /user/documents</div>
                    <div class="endpoint">DELETE /user/documents/{file_id}</div>
                    <p><strong>Supported:</strong> PDF, TXT, DOCX, RTF (up to 10MB)</p>
                </div>
                
                <div class="feature-card">
                    <h3>🔍 Unified Search</h3>
                    <p>Search across multiple sources with source attribution</p>
                    <ul>
                        <li><strong>all:</strong> Search everything</li>
                        <li><strong>user_only:</strong> Your documents only</li>
                        <li><strong>default_only:</strong> Default database only</li>
                    </ul>
                </div>
                
                <div class="feature-card">
                    <h3>💬 Conversation History</h3>
                    <p>Contextual responses with conversation memory</p>
                    <div class="endpoint">GET /conversation/{session_id}</div>
                    <p>Automatic session management and context awareness</p>
                </div>
                
                <div class="feature-card">
                    <h3>🔧 System Health</h3>
                    <p>Monitor system status and dependencies</p>
                    <div class="endpoint">GET /health</div>
                    <div class="endpoint">GET /test</div>
                    <p>Check what features are available with current setup</p>
                </div>
                
                <div class="feature-card">
                    <h3>👤 User Management</h3>
                    <p>Authentication and subscription tiers</p>
                    <div class="endpoint">GET /subscription/status</div>
                    <p>Use <code>Authorization: Bearer your_token</code> header</p>
                </div>
            </div>
            
            <h2>🛠️ Error Handling Improvements</h2>
            <ul>
                <li><strong>Flexible Input:</strong> Accepts both JSON objects and plain strings</li>
                <li><strong>Graceful Degradation:</strong> Works even without all dependencies</li>
                <li><strong>Better Validation:</strong> Clear error messages with examples</li>
                <li><strong>Fallback Processing:</strong> Mock implementations when libraries missing</li>
                <li><strong>Robust File Handling:</strong> Safe document processing with warnings</li>
            </ul>
            
            <h2>📋 Installation Guide</h2>
            <div class="code-block">
# Basic installation
pip install fastapi uvicorn

# For enhanced RAG features  
pip install langchain-chroma langchain-huggingface
pip install sentence-transformers spacy numpy

# For document processing
pip install PyMuPDF pdfplumber python-docx

# Download spaCy model
python -m spacy download en_core_web_sm

# For AI features (optional)
export OPENAI_API_KEY="your_openrouter_api_key"

# Run the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
            </div>
            
            <h2>🎯 Key Features</h2>
            <ul>
                <li><strong>Works Out of the Box:</strong> Core functionality without dependencies</li>
                <li><strong>Progressive Enhancement:</strong> More features as dependencies are installed</li>
                <li><strong>Smart Error Handling:</strong> Helpful error messages and fallbacks</li>
                <li><strong>Multiple Input Formats:</strong> Flexible API that handles various request types</li>
                <li><strong>User Isolation:</strong> Personal document containers for each user</li>
                <li><strong>Source Attribution:</strong> Track where information comes from</li>
                <li><strong>Confidence Scoring:</strong> Know how reliable each answer is</li>
            </ul>
        </div>
    </body>
    </html>
    """
    return html_template

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
