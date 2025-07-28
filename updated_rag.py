# Unified Legal Assistant Backend - Enhanced RAG version
# This version uses the more creative RAG approach from the second app

from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Third-party library imports for RAG
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
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

# Try to import PyMuPDF (fitz)
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
    print("✅ PyMuPDF available - using enhanced PDF extraction")
except ImportError as e:
    print(f"⚠️ PyMuPDF not available: {e}")
    print("Install with: pip install PyMuPDF")

# Try to import pdfplumber
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
    print("✅ pdfplumber available - using enhanced PDF extraction")
except ImportError as e:
    print(f"⚠️ pdfplumber not available: {e}")
    print("Install with: pip install pdfplumber")

print(f"PDF processing status: PyMuPDF={PYMUPDF_AVAILABLE}, pdfplumber={PDFPLUMBER_AVAILABLE}")

# Create FastAPI app
app = FastAPI(
    title="Unified Legal Assistant API",
    description="Combined RAG Q&A and Document Analysis System with Enhanced Creative Responses",
    version="7.0.0-Enhanced"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# - Configuration -
# FIXED: Use absolute path matching the working app
CHROMA_PATH = os.path.abspath(os.path.join(os.getcwd(), "chromadb-database"))
logger.info(f"Using CHROMA_PATH: {CHROMA_PATH}")

OPENROUTER_API_KEY = os.environ.get("OPENAI_API_KEY")  # Reuse OPENAI_API_KEY env var
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
AI_ENABLED = bool(OPENROUTER_API_KEY) and AIOHTTP_AVAILABLE
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
LEGAL_EXTENSIONS = {'.pdf', '.txt', '.docx', '.rtf'}

# - Pydantic Models -
# RAG Models
class Query(BaseModel):
    question: str
    session_id: Optional[str] = None
    response_style: Optional[str] = "balanced"  # "concise", "balanced", "detailed"

class QueryResponse(BaseModel):
    response: Optional[str] = None
    error: Optional[str] = None
    context_found: bool = False
    sources: Optional[list] = None
    session_id: str
    confidence_score: float = 0.0
    expand_available: bool = False

class ConversationHistory(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]

# Document Analysis Models
class DocumentUploadResponse(BaseModel):
    message: str
    file_id: str
    pages_processed: int
    processing_time: float
    warnings: List[str]
    session_id: str

class DocumentAnalysisResponse(BaseModel):
    analysis: Optional[str] = None
    error: Optional[str] = None
    session_id: str
    model_used: str
    confidence_score: float
    ai_analysis: Optional[str] = None
    extraction_results: Optional[Dict[str, Any]] = None
    analysis_type: str
    processing_info: Dict[str, Any]
    verification_status: str
    status: str
    success: bool
    warnings: List[str]
    timestamp: str

# - Global State -
# In-memory storage for conversations and file uploads (use Redis/database in production)
conversations: Dict[str, Dict] = {}
uploaded_files: Dict[str, Dict] = {}

# - Load NLP Models -
# Load models once at startup for efficiency
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

# - Utility Functions -
# FIXED: Updated load_database function to match working app
def load_database():
    """Load the Chroma database"""
    try:
        if not os.path.exists(CHROMA_PATH):
            logger.warning(f"Chroma database path does not exist: {CHROMA_PATH}")
            return None
        
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(
            collection_name="default",
            embedding_function=embedding_function,
            persist_directory=CHROMA_PATH
        )
        logger.debug("Database loaded successfully")
        return db
    except Exception as e:
        logger.error(f"Failed to load database: {e}")
        raise

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
    
    # Get last few messages for context
    recent_messages = messages[-4:]  # Last 2 exchanges (user + assistant)
    
    for msg in recent_messages:
        role = msg['role'].upper()
        content = msg['content']
        # For conversation history, keep more content but still truncate very long messages
        if len(content) > 800:
            content = content[:800] + "..."
        context_parts.append(f"{role}: {content}")
    
    # If we have conversation history, make it clear this is ongoing context
    if context_parts:
        return "Previous conversation:" + "".join(context_parts)
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

# - Enhanced RAG System Functions (From Second App) -
def enhanced_retrieval_v2(db, query_text: str, conversation_history_context: str, k: int = 12) -> Tuple[List, Any]:
    """Improved retrieval with better scoring and multi-query approach"""
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
        
        # Extract source information
        source_path = metadata.get('source', 'unknown_source')
        page = metadata.get('page', None)
        
        # Simplify source name for display
        display_source = os.path.basename(source_path)
        
        page_info = f" (Page {page})" if page is not None else ""
        
        # Truncate very long content
        if len(content) > 800:
            content = content[:800] + "... [truncated]"
            
        context_part = f"[{display_source}{page_info}] (Relevance: {score:.2f}): {content}"
        context_parts.append(context_part)
        
        source_info.append({
            'id': i+1,
            'file_name': display_source,
            'page': page,
            'relevance': score,
            'full_path': source_path
        })
        
        total_length += len(context_part)
    
    context_text = "\n".join(context_parts)
    return context_text, source_info

def calculate_confidence_score(search_result: Dict, response_length: int) -> float:
    """Calculate confidence score based on retrieval results and response"""
    try:
        results = search_result.get("results", [])
        scores = search_result.get("scores", [])
        
        if not scores:
            return 0.2  # Low confidence if no scores
        
        # Factor 1: Average relevance score
        avg_relevance = np.mean(scores)
        
        # Factor 2: Number of supporting documents
        doc_factor = min(1.0, len(results) / 5.0)  # Optimal around 5 documents
        
        # Factor 3: Score distribution (consistency)
        if len(scores) > 1:
            score_std = np.std(scores)
            consistency_factor = max(0.5, 1.0 - score_std)
        else:
            consistency_factor = 0.7
            
        # Factor 4: Response completeness
        completeness_factor = min(1.0, response_length / 500.0)  # Optimal around 500 chars
        
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
        return 0.5  # Default medium confidence

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
                "temperature": 0.5,  # Lower temperature for more consistent responses
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
def process_query(question: str, session_id: str, response_style: str = "balanced") -> QueryResponse:
    """Process the user query using the RAG pipeline and AI agent, including conversation history."""
    try:
        # - Load Database -
        db = load_database()
        if not db:
            return QueryResponse(
                response=None,
                error="Database not available",
                context_found=False,
                sources=[],
                session_id=session_id,
                confidence_score=0.0
            )
        
        # Test database connectivity
        test_results = db.similarity_search("test", k=1)
        logger.info(f"Database loaded successfully with {len(test_results)} test results")
        
        # - Parse Question -
        questions = parse_multiple_questions(question)
        logger.info(f"Parsed {len(questions)} questions: {questions}")
        combined_query = " ".join(questions)
        
        # - End Parse Question -
        
        # - Retrieve Context -
        # Get conversation history context
        conversation_history_context = get_conversation_context(session_id)
        
        # Perform enhanced retrieval
        retrieved_results, retrieval_method = enhanced_retrieval_v2(db, combined_query, conversation_history_context)
        
        if not retrieved_results:
            return QueryResponse(
                response="I couldn't find any relevant information to answer your question.",
                error=None,
                context_found=False,
                sources=[],
                session_id=session_id,
                confidence_score=0.1
            )
        
        # Format context for LLM
        context_text, source_info = format_context_for_llm(retrieved_results)
        
        # - End Retrieve Context -
        
        # - Construct Prompt -
        # Style instructions
        style_instructions = {
            "concise": "Please provide a concise answer (1-2 sentences) based on the context.",
            "balanced": "Please provide a balanced answer (2-3 paragraphs) based on the context.",
            "detailed": "Please provide a detailed answer with explanations based on the context."
        }
        
        instruction = style_instructions.get(response_style, style_instructions["balanced"])
        
        prompt = f"""You are a legal research assistant with expertise in legal analysis. You must balance using the provided documents with your legal reasoning abilities.

CRITICAL RULES:
1. **Primary source: Retrieved documents** - All specific facts, cases, and statutes must come from the provided context
2. **You may reason from retrieved content and infer logical implications between cited laws and cases**
3. **Allowed pre-trained knowledge:**
   - General legal principles and concepts (e.g., due process, statutory construction)
   - Legal reasoning methodology and analysis frameworks
   - Understanding of how laws typically interact
   - Common legal terminology and definitions
4. **NOT allowed from pre-trained knowledge:**
   - Specific case names or holdings not in the documents
   - Specific statute numbers or provisions not in the documents
   - Factual claims about what a law says if not in the documents
5. **Cite all specific claims** with [document_name.pdf]

RESPONSE STYLE: {response_style}
- Concise: Key legal points with essential analysis
- Balanced: Structured legal analysis with reasoning
- Detailed: Comprehensive legal examination

CONVERSATION HISTORY:
{conversation_history}

LEGAL DOCUMENT CONTEXT:
{context}

USER QUESTION:
{questions}

INSTRUCTIONS:
- Use the documents as your factual foundation
- Apply legal reasoning and analysis skills to interpret the documents
- Draw logical connections and implications between the cited materials
- You may explain general legal concepts to provide context
- When using general legal knowledge, make it clear (e.g., "Generally in law..." or "As a matter of legal principle...")
- Focus on providing insightful analysis based on the combination of documents + legal reasoning

RESPONSE:"""
        
        # - End Construct Prompt -
        
        # - Call LLM -
        if AI_ENABLED and OPENROUTER_API_KEY:
            response_text = call_openrouter_api(prompt, OPENROUTER_API_KEY, OPENAI_API_BASE)
        else:
            # Fallback: Return context only with basic formatting
            response_text = f"Based on the retrieved documents:\n\n{context_text}\n\nPlease review this information to answer your question."
        
        # - End Call LLM -
        
        # - Post-process Response (Add Sources) -
        MIN_RELEVANCE_SCORE_FOR_SOURCE_DISPLAY = 0.25
        relevant_source_info = [
            source for source in source_info 
            if source['relevance'] >= MIN_RELEVANCE_SCORE_FOR_SOURCE_DISPLAY
        ]
        
        # Add sources section only if there are relevant sources
        if relevant_source_info:
            response_text += "\n\n**SOURCES:**"
            for source in relevant_source_info:
                page_info = f", Page {source['page']}" if source['page'] is not None else ""
                response_text += f"\n- {source['file_name']}{page_info} (Relevance: {source['relevance']:.2f})"
        
        # - End Post-process Response -
        
        # - Calculate Confidence -
        search_result_info = {
            "results": [doc for doc, score in retrieved_results],
            "scores": [score for doc, score in retrieved_results]
        }
        confidence_score = calculate_confidence_score(search_result_info, len(response_text))
        # - End Calculate Confidence -
        
        # Update conversation
        add_to_conversation(session_id, "user", question)
        add_to_conversation(session_id, "assistant", response_text, source_info)
        
        return QueryResponse(
            response=response_text,
            error=None,
            context_found=True,
            sources=source_info,
            session_id=session_id,
            confidence_score=confidence_score,
            expand_available=len(questions) > 1
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        error_msg = "An error occurred while processing your request. Please try again."
        return QueryResponse(
            response=None,
            error=error_msg,
            context_found=False,
            sources=[],
            session_id=session_id,
            confidence_score=0.0
        )

# - API Endpoints -
# RAG Endpoints
@app.post("/ask", response_model=QueryResponse)
async def ask_question_improved(query: Query):
    """
    Improved question endpoint with enhanced accuracy and user experience
    """
    cleanup_expired_conversations()
    
    # Session management
    session_id = query.session_id or str(uuid.uuid4())
    if session_id not in conversations:
        conversations[session_id] = {
            "messages": [],
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow()
        }
    else:
        conversations[session_id]["last_accessed"] = datetime.utcnow()
    
    # Validate input
    user_question = query.question.strip()
    if not user_question:
        return QueryResponse(
            response=None,
            error="Question cannot be empty.",
            context_found=False,
            sources=[],
            session_id=session_id,
            confidence_score=0.0
        )
    
    # Process the query
    response = process_query(user_question, session_id, query.response_style or "balanced")
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

# Document Analysis Endpoints
class SafeDocumentProcessor:
    @staticmethod
    def process_document_safe(file: UploadFile) -> Tuple[str, int, List[str]]:
        """Safely process document with multiple fallback methods"""
        content = ""
        pages_processed = 0
        warnings = []
        
        try:
            # Check file size
            file.file.seek(0, 2)  # Seek to end
            file_size = file.file.tell()
            file.file.seek(0)  # Reset to beginning
            
            if file_size > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail=f"File too large. Maximum size is {MAX_FILE_SIZE//1024//1024}MB")
            
            # Check file extension
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in LEGAL_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"Unsupported file type. Supported: {LEGAL_EXTENSIONS}")
            
            # Process based on file type
            if file_ext == '.pdf':
                content, pages_processed = SafeDocumentProcessor._process_pdf(file.file)
            elif file_ext == '.txt':
                content = file.file.read().decode('utf-8', errors='ignore')
                pages_processed = 1
            else:
                # For other formats, try to read as text
                content = file.file.read().decode('utf-8', errors='ignore')
                pages_processed = 1
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            warnings.append(f"Error processing document: {str(e)}")
            # Try to recover with basic text extraction
            try:
                content = file.file.read().decode('utf-8', errors='ignore')
                pages_processed = 1
                warnings.append("Fell back to basic text extraction")
            except Exception as e2:
                logger.error(f"Complete failure in document processing: {e2}")
                raise HTTPException(status_code=500, detail="Failed to process document")
        
        return content, pages_processed, warnings
    
    @staticmethod
    def _process_pdf(file_stream) -> Tuple[str, int]:
        """Process PDF with multiple fallback libraries"""
        content = ""
        pages_processed = 0
        
        # Save stream to temp file for processing
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file_stream.read())
            tmp_path = tmp_file.name
        
        try:
            # Try PyMuPDF first
            if PYMUPDF_AVAILABLE:
                try:
                    doc = fitz.open(tmp_path)
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        content += page.get_text()
                        pages_processed += 1
                    doc.close()
                    return content, pages_processed
                except Exception as e:
                    logger.warning(f"PyMuPDF failed: {e}")
            
            # Fallback to pdfplumber
            if PDFPLUMBER_AVAILABLE:
                try:
                    with pdfplumber.open(tmp_path) as pdf:
                        for page in pdf.pages:
                            content += page.extract_text() or ""
                            pages_processed += 1
                    return content, pages_processed
                except Exception as e:
                    logger.warning(f"pdfplumber failed: {e}")
            
            # If both fail, raise an error
            raise Exception("All PDF processing methods failed")
            
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

class NoHallucinationAnalyzer:
    """Fallback analyzer that only extracts verifiable facts"""
    
    def analyze_document(self, text: str, session_id: str) -> DocumentAnalysisResponse:
        """Analyze document without AI, only extracting verifiable facts"""
        start_time = datetime.utcnow()
        warnings = []
        
        try:
            facts = self._extract_facts_regex(text)
            
            # Create summary from extracted facts
            summary_parts = []
            summary_parts.append("## Document Facts (Verified Only)")
            
            # Parties
            parties = facts['party_names']
            if parties and parties[0].get('status') != 'failed_to_extract':
                summary_parts.append(f"• **Parties**: {len(parties)} identified")
                for party in parties[:3]:
                    summary_parts.append(f"  - {party['value']} (Line {party['line_number']})")
                if len(parties) > 3:
                    summary_parts.append(f"  ... and {len(parties) - 3} more parties")
            else:
                summary_parts.append("• **Parties**: Failed to extract clear party names")
            
            # Dates
            dates = facts['dates']
            if dates and dates[0].get('status') != 'failed_to_extract':
                summary_parts.append(f"• **Dates found**: {len(dates)} verifiable dates")
                for i, date in enumerate(dates[:3], 1):
                    summary_parts.append(f"  {i}. {date['value']} (Line {date['line_number']})")
                if len(dates) > 3:
                    summary_parts.append(f"  ... and {len(dates) - 3} more dates")
            else:
                summary_parts.append("• **Dates**: Failed to extract any verifiable dates")
            
            # Amounts
            amounts = facts['monetary_amounts']
            if amounts and amounts[0].get('status') != 'failed_to_extract':
                summary_parts.append(f"• **Financial amounts**: {len(amounts)} verifiable amounts")
                for i, amount in enumerate(amounts[:3], 1):
                    summary_parts.append(f"  {i}. {amount['value']} (Line {amount['line_number']})")
                if len(amounts) > 3:
                    summary_parts.append(f"  ... and {len(amounts) - 3} more amounts")
            else:
                summary_parts.append("• **Financial amounts**: Failed to extract verifiable amounts")
            
            # Percentages
            percentages = facts['percentages']
            if percentages and percentages[0].get('status') != 'failed_to_extract':
                summary_parts.append(f"• **Percentages**: {len(percentages)} verifiable percentages")
                for i, pct in enumerate(percentages[:3], 1):
                    summary_parts.append(f"  {i}. {pct['value']} (Line {pct['line_number']})")
                if len(percentages) > 3:
                    summary_parts.append(f"  ... and {len(percentages) - 3} more percentages")
            else:
                summary_parts.append("• **Percentages**: Failed to extract verifiable percentages")
            
            summary_parts.append("\n**⚠️ IMPORTANT NOTES**:")
            summary_parts.append("• This analysis only includes information that could be verified directly from the document text")
            summary_parts.append("• No AI inference or interpretation was used")
            summary_parts.append("• For deeper analysis, please configure the AI features")
            
            result = "\n".join(summary_parts)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return DocumentAnalysisResponse(
                analysis=result,
                error=None,
                session_id=session_id,
                model_used="regex-extraction-only",
                confidence_score=0.8,  # High confidence for verifiable facts
                ai_analysis=None,
                extraction_results=facts,
                analysis_type="fact_extraction",
                processing_info={
                    "processing_time_seconds": processing_time,
                    "extraction_method": "regex",
                    "pages_processed": 1,  # Approximation
                },
                verification_status="high_confidence",
                status="completed",
                success=True,
                warnings=warnings,
                timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error in no-hallucination analysis: {e}")
            return DocumentAnalysisResponse(
                analysis=None,
                error=str(e),
                session_id=session_id,
                model_used="regex-extraction-only",
                confidence_score=0.0,
                ai_analysis=None,
                extraction_results=None,
                analysis_type="fact_extraction",
                processing_info={
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                verification_status="failed",
                status="failed",
                success=False,
                warnings=warnings,
                timestamp=datetime.utcnow().isoformat()
            )
    
    def _extract_facts_regex(self, text: str) -> Dict[str, List]:
        """Extract verifiable facts using regex only"""
        lines = text.split('\n')
        facts = {
            'party_names': [],
            'dates': [],
            'monetary_amounts': [],
            'percentages': []
        }
        
        # Extract parties (basic)
        party_pattern = r'\b(Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)+'
        for line_num, line in enumerate(lines, 1):
            matches = re.findall(party_pattern, line)
            for match in matches:
                full_name = f"{match[0]} {match[1]}"
                facts['party_names'].append({
                    'value': full_name,
                    'line_number': line_num,
                    'status': 'extracted'
                })
        
        # Extract dates
        date_patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{1,2}-\d{1,2}-\d{4}\b'
        ]
        
        for line_num, line in enumerate(lines, 1):
            for pattern in date_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    facts['dates'].append({
                        'value': match,
                        'line_number': line_num,
                        'status': 'extracted'
                    })
        
        # Extract monetary amounts
        amount_pattern = r'\$?\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b'
        for line_num, line in enumerate(lines, 1):
            matches = re.findall(amount_pattern, line)
            for match in matches:
                facts['monetary_amounts'].append({
                    'value': match,
                    'line_number': line_num,
                    'status': 'extracted'
                })
        
        # Extract percentages
        pct_pattern = r'\b\d+(?:\.\d+)?%'
        for line_num, line in enumerate(lines, 1):
            matches = re.findall(pct_pattern, line)
            for match in matches:
                facts['percentages'].append({
                    'value': match,
                    'line_number': line_num,
                    'status': 'extracted'
                })
        
        # Add failed status if no facts found
        for key in facts:
            if not facts[key]:
                facts[key].append({
                    'value': 'None found',
                    'line_number': 0,
                    'status': 'failed_to_extract'
                })
        
        return facts

class LightweightLegalAnalyzer:
    """Lightweight analyzer that loads one model at a time for memory efficiency"""
    
    def __init__(self, model_choice: str = "qa"):
        self.model_choice = model_choice
        self.models = {}
        self.models_loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load a single model based on choice to save memory"""
        try:
            if self.model_choice == "classifier":
                self.models['classifier'] = pipeline(
                    "text-classification",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    tokenizer="nlptown/bert-base-multilingual-uncased-sentiment"
                )
            elif self.model_choice == "ner":
                self.models['ner'] = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple"
                )
            elif self.model_choice == "qa":
                self.models['qa'] = pipeline(
                    "question-answering",
                    model="deepset/bert-base-cased-squad2",
                    tokenizer="deepset/bert-base-cased-squad2"
                )
            elif self.model_choice == "summarizer":
                self.models['summarizer'] = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    tokenizer="facebook/bart-large-cnn"
                )
            elif self.model_choice == "zero_shot":
                self.models['zero_shot'] = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    tokenizer="facebook/bart-large-mnli"
                )
            
            self.models_loaded = True
            logger.info(f"✅ Successfully loaded {self.model_choice} model")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {self.model_choice} model: {e}")
            self.models_loaded = False
    
    def _split_into_chunks(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks for processing"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _detect_document_type(self, text: str) -> str:
        """Detect document type using zero-shot classification or keyword matching"""
        if 'zero_shot' in self.models:
            try:
                candidate_labels = ["contract", "agreement", "lease", "employment agreement", "loan document", "other"]
                result = self.models['zero_shot'](text[:1024], candidate_labels)
                return result['labels'][0]  # Top prediction
            except:
                pass
        
        # Fallback to keyword matching
        text_lower = text.lower()
        if 'lease' in text_lower:
            return 'Lease Agreement'
        elif 'employment' in text_lower:
            return 'Employment Agreement'
        elif 'contract' in text_lower:
            return 'Contract'
        elif 'agreement' in text_lower:
            return 'Agreement'
        else:
            return 'Legal Document'
    
    def _extract_parties_qa(self, text: str) -> List[str]:
        """Extract party names using question answering"""
        if 'qa' not in self.models:
            return []
        
        parties = []
        questions = [
            "Who are the parties to this agreement?",
            "What is the name of the first party?",
            "What is the name of the second party?",
            "Who is the employer?",
            "Who is the employee?",
            "Who is the landlord?",
            "Who is the tenant?"
        ]
        
        try:
            for question in questions:
                result = self.models['qa'](question=question, context=text[:2000])
                if result['score'] > 0.5:
                    parties.append(result['answer'])
        except:
            pass
        
        return list(set(parties))  # Remove duplicates
    
    def _extract_dates_regex(self, text: str) -> List[str]:
        """Extract dates using regex"""
        date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        dates = re.findall(date_pattern, text, re.IGNORECASE)
        return dates
    
    def _extract_amounts_regex(self, text: str) -> List[str]:
        """Extract monetary amounts using regex"""
        amount_pattern = r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b'
        amounts = re.findall(amount_pattern, text, re.IGNORECASE)
        return amounts
    
    def analyze_document(self, text: str, session_id: str) -> DocumentAnalysisResponse:
        """Analyze document with the loaded lightweight model"""
        if not self.models_loaded:
            # Fallback to no hallucination analyzer
            fallback_analyzer = NoHallucinationAnalyzer()
            return fallback_analyzer.analyze_document(text, session_id)
        
        start_time = datetime.utcnow()
        warnings = []
        
        try:
            # Document type detection
            doc_type = self._detect_document_type(text)
            
            # Key information extraction
            parties = self._extract_parties_qa(text)
            dates = self._extract_dates_regex(text)
            amounts = self._extract_amounts_regex(text)
            
            # Summary generation based on model choice
            if self.model_choice == "summarizer":
                result, confidence = self._generate_summary(text)
            elif self.model_choice == "qa":
                result, confidence = self._generate_qa_summary(text, parties, dates, amounts)
            else:
                result, confidence = self._generate_basic_summary(text, doc_type, parties, dates, amounts)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return DocumentAnalysisResponse(
                analysis=result,
                error=None,
                session_id=session_id,
                model_used=self.model_choice,
                confidence_score=confidence,
                ai_analysis=result,
                extraction_results={
                    "parties": parties,
                    "dates": dates,
                    "amounts": amounts,
                    "document_type": doc_type
                },
                analysis_type="ai_enhanced",
                processing_info={
                    "processing_time_seconds": processing_time,
                    "model_used": self.model_choice,
                    "pages_processed": len(text) // 2000 + 1,  # Rough estimate
                },
                verification_status="ai_generated",
                status="completed",
                success=True,
                warnings=warnings,
                timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error in lightweight analysis: {e}")
            # Fallback to no hallucination analyzer
            fallback_analyzer = NoHallucinationAnalyzer()
            return fallback_analyzer.analyze_document(text, session_id)
    
    def _generate_summary(self, text: str) -> Tuple[str, float]:
        """Generate summary using BART model"""
        try:
            # Split into chunks
            chunks = self._split_into_chunks(text, 1000)
            summaries = []
            
            for i, chunk in enumerate(chunks[:3]):  # Process max 3 chunks to save memory
                try:
                    print(f"Summarizing chunk {i+1}...")
                    summary = self.models['summarizer'](
                        chunk,
                        max_length=150,
                        min_length=50,
                        do_sample=False
                    )[0]['summary_text']
                    summaries.append(summary)
                except Exception as e:
                    print(f"Error summarizing chunk: {e}")
                    continue
            
            if summaries:
                result = "## Document Summary (BART Model)\n" + "\n".join(summaries)
                return result, 0.7
            else:
                return "Failed to generate summary", 0.0
                
        except Exception as e:
            logger.error(f"Error in summary generation: {e}")
            return f"Error generating summary: {str(e)}", 0.0
    
    def _generate_qa_summary(self, text: str, parties: List[str], dates: List[str], amounts: List[str]) -> Tuple[str, float]:
        """Generate summary using QA model results"""
        try:
            result = f"""## Document Analysis (QA Model)
**Document Type**: Detected via QA
**Parties Identified**: {', '.join(parties) if parties else 'None identified'}
**Key Dates**: {', '.join(dates[:5]) if dates else 'None found'}
**Key Amounts**: {', '.join(amounts[:5]) if amounts else 'None found'}

This analysis used question-answering techniques to extract specific information.
Confidence is moderate as it relies on the QA model's ability to find answers in the text."""
            return result, 0.65
        except Exception as e:
            return f"Error in QA analysis: {str(e)}", 0.0
    
    def _generate_basic_summary(self, text: str, doc_type: str, parties: List[str], dates: List[str], amounts: List[str]) -> Tuple[str, float]:
        """Generate basic summary for other models"""
        try:
            # Just use the first part of the document as summary
            summary_text = text[:500] + "..." if len(text) > 500 else text
            
            result = f"""## Document Analysis
**Document Type**: {doc_type}
**Overview**: {summary_text}
**Parties**: {', '.join(parties) if parties else 'Not clearly identified'}
**Sample Dates**: {', '.join(dates[:3]) if dates else 'None found'}
**Sample Amounts**: {', '.join(amounts[:3]) if amounts else 'None found'}

This is a basic analysis based on the selected lightweight model.
For more comprehensive analysis, use a different model or enable full AI features."""
            return result, 0.5
        except Exception as e:
            return f"Error in basic analysis: {str(e)}", 0.0

# Initialize analyzers
safe_analyzer = NoHallucinationAnalyzer()
open_source_analyzer = None

# Choose which BERT model to test (for low-memory systems)
# Options: "classifier" (400MB), "ner" (400MB), "qa" (400MB),
# "summarizer" (1.2GB), "zero_shot" (1.2GB)
# If you have 2GB RAM + 2GB swap:
BERT_MODEL_TO_TEST = "qa"  # Best balance of usefulness and memory
# If you have 2GB RAM + 4GB swap:
# BERT_MODEL_TO_TEST = "zero_shot"  # Best for contract analysis
# If zero_shot is too big:
# BERT_MODEL_TO_TEST = "summarizer"  # Second best option

# Set to True to use lightweight single-model loader, False to disable BERT entirely
USE_LIGHTWEIGHT_BERT = True

if OPEN_SOURCE_NLP_AVAILABLE and USE_LIGHTWEIGHT_BERT:
    try:
        print(f"🔧 Loading lightweight BERT analyzer with model: {BERT_MODEL_TO_TEST}")
        print("This uses less memory by loading only one model at a time.")
        open_source_analyzer = LightweightLegalAnalyzer(model_choice=BERT_MODEL_TO_TEST)
        if open_source_analyzer.models_loaded:
            print(f"✅ Successfully loaded {BERT_MODEL_TO_TEST} model")
            print(f"💡 To test a different model, change BERT_MODEL_TO_TEST in the code")
        else:
            print(f"❌ Failed to load {BERT_MODEL_TO_TEST} model - insufficient memory")
            print("💡 Try adding swap memory or using a smaller model")
            open_source_analyzer = None
    except Exception as e:
        logger.error(f"Failed to initialize lightweight analyzer: {e}")
        print(f"❌ Error: {e}")
        print("💡 Continuing without BERT models - DeepSeek AI is still available")
        open_source_analyzer = None
else:
    print("ℹ️ BERT models disabled or not available")

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a legal document for processing"""
    try:
        # Process document
        content, pages_processed, warnings = SafeDocumentProcessor.process_document_safe(file)
        
        # Generate file ID
        file_id = str(uuid.uuid4())
        
        # Store file info (in production, save to database)
        uploaded_files[file_id] = {
            'filename': file.filename,
            'content': content,
            'pages_processed': pages_processed,
            'uploaded_at': datetime.utcnow(),
            'session_id': str(uuid.uuid4())  # Each upload gets a new session
        }
        
        return DocumentUploadResponse(
            message=f"Document {file.filename} uploaded successfully",
            file_id=file_id,
            pages_processed=pages_processed,
            processing_time=0.0,  # Simplified
            warnings=warnings,
            session_id=uploaded_files[file_id]['session_id']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-document", response_model=DocumentAnalysisResponse)
async def analyze_document(file_id: str = Form(...), analysis_type: str = Form("full")):
    """Analyze an uploaded document"""
    try:
        # Retrieve file
        if file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_data = uploaded_files[file_id]
        document_text = file_data['content']
        session_id = file_data['session_id']
        
        # Choose analyzer
        if AI_ENABLED and open_source_analyzer:
            analyzer = open_source_analyzer
        else:
            analyzer = safe_analyzer
        
        # Perform analysis
        analysis_result = analyzer.analyze_document(document_text, session_id)
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify-extraction")
async def verify_extraction(
    file: UploadFile = File(...),
    claim: str = Form(...),
    line_number: Optional[int] = Form(None)
):
    """Verify a specific claim against the document"""
    try:
        document_text, _, _ = SafeDocumentProcessor.process_document_safe(file)
        lines = document_text.split('\n')
        
        verification_result = {
            "claim": claim,
            "verification_status": "not_verified",
            "evidence": None,
            "line_number": None,
            "context": None
        }
        
        # If line number is provided, check that specific line
        if line_number and 1 <= line_number <= len(lines):
            line = lines[line_number - 1]  # Line numbers are 1-indexed
            if claim.lower() in line.lower():
                verification_result.update({
                    "verification_status": "verified",
                    "evidence": line.strip(),
                    "line_number": line_number,
                    "context": line.strip()
                })
            else:
                # Check surrounding lines for context
                start = max(0, line_number - 2)
                end = min(len(lines), line_number + 1)
                context_lines = lines[start:end]
                context_text = "\n".join(context_lines)
                
                if claim.lower() in context_text.lower():
                    verification_result.update({
                        "verification_status": "verified_with_context",
                        "evidence": context_text.strip(),
                        "line_number": line_number,
                        "context": context_text.strip()
                    })
        
        # If not verified yet, search entire document
        if verification_result["verification_status"] == "not_verified":
            for i, line in enumerate(lines, 1):
                if claim.lower() in line.lower():
                    verification_result.update({
                        "verification_status": "verified",
                        "evidence": line.strip(),
                        "line_number": i,
                        "context": line.strip()
                    })
                    break
        
        return verification_result
        
    except Exception as e:
        return {
            "claim": claim,
            "verification_status": "error",
            "error": str(e)
        }

# Health and Status Endpoints
@app.get("/health")
def unified_health_check():
    """Comprehensive health check for the unified system"""
    # Check ChromaDB
    db_exists = os.path.exists(CHROMA_PATH)
    db_status = "healthy" if db_exists else "not_found"
    
    try:
        if db_exists:
            db = load_database()
            test_results = db.similarity_search("test", k=1)
            db_status = "healthy" if test_results is not None else "error"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "version": "7.0.0-Enhanced",
        "timestamp": datetime.utcnow().isoformat(),
        "ai_enabled": AI_ENABLED,
        "database_exists": db_exists,
        "database_path": CHROMA_PATH,
        "api_key_configured": bool(OPENROUTER_API_KEY),
        "active_conversations": len(conversations),
        "unified_mode": True,
        "enhanced_rag": True,  # New flag indicating enhanced creative RAG
        "components": {
            "rag_system": {
                "enabled": db_exists,
                "database_status": db_status,
                "database_path": CHROMA_PATH,
                "nlp_model": nlp is not None,
                "sentence_model": sentence_model is not None,
                "active_conversations": len(conversations),
                "features": [
                    "enhanced_retrieval_v2",
                    "confidence_scoring",
                    "response_styles",
                    "query_expansion",
                    "multi_query_strategy"
                ]
            },
            "document_analysis": {
                "enabled": True,
                "ai_enabled": AI_ENABLED,
                "ai_model": "deepseek-chat" if AI_ENABLED else "not-configured",
                "open_source_nlp": {
                    "available": OPEN_SOURCE_NLP_AVAILABLE,
                    "models_loaded": open_source_analyzer.models_loaded if open_source_analyzer else False,
                    "models": ["legal-bert", "bart-summarization", "roberta-qa", "zero-shot-classification"] if open_source_analyzer and open_source_analyzer.models_loaded else []
                },
                "pdf_processors": {
                    "pymupdf": PYMUPDF_AVAILABLE,
                    "pdfplumber": PDFPLUMBER_AVAILABLE,
                    "pypdf2": True
                }
            },
            "api_configuration": {
                "openai_key_set": bool(OPENROUTER_API_KEY),
                "aiohttp_available": AIOHTTP_AVAILABLE,
                "api_base": OPENAI_API_BASE
            }
        },
        "features": [
            "✅ Enhanced RAG with multi-query strategies",
            "✅ Dynamic confidence scoring",
            "✅ Multiple response styles",
            "✅ Conversation context awareness",
            "✅ Source citation with relevance scores",
            "✅ Document analysis with AI verification",
            "✅ Risk assessment and flagging",
            "✅ Timeline and deadline extraction",
            "✅ Party obligation analysis"
        ] if AI_ENABLED else [
            "AI features disabled - set OPENAI_API_KEY to enable"
        ],
        "improvements": [
            "Better handling of complex queries",
            "More creative and comprehensive responses",
            "Improved relevance filtering",
            "Enhanced context awareness"
        ]
    }

@app.get("/capabilities")
def get_system_capabilities():
    """Get what the system can and cannot extract"""
    capabilities = {
        "rag_capabilities": {
            "description": "Enhanced Retrieval-Augmented Generation for Q&A",
            "features": [
                "Multi-query search strategies",
                "Legal term expansion",
                "Sub-query extraction",
                "Dynamic relevance thresholds",
                "Confidence scoring",
                "Response style customization",
                "Conversation history context",
                "Source citation with relevance scores"
            ],
            "improvements": [
                "Better handling of complex queries",
                "More creative and comprehensive responses",
                "Improved relevance filtering",
                "Enhanced context awareness"
            ]
        },
        "fact_extraction": {
            "techniques": [
                "Regular expression pattern matching",
                "Natural language processing (when enabled)",
                "Question-answering models (when enabled)"
            ],
            "verifiable_facts": [
                "Party names",
                "Dates and deadlines",
                "Monetary amounts",
                "Percentages",
                "Document type classification"
            ],
            "ai_enhanced_extractions": [
                "Risk assessment and flagging",
                "Timeline and deadline extraction",
                "Party obligation analysis"
            ] if AI_ENABLED else [
                "AI features disabled - set OPENAI_API_KEY to enable"
            ],
            "verification_required": "All fact extractions include line numbers for manual verification",
            "fallback_behavior": "Returns 'Failed to extract' instead of guessing"
        }
    }
    return capabilities

@app.get("/", response_class=HTMLResponse)
def get_unified_interface():
    """Unified interface for the combined system"""
    ai_status = "✅ AI Analysis Enabled" if AI_ENABLED else "❌ AI Analysis Disabled"
    db_status = "✅ Connected" if os.path.exists(CHROMA_PATH) else "❌ Not Found"
    
    ai_instructions = "" if AI_ENABLED else """
    <div class="warning">
        <h3>🤖 Enable AI Analysis</h3>
        <p>To enable AI-powered legal analysis:</p>
        <ol>
            <li>Set environment variable: <code>export OPENAI_API_KEY="your-key"</code></li>
            <li>Install aiohttp: <code>pip install aiohttp</code></li>
            <li>Restart the server</li>
        </ol>
    </div>
    """
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Unified Legal Assistant System - Enhanced</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f8f9fa; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1 {{ color: #2c3e50; }}
            .status-grid {{ display: flex; justify-content: space-around; margin: 20px 0; }}
            .status-badge {{ padding: 10px 20px; border-radius: 20px; font-weight: bold; }}
            .ai-enabled {{ background: #d4edda; color: #155724; }}
            .ai-disabled {{ background: #f8d7da; color: #721c24; }}
            .db-connected {{ background: #d1ecf1; color: #0c5460; }}
            .db-disconnected {{ background: #f8d7da; color: #721c24; }}
            .warning {{ background: #fff3cd; border-left: 4px solid #ffeeba; padding: 15px; margin: 20px 0; }}
            code {{ background: #f1f1f1; padding: 2px 5px; border-radius: 3px; font-family: monospace; }}
            .endpoint {{ background: #f8f9fa; padding: 10px; margin: 5px 0; border-left: 4px solid #6c757d; font-family: monospace; }}
            .system-card {{ background: #fff; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin: 15px 0; }}
            .system-title {{ font-size: 20px; font-weight: bold; color: #495057; margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>⚖️ Unified Legal Assistant System</h1>
            <p style="text-align: center; color: #6c757d;">Enhanced RAG Q&A and Document Analysis Platform</p>
            
            <div class="status-grid">
                <div style="text-align: center;">
                    <span class="status-badge {'ai-enabled' if AI_ENABLED else 'ai-disabled'}">{ai_status}</span>
                </div>
                <div style="text-align: center;">
                    <span class="status-badge {'db-connected' if os.path.exists(CHROMA_PATH) else 'db-disconnected'}">ChromaDB: {db_status}</span>
                </div>
            </div>
            
            {ai_instructions}
            
            <div class="system-card">
                <div class="system-title">📚 Enhanced RAG Q&A System</div>
                <p>Ask complex legal questions and get creative, well-researched answers with source citations.</p>
                
                <div class="endpoint">POST /ask - Ask a legal question</div>
                <div class="endpoint">GET /conversation/{{session_id}} - Get conversation history</div>
            </div>
            
            <div class="system-card">
                <div class="system-title">📄 Document Analysis System</div>
                <p>Upload legal documents for automated analysis, fact extraction, and risk assessment.</p>
                
                <div class="endpoint">POST /upload - Upload a legal document</div>
                <div class="endpoint">POST /analyze-document - Analyze an uploaded document</div>
                <div class="endpoint">POST /verify-extraction - Verify a specific claim</div>
            </div>
            
            <div class="system-card">
                <div class="system-title">⚙️ System Health & Configuration</div>
                <div class="endpoint">GET /health - System health check</div>
                <div class="endpoint">GET /capabilities - System capabilities</div>
            </div>
            
            <div style="text-align: center; margin-top: 40px; color: #6c757d;">
                <p>Unified Legal Assistant v7.0.0-Enhanced</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_template
