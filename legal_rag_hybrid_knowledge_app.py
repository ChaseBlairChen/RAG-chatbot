# this version uses it's pretrained knowledge while anwering questions(risk of hallucination), but it is more creative.
# this is by far the best

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import requests
import re
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Set, Any

# Third-party library imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from chromadb.config import Settings

# Set up logging FIRST
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ChromaDB Configuration ---
CHROMA_PATH = os.path.abspath(os.path.join(os.getcwd(), "chromadb-database"))
logger.info(f"Using CHROMA_PATH: {CHROMA_PATH}")

CHROMA_CLIENT_SETTINGS = Settings(
    persist_directory=CHROMA_PATH,
    anonymized_telemetry=False,
    allow_reset=True,
    is_persistent=True
)

# In-memory conversation storage
conversations: Dict[str, Dict] = {}

# Load NLP Models
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

app = FastAPI(title="Improved Legal Assistant API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Enhanced Pydantic Models ---
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
    confidence_score: float = 0.0  # NEW: Confidence in response accuracy
    expand_available: bool = False  # NEW: Whether user can request more details

# --- IMPROVEMENT 1: Enhanced Retrieval System ---
def enhanced_retrieval_v2(db, query_text: str, conversation_history_context: str, k: int = 12) -> Tuple[List, Any]:
    """
    Improved retrieval with better scoring and multi-query approach
    """
    logger.info(f"[ENHANCED_RETRIEVAL] Original query: '{query_text}'")
    
    try:
        # Strategy 1: Direct query
        results_with_scores = db.similarity_search_with_relevance_scores(query_text, k=k)
        
        # Strategy 2: Query expansion with legal terms
        legal_expanded_query = expand_legal_query(query_text)
        if legal_expanded_query != query_text:
            logger.info(f"[ENHANCED_RETRIEVAL] Expanded query: '{legal_expanded_query}'")
            expanded_results = db.similarity_search_with_relevance_scores(legal_expanded_query, k=k//2)
            results_with_scores.extend(expanded_results)
        
        # Strategy 3: Break down complex questions
        sub_queries = extract_sub_queries(query_text)
        for sub_query in sub_queries[:2]:  # Limit to avoid too many queries
            sub_results = db.similarity_search_with_relevance_scores(sub_query, k=k//3)
            results_with_scores.extend(sub_results)
        
        # Remove duplicates and re-rank
        unique_results = remove_duplicate_documents(results_with_scores)
        
        # IMPROVED: Dynamic threshold based on query complexity
        min_threshold = calculate_dynamic_threshold(query_text, unique_results)
        filtered_results = [(doc, score) for doc, score in unique_results if score > min_threshold]
        
        logger.info(f"[ENHANCED_RETRIEVAL] Found {len(unique_results)} unique results, {len(filtered_results)} above threshold {min_threshold:.3f}")
        
        final_results = filtered_results if filtered_results else unique_results[:k//2]
        docs, scores = zip(*final_results) if final_results else ([], [])
        
        return list(docs), {
            "query_used": query_text,
            "scores": list(scores),
            "threshold_used": min_threshold,
            "strategies_used": ["direct", "expanded", "sub_queries"]
        }
        
    except Exception as e:
        logger.error(f"[ENHANCED_RETRIEVAL] Search failed: {e}")
        return [], {"error": str(e)}

def expand_legal_query(query: str) -> str:
    """Expand query with legal synonyms and terms"""
    legal_expansions = {
        "asylum": "asylum refugee protection persecution",
        "immigration": "immigration visa deportation removal",
        "contract": "contract agreement terms conditions breach",
        "criminal": "criminal penal prosecution sentence conviction",
        "civil": "civil tort liability damages compensation",
        "court": "court tribunal judge judicial proceeding",
        "law": "law statute regulation rule code",
        "rights": "rights protections constitutional fundamental"
    }
    
    expanded_terms = []
    query_lower = query.lower()
    
    for term, expansion in legal_expansions.items():
        if term in query_lower:
            expanded_terms.extend(expansion.split())
    
    if expanded_terms:
        return f"{query} {' '.join(set(expanded_terms))}"
    return query

def extract_sub_queries(query: str) -> List[str]:
    """Extract sub-questions from complex queries"""
    sub_queries = []
    
    # Split on common conjunctions
    conjunctions = [" and ", " or ", " also ", " plus ", " additionally "]
    current_query = query
    
    for conj in conjunctions:
        if conj in current_query.lower():
            parts = current_query.split(conj)
            sub_queries.extend([part.strip() for part in parts if len(part.strip()) > 10])
            break
    
    # Extract specific legal concepts
    legal_patterns = [
        r"what (?:is|are) ([^?]+)\??",
        r"how (?:does|do) ([^?]+)\??",
        r"when (?:is|are) ([^?]+)\??",
        r"(?:explain|describe) ([^?]+)\??"
    ]
    
    for pattern in legal_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for match in matches:
            if len(match.strip()) > 5:
                sub_queries.append(f"what is {match.strip()}?")
    
    return sub_queries[:3]  # Limit to 3 sub-queries

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
    return sorted(unique_results, key=lambda x: x[1], reverse=True)

def calculate_dynamic_threshold(query: str, results: List[Tuple]) -> float:
    """Calculate dynamic threshold based on query complexity and result distribution"""
    if not results:
        return 0.3
    
    scores = [score for _, score in results]
    
    # Query complexity factors
    is_complex = len(query.split()) > 8 or '?' in query[:-1]  # Multiple questions
    has_legal_terms = any(term in query.lower() for term in ['law', 'legal', 'court', 'statute', 'regulation'])
    
    # Statistical analysis of scores
    if len(scores) > 1:
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        if is_complex:
            return max(0.25, avg_score - std_score)  # Lower threshold for complex queries
        elif has_legal_terms:
            return max(0.35, avg_score - 0.5 * std_score)  # Medium threshold
        else:
            return max(0.4, avg_score - 0.3 * std_score)  # Higher threshold
    
    return 0.3  # Default fallback

# --- IMPROVEMENT 2: Response Style Management ---
def format_response_by_style(content: str, sources: List[Dict], style: str = "balanced") -> Tuple[str, bool]:
    """Format response based on user's preferred style"""
    
    if style == "concise":
        # Extract key points and create concise response
        concise_response = create_concise_response(content, sources)
        return concise_response, True  # Expansion available
    
    elif style == "detailed":
        # Return full detailed response
        detailed_response = create_detailed_response(content, sources)
        return detailed_response, False  # No expansion needed
    
    else:  # balanced
        # Provide balanced response with clear structure
        balanced_response = create_balanced_response(content, sources)
        return balanced_response, True  # Expansion available

def create_concise_response(content: str, sources: List[Dict]) -> str:
    """Create a concise, bullet-point response"""
    # This is a simplified version - in practice, you'd use NLP to extract key points
    lines = content.split('\n')
    key_points = []
    
    for line in lines[:5]:  # Limit to first 5 lines
        if line.strip() and not line.startswith('#'):
            key_points.append(f"• {line.strip()}")
    
    concise = f"""**Quick Answer:**
{chr(10).join(key_points)}

💡 *Need more details? Ask me to expand on any point above.*"""
    
    return concise

def create_balanced_response(content: str, sources: List[Dict]) -> str:
    """Create a balanced response with clear sections"""
    # Structure the response with clear sections
    if len(content) > 800:
        preview = content[:600] + "..."
        balanced = f"""{preview}

📖 **Want the complete analysis?** Ask me to provide the full detailed response.
🔍 **Have specific questions?** Ask about any particular aspect mentioned above."""
    else:
        balanced = content
    
    return balanced

def create_detailed_response(content: str, sources: List[Dict]) -> str:
    """Return the full detailed response"""
    return content

# --- IMPROVEMENT 3: Confidence Scoring ---
def calculate_confidence_score(results: List, search_result: Dict, response_length: int) -> float:
    """Calculate confidence score based on multiple factors"""
    if not results:
        return 0.1
    
    scores = search_result.get("scores", [])
    if not scores:
        return 0.2
    
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
    
    return min(1.0, max(0.0, confidence))

# --- IMPROVEMENT 4: Enhanced Prompt Template ---
IMPROVED_PROMPT_TEMPLATE = """You are a legal research assistant. Your responses must be STRICTLY based on the provided legal documents.

CRITICAL REQUIREMENTS:
1. **ONLY use information from the provided context below**
2. **If the context doesn't contain sufficient information, explicitly state this**
3. **Cite specific document names for each claim**
4. **Do NOT add general legal knowledge not found in the context**

RESPONSE STYLE: {response_style}
- Concise: Provide key points only
- Balanced: Structured overview with main points
- Detailed: Comprehensive analysis

CONVERSATION HISTORY:
{conversation_history}

LEGAL DOCUMENT CONTEXT (USE ONLY THIS INFORMATION):
{context}

USER QUESTION:
{questions}

INSTRUCTIONS:
- Base your response ONLY on the provided context
- If context is insufficient, say: "Based on the available documents, I can only provide limited information..."
- Cite document names for each fact: [document_name.pdf]
- If no relevant information exists, say: "The available documents do not contain information about this topic."

RESPONSE:"""

# --- Updated Main Processing Function ---
def process_query_improved(question: str, session_id: str, response_style: str = "balanced") -> QueryResponse:
    """
    Improved query processing with enhanced accuracy and user experience
    """
    try:
        # Load Database
        db = load_database()
        
        # Parse Question
        questions = parse_multiple_questions(question)
        combined_query = " ".join(questions)
        
        # Get Conversation History
        conversation_history_context = get_conversation_context(session_id, max_messages=8)
        
        # IMPROVED: Enhanced Retrieval
        results, search_result = enhanced_retrieval_v2(db, combined_query, conversation_history_context, k=12)
        
        if not results:
            logger.warning("No relevant documents found")
            no_info_response = "I couldn't find any relevant information in the available legal documents to answer your question. The documents may not contain information about this specific topic."
            add_to_conversation(session_id, "assistant", no_info_response)
            return QueryResponse(
                response=no_info_response,
                error=None,
                context_found=False,
                sources=[],
                session_id=session_id,
                confidence_score=0.0,
                expand_available=False
            )
        
        # Create Context with better filtering
        context_text, source_info = create_enhanced_context(results, search_result, questions)
        
        # IMPROVED: Calculate confidence before generating response
        confidence_score = calculate_confidence_score(results, search_result, len(context_text))
        
        # Enhanced Prompt
        formatted_questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions)) if len(questions) > 1 else questions[0]
        
        formatted_prompt = IMPROVED_PROMPT_TEMPLATE.format(
            response_style=response_style.capitalize(),
            conversation_history=conversation_history_context if conversation_history_context else "No previous conversation.",
            context=context_text,
            questions=formatted_questions
        )
        
        # Call LLM
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        
        if not api_key:
            error_msg = "OPENAI_API_KEY environment variable not set."
            return QueryResponse(
                response=None,
                error=f"Configuration Error: {error_msg}",
                context_found=True,
                sources=source_info,
                session_id=session_id,
                confidence_score=0.0,
                expand_available=False
            )
        
        raw_response = call_openrouter_api(formatted_prompt, api_key, api_base)
        
        # IMPROVED: Format response based on style
        formatted_response, expand_available = format_response_by_style(raw_response, source_info, response_style)
        
        # Add sources with confidence indicator
        if source_info:
            formatted_response += f"\n\n**SOURCES** (Confidence: {confidence_score:.1%}):\n"
            for source in source_info:
                page_info = f", Page {source['page']}" if source['page'] is not None else ""
                formatted_response += f"- {source['file_name']}{page_info} (Relevance: {source['relevance']:.2f})\n"
        
        # Update conversation
        add_to_conversation(session_id, "user", question)
        add_to_conversation(session_id, "assistant", formatted_response, source_info)
        
        return QueryResponse(
            response=formatted_response,
            error=None,
            context_found=True,
            sources=source_info,
            session_id=session_id,
            confidence_score=confidence_score,
            expand_available=expand_available
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        error_msg = f"Failed to process your request: {str(e)}"
        return QueryResponse(
            response=None,
            error=error_msg,
            context_found=False,
            sources=[],
            session_id=session_id,
            confidence_score=0.0,
            expand_available=False
        )

def create_enhanced_context(results: List, search_result: Dict, questions: List[str]) -> Tuple[str, List[Dict]]:
    """Enhanced context creation with better relevance filtering"""
    if not results:
        return "", []
    
    context_parts = []
    source_info = []
    seen_sources = set()
    
    # IMPROVED: Higher minimum threshold for source inclusion
    MIN_RELEVANCE_FOR_CONTEXT = 0.4
    
    for i, (doc, score) in enumerate(zip(results, search_result.get("scores", [0.0]*len(results)))):
        # Skip low-relevance documents
        if score < MIN_RELEVANCE_FOR_CONTEXT:
            continue
            
        content = doc.page_content.strip()
        if not content:
            continue
        
        source_path = doc.metadata.get('source', 'Unknown Source')
        page = doc.metadata.get('page', None)
        
        source_id = (source_path, page)
        if source_id in seen_sources:
            continue
        seen_sources.add(source_id)
        
        display_source = os.path.basename(source_path)
        page_info = f" (Page {page})" if page is not None else ""
        
        # Truncate very long content
        if len(content) > 800:
            content = content[:800] + "... [truncated]"
        
        context_part = f"[{display_source}{page_info}] (Relevance: {score:.2f}):\n{content}"
        context_parts.append(context_part)
        
        source_info.append({
            'id': i+1,
            'file_name': display_source,
            'page': page,
            'relevance': score,
            'full_path': source_path
        })
    
    context_text = "\n\n---\n\n".join(context_parts)
    return context_text, source_info

# --- Updated API Endpoint ---
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
            confidence_score=0.0,
            expand_available=False
        )
    
    # Process with improvements
    response = process_query_improved(user_question, session_id, query.response_style)
    return response

# ... [Include all the utility functions from the original code] ...
def cleanup_expired_conversations():
    """Remove conversations older than 1 hour"""
    now = datetime.utcnow()
    expired_sessions = [
        session_id for session_id, data in conversations.items()
        if now - data['last_accessed'] > timedelta(hours=1)
    ]
    for session_id in expired_sessions:
        del conversations[session_id]

def load_database():
    """Load the Chroma database"""
    try:
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(
            collection_name="default",
            embedding_function=embedding_function,
            persist_directory=CHROMA_PATH,
            client_settings=CHROMA_CLIENT_SETTINGS
        )
        return db
    except Exception as e:
        logger.error(f"Failed to load database: {e}")
        raise

def get_conversation_context(session_id: str, max_messages: int = 8) -> str:
    """Get conversation context (shortened for better performance)"""
    if session_id not in conversations:
        return ""
    messages = conversations[session_id]['messages'][-max_messages:]
    context_parts = []
    for msg in messages:
        role = msg['role'].upper()
        content = msg['content'][:400] + "..." if len(msg['content']) > 400 else msg['content']
        context_parts.append(f"{role}: {content}")
    return "\n".join(context_parts) if context_parts else ""

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
    if len(conversations[session_id]['messages']) > 20:
        conversations[session_id]['messages'] = conversations[session_id]['messages'][-20:]

def parse_multiple_questions(query_text: str) -> List[str]:
    """Parse multiple questions from input"""
    questions = []
    query_text = query_text.strip()
    
    if '?' in query_text and not query_text.endswith('?'):
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

def call_openrouter_api(prompt: str, api_key: str, api_base: str = "https://openrouter.ai/api/v1") -> str:
    """Call OpenRouter API with fallback models"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Legal Assistant"
    }
    
    models_to_try = [
        "deepseek/deepseek-chat-v3-0324:free",
        "microsoft/phi-3-mini-128k-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free"
    ]
    
    for model in models_to_try:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,  # Lower temperature for more consistent responses
                "max_tokens": 2000,
                "top_p": 0.9
            }
            
            response = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and result['choices']:
                    content = result['choices'][0]['message']['content']
                    if content and content.strip():
                        return content.strip()
        except Exception as e:
            logger.error(f"Error with model {model}: {e}")
            continue
    
    return "I apologize, but I'm experiencing technical difficulties. Please try again."

# Additional endpoints...
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "3.0.0", "improvements": ["enhanced_retrieval", "confidence_scoring", "response_styles"]}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
# --- Load NLP Models ---
# Load models once at startup for efficiency
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
# --- End Load NLP Models ---

app = FastAPI(title="Legal Assistant API", description="Retrieval-Augmented Generation API for Legal Documents", version="2.1.1")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class Query(BaseModel):
    question: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: Optional[str] = None
    error: Optional[str] = None
    context_found: bool = False
    sources: Optional[list] = None
    session_id: str

class ConversationHistory(BaseModel):
    session_id: str
    messages: List[Dict]
    created_at: str
    last_updated: str
# --- End Pydantic Models ---

# --- Utility Functions ---
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

def load_database():
    """
    Loads the Chroma database with explicit settings.
    This function centralizes database loading logic.
    """
    try:
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        db = Chroma(
            collection_name="default", # Explicitly specify collection name
            embedding_function=embedding_function,
            persist_directory=CHROMA_PATH,
            client_settings=CHROMA_CLIENT_SETTINGS # Use the explicit settings
        )
        logger.debug("Database loaded successfully")
        return db
    except Exception as e:
        logger.error(f"Failed to load database: {e}")
        raise

def get_conversation_context(session_id: str, max_messages: int = 12) -> str:
    """Get recent conversation history as context"""
    if session_id not in conversations:
        return ""
    messages = conversations[session_id]['messages'][-max_messages:]
    context_parts = []
    for msg in messages:
        role = msg['role'].upper()
        content = msg['content']
        # For conversation history, keep more content but still truncate very long messages
        if len(content) > 800:
            content = content[:800] + "..."
        context_parts.append(f"{role}: {content}")
    
    # If we have conversation history, make it clear this is ongoing context
    if context_parts:
        return "Previous conversation:\n" + "\n".join(context_parts)
    return ""

def add_to_conversation(session_id: str, role: str, content: str, sources: Optional[List] = None):
    """Add a message to the conversation history"""
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
    # Keep only last 30 messages to prevent memory issues but maintain more history
    if len(conversations[session_id]['messages']) > 30:
        conversations[session_id]['messages'] = conversations[session_id]['messages'][-30:]

def parse_multiple_questions(query_text: str) -> List[str]:
    """Parse a query that might contain multiple questions."""
    questions = []
    query_text = query_text.strip()
    if not query_text:
        return questions

    # Check for question mark delimiter
    if '?' in query_text and not query_text.endswith('?'):
        parts = query_text.split('?')
        for part in parts:
            part = part.strip()
            if part:
                questions.append(part + '?')
    # Check for numbered list format (e.g., "1. What is... 2. How does...")
    elif re.search(r'(?:^|\s)\d+[\.\)]\s*.+', query_text, re.MULTILINE):
        numbered_pattern = r'(?:^|\s)\d+[\.\)]\s*(.+?)(?=(?:\s*\d+[\.\)])|$)'
        numbered_matches = re.findall(numbered_pattern, query_text, re.MULTILINE | re.DOTALL)
        for match in numbered_matches:
            match = match.strip()
            if match and len(match) > 5: # Basic sanity check
                if not match.endswith('?') and '?' not in match:
                    match += '?'
                questions.append(match)
    
    # If no specific format found, treat as single question
    if not questions:
        final_question = query_text
        if not final_question.endswith('?') and '?' not in final_question:
            final_question += '?'
        questions = [final_question]
        
    return questions

# --- End Utility Functions ---

# --- AI Agent and Retrieval Logic ---
# --- (Based on the provided original code snippets) ---

# Placeholder for more complex agent logic if needed in the future
# For now, we'll use the core retrieval augmented by conversation history

def enhanced_retrieval(db, query_text: str, conversation_history_context: str, k: int = 8) -> Tuple[List, Any]:
    """
    Performs retrieval, potentially using conversation history to refine the query.
    This is a simplified version based on the original logic.
    """
    # First, try the original query
    logger.info(f"[RETRIEVAL] Searching for query: '{query_text}'")
    
    try:
        # Perform similarity search with scores using original query
        results_with_scores = db.similarity_search_with_relevance_scores(query_text, k=k)
        logger.info(f"[RETRIEVAL] Found {len(results_with_scores)} raw results")
        
        # If we have conversation history and few results, try combined query
        if conversation_history_context and len(results_with_scores) < 3:
            combined_query = f"{conversation_history_context}\n\n{query_text}".strip()
            logger.info(f"[RETRIEVAL] Trying combined query: '{combined_query}'")
            combined_results = db.similarity_search_with_relevance_scores(combined_query, k=k)
            # Use combined results if they're better
            if len(combined_results) > len(results_with_scores):
                results_with_scores = combined_results
        
        # Filter based on a minimum relevance score
        filtered_results = [(doc, score) for doc, score in results_with_scores if score > 0.2]
        logger.info(f"[RETRIEVAL] Filtered to {len(filtered_results)} results with score > 0.2")
        
        # If we have filtered results, use them; otherwise use all results
        final_results = filtered_results if filtered_results else results_with_scores
        
        docs, scores = zip(*final_results) if final_results else ([], [])
        return list(docs), {"query_used": query_text, "scores": list(scores)}
        
    except Exception as e:
        logger.error(f"[RETRIEVAL] Search failed: {e}")
        return [], {"error": str(e)}

def create_context(results: List, search_result: Dict, questions: List[str]) -> Tuple[str, List[Dict]]:
    """
    Create a context string and source info list from retrieved documents.
    Based on the original `create_universal_context` logic.
    """
    if not results:
        return "", []

    context_parts = []
    source_info = []
    seen_sources = set() # Avoid duplicate source entries

    for i, (doc, score) in enumerate(zip(results, search_result.get("scores", [0.0]*len(results)))):
        content = doc.page_content.strip()
        if not content:
            continue

        source_path = doc.metadata.get('source', 'Unknown Source')
        page = doc.metadata.get('page', None)
        
        # Create a unique identifier for the source to check for duplicates
        source_id = (source_path, page)
        if source_id in seen_sources:
             continue
        seen_sources.add(source_id)

        display_source = os.path.basename(source_path)
        page_info = f" (Page {page})" if page is not None else ""
        
        # Format context part for the prompt
        context_part = f"[{display_source}{page_info}] (Relevance: {score:.2f}): {content}"
        context_parts.append(context_part)
        
        # Store source info for the response
        source_info.append({
            'id': i+1,
            'file_name': display_source,
            'page': page,
            'relevance': score,
            'full_path': source_path
        })

    context_text = "\n\n".join(context_parts)
    return context_text, source_info

# --- Prompt Template ---
ENHANCED_PROMPT_TEMPLATE = """You are a legal assistant providing detailed, comprehensive answers about legal documents and policies. Your role is to analyze legal documents thoroughly and provide informative, well-structured responses.

RESPONSE REQUIREMENTS:
1. **Provide Detailed Analysis**: Give comprehensive explanations, not brief summaries. Include relevant details, context, and implications.
2. **Use All Available Context**: Analyze ALL provided documents and sources. Don't limit yourself to just one source when multiple are available.
3. **Structure Your Response**: Use clear paragraphs, bullet points when appropriate, and logical organization.
4. **Answer Follow-up Questions**: If the user asks about a document you just cited, provide detailed information from that document.
5. **Cite Properly**: Use document names as shown in context (e.g., [document_name.pdf] or [document_name.pdf (Page X)]).
6. **Be Conversational and Context-Aware**: 
   - Always acknowledge and reference previous parts of our conversation when relevant
   - If the user asks about something mentioned earlier, explicitly connect it to the previous discussion
   - Build upon previous answers and maintain conversational flow
   - Reference earlier questions and answers when they provide context for the current question

CONVERSATION HISTORY (ALWAYS consider this context):
{conversation_history}

AVAILABLE LEGAL DOCUMENTS AND CONTEXT:
{context}

USER QUESTION:
{questions}

Provide a comprehensive, detailed response using the available legal documents AND the conversation history above. If this question relates to something we discussed earlier, acknowledge that connection. Include specific provisions, explanations, and relevant details from the sources. If multiple documents are relevant, synthesize information from all of them. Aim for thorough, informative responses that fully address the user's question while maintaining awareness of our ongoing conversation.

RESPONSE:"""
# --- End Prompt Template ---

def call_openrouter_api(prompt: str, api_key: str, api_base: str = "https://openrouter.ai/api/v1") -> str:
    """
    Call the OpenRouter API (or compatible service) with the given prompt.
    Includes fallback logic for models and better error handling.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000", # Or your frontend URL
        "X-Title": "Legal Assistant"
    }
    
    # List of models to try in order
    models_to_try = [
        "deepseek/deepseek-chat",
        "microsoft/phi-3-mini-128k-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "google/gemma-2-9b-it:free",
        "mistralai/mistral-7b-instruct:free",
        "openchat/openchat-7b:free"
    ]

    logger.info(f"Trying {len(models_to_try)} models...")
    last_exception = None

    for i, model in enumerate(models_to_try):
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 3000,
                "top_p": 0.95
            }
            logger.info(f"Attempting model {i+1}/{len(models_to_try)}: {model}")
            
            response = requests.post(
                f"{api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            # Log response details for debugging
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            
            # Check if response is empty
            if not response.content:
                logger.error(f"Empty response from {model}")
                last_exception = Exception("Empty response from API")
                continue
                
            # Log raw response content for debugging
            response_text = response.content.decode('utf-8')
            logger.info(f"Raw response content (first 200 chars): {response_text[:200]}")
            
            # Check HTTP status
            if response.status_code != 200:
                logger.error(f"HTTP {response.status_code} error from {model}: {response_text}")
                last_exception = requests.exceptions.HTTPError(f"HTTP {response.status_code}: {response_text}")
                continue
            
            # Try to parse JSON
            try:
                result = response.json()
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON decode error from {model}: {json_err}")
                logger.error(f"Response content: {response_text}")
                last_exception = json_err
                continue
            
            # Check if response has expected structure
            if 'choices' not in result or not result['choices']:
                logger.error(f"Invalid response structure from {model}: {result}")
                last_exception = Exception(f"Invalid response structure: {result}")
                continue
                
            if 'message' not in result['choices'][0] or 'content' not in result['choices'][0]['message']:
                logger.error(f"Missing message content from {model}: {result}")
                last_exception = Exception(f"Missing message content: {result}")
                continue
            
            response_content = result['choices'][0]['message']['content']
            if not response_content or not response_content.strip():
                logger.error(f"Empty content from {model}")
                last_exception = Exception("Empty content in response")
                continue
                
            logger.info(f"Success with model {model}! Response length: {len(response_content)}")
            return response_content.strip()
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout error for model {model}")
            last_exception = Exception("Request timeout")
            continue
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error for model {model}: {conn_err}")
            last_exception = conn_err
            continue
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request error for model {model}: {req_err}")
            last_exception = req_err
            continue
        except Exception as e:
            logger.error(f"Unexpected error for model {model}: {e}")
            last_exception = e
            continue

    # If all models failed
    error_msg = f"All models failed. Last error: {str(last_exception)}"
    logger.error(error_msg)
    
    # Return a fallback response instead of raising an exception
    fallback_response = (
        "I apologize, but I'm currently experiencing technical difficulties with the AI service. "
        "This could be due to API limitations, network issues, or temporary service unavailability. "
        "Please try again in a few moments, or contact support if the issue persists."
    )
    
    return fallback_response

# --- End AI Agent and Retrieval Logic ---

# --- Main Query Processing ---
def process_query(question: str, session_id: str) -> QueryResponse:
    """
    Process the user query using the RAG pipeline and AI agent, including conversation history.
    """
    try:
        # --- Load Database ---
        db = load_database()
        
        # Test database connectivity
        test_results = db.similarity_search("test", k=1)
        logger.info(f"Database loaded successfully with {len(test_results)} test results")

        # --- Parse Question ---
        questions = parse_multiple_questions(question)
        logger.info(f"Parsed {len(questions)} questions: {questions}")
        combined_query = " ".join(questions)
        # --- End Parse Question ---

        # --- Get Conversation History ---
        conversation_history_list = conversations.get(session_id, {}).get('messages', [])
        conversation_history_context = get_conversation_context(session_id, max_messages=12)
        logger.info(f"Using conversation history with {len(conversation_history_list)} total messages")
        # --- End Get Conversation History ---

        # --- Perform Retrieval ---
        results, search_result = enhanced_retrieval(db, combined_query, conversation_history_context, k=8)
        logger.info(f"Retrieved {len(results)} results")
        # --- End Perform Retrieval ---

        if not results:
            logger.warning("No relevant documents found")
            no_info_response = "I couldn't find any relevant information in the documents to answer your question. Please try rephrasing your question or check if the documents contain information about this topic."
            add_to_conversation(session_id, "assistant", no_info_response)
            return QueryResponse(
                response=no_info_response,
                error=None,
                context_found=False,
                sources=[],
                session_id=session_id
            )

        # --- Create Context ---
        context_text, source_info = create_context(results, search_result, questions)
        logger.info(f"Created context with {len(source_info)} sources")
        # --- End Create Context ---

        # --- Format Prompt ---
        if len(questions) > 1:
            formatted_questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        else:
            formatted_questions = questions[0]

        formatted_prompt = ENHANCED_PROMPT_TEMPLATE.format(
            conversation_history=conversation_history_context if conversation_history_context else "No previous conversation in this session.",
            context=context_text,
            questions=formatted_questions
        )
        logger.info(f"Prompt length: {len(formatted_prompt)} characters")
        logger.info(f"Conversation history length: {len(conversation_history_context)} characters")
        # --- End Format Prompt ---

        # --- Call LLM ---
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        
        if not api_key:
            error_msg = "OPENAI_API_KEY environment variable not set."
            logger.error(error_msg)
            add_to_conversation(session_id, "assistant", f"Configuration Error: {error_msg}")
            return QueryResponse(
                response=None,
                error=f"Configuration Error: {error_msg}",
                context_found=True,
                sources=source_info,
                session_id=session_id
            )

        response_text = call_openrouter_api(formatted_prompt, api_key, api_base)
        if not response_text:
            response_text = "I received an empty response. Please try again."
        # --- End Call LLM ---

        # --- Post-process Response (Add Sources) ---
        MIN_RELEVANCE_SCORE_FOR_SOURCE_DISPLAY = 0.25
        relevant_source_info = [source for source in source_info if source['relevance'] >= MIN_RELEVANCE_SCORE_FOR_SOURCE_DISPLAY]

        # Add sources section only if there are relevant sources
        if relevant_source_info:
            response_text += "\n\n**SOURCES:**\n"
            for source in relevant_source_info:
                page_info = f", Page {source['page']}" if source['page'] is not None else ""
                response_text += f"- {source['file_name']}{page_info} (Relevance: {source['relevance']:.2f})\n"
        # --- End Post-process Response ---

        # --- Update Conversation History ---
        # Add user query and assistant response
        add_to_conversation(session_id, "user", question) # Add original question
        add_to_conversation(session_id, "assistant", response_text, source_info) # Add full source info to history
        logger.info(f"Successfully generated response of length {len(response_text)}")
        # --- End Update Conversation History ---

        return QueryResponse(
            response=response_text,
            error=None,
            context_found=True,
            sources=relevant_source_info, # Return the filtered list for the API response
            session_id=session_id
        )

    except Exception as e:
        logger.error(f"Failed to load database or process query: {e}", exc_info=True)
        error_msg = f"Failed to process your request: {str(e)}"
        add_to_conversation(session_id, "assistant", error_msg)
        return QueryResponse(
            response=None,
            error=error_msg,
            context_found=False,
            sources=[],
            session_id=session_id
        )
# --- End Main Query Processing ---

# --- API Endpoints ---
@app.post("/ask", response_model=QueryResponse)
async def ask_question(query: Query):
    """
    Ask a question and get a response based on the ingested documents.
    Maintains conversation context using session_id.
    """
    # --- Cleanup Expired Conversations ---
    cleanup_expired_conversations()
    # --- End Cleanup ---

    # --- Manage Session ---
    session_id = query.session_id
    if not session_id:
        # Create new session
        session_id = str(uuid.uuid4())
        conversations[session_id] = {
            "messages": [],
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow()
        }
        logger.info(f"Created new conversation session: {session_id}")
    elif session_id not in conversations:
        # Session ID provided but not found, treat as new
        conversations[session_id] = {
             "messages": [],
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow()
        }
        logger.info(f"Recreated conversation session: {session_id}")
    else:
        # Existing session, update last accessed
        conversations[session_id]["last_accessed"] = datetime.utcnow()
    # --- End Manage Session ---

    # --- Process the Query ---
    user_question = query.question.strip()
    if not user_question:
        return QueryResponse(
            response=None,
            error="Question cannot be empty.",
            context_found=False,
            sources=[],
            session_id=session_id
        )

    logger.info(f"Received query: '{user_question}'")

    response = process_query(user_question, session_id)
    return response
    # --- End Process Query ---

@app.get("/health")
def health_check():
    """Comprehensive health check endpoint"""
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE")
    
    db_exists = os.path.exists(CHROMA_PATH)
    db_contents = []
    if db_exists:
        try:
            db_contents = os.listdir(CHROMA_PATH)
        except Exception as e:
            db_contents = [f"Error reading directory: {e}"]
            
    return {
        "status": "healthy" if db_exists and bool(api_key) else "unhealthy",
        "database_exists": db_exists,
        "database_path": CHROMA_PATH,
        "database_contents": db_contents,
        "api_key_configured": bool(api_key),
        "api_base_configured": bool(api_base),
        "active_conversations": len(conversations),
        "ai_agent_status": {
            "loaded": True,
            "nlp_model_available": nlp is not None,
            "sentence_model_available": sentence_model is not None
        }
    }

@app.get("/debug-db")
def debug_database():
    """Debug endpoint to check database status"""
    if not os.path.exists(CHROMA_PATH):
        return {"error": "Database folder does not exist", "path": CHROMA_PATH}
    
    try:
        # Check database folder contents
        db_contents = os.listdir(CHROMA_PATH)
        
        # Load database using the centralized function
        db = load_database()
        
        # Perform test searches
        test_queries = ["test", "document", "content", "information"]
        search_results = {}
        for query in test_queries:
            try:
                results = db.similarity_search(query, k=3)
                previews = [doc.page_content[:100] + "..." for doc in results]
                search_results[query] = {
                    "count": len(results),
                    "previews": previews
                }
            except Exception as e:
                search_results[query] = {"error": str(e)}
        
        status = "Database appears to be working" if any(r.get("count", 0) > 0 for r in search_results.values()) else "Database exists but no search results found"
        
        return {
            "database_exists": True,
            "database_path": CHROMA_PATH,
            "database_contents": db_contents,
            "search_tests": search_results,
            "status": status
        }
    except Exception as e:
        logger.error(f"Database debug failed: {e}", exc_info=True)
        return {
            "error": f"Database test failed: {str(e)}",
            "path": CHROMA_PATH,
            "suggestion": "Try running the ingestion script to recreate the database"
        }


@app.get("/conversation/{session_id}", response_model=ConversationHistory)
def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conv_data = conversations[session_id]
    # Ensure 'messages' key exists
    messages = conv_data.get('messages', [])
    return ConversationHistory(
        session_id=session_id,
        messages=messages,
        created_at=conv_data['created_at'].isoformat() if 'created_at' in conv_data else datetime.utcnow().isoformat(),
        last_updated=conv_data['last_accessed'].isoformat() if 'last_accessed' in conv_data else datetime.utcnow().isoformat()
    )

@app.delete("/conversation/{session_id}")
def clear_conversation(session_id: str):
    """Clear conversation history for a session"""
    if session_id in conversations:
        del conversations[session_id]
        return {"message": f"Conversation {session_id} cleared"}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")

@app.get("/conversations")
def list_conversations():
    """List all active conversations"""
    return {
        "active_conversations": len(conversations),
        "sessions": [
            {
                "session_id": session_id,
                "message_count": len(data.get('messages', [])),
                "created_at": data.get('created_at', datetime.min).isoformat(),
                "last_updated": data.get('last_accessed', datetime.min).isoformat()
            }
            for session_id, data in conversations.items()
        ]
    }

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Legal Assistant API is running"}

# --- End API Endpoints ---

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

# - End of app.py -
