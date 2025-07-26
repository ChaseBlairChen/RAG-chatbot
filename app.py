# - Improved app.py with Enhanced Accuracy & User Experience -
# Standard library imports
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

app = FastAPI(title="Enhanced Legal Assistant API", version="3.1.0")

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
    confidence_score: float = 0.0
    expand_available: bool = False

class ConversationHistory(BaseModel):
    session_id: str
    messages: List[Dict]
    created_at: str
    last_updated: str

class DocumentTopics(BaseModel):
    total_documents: int
    sample_topics: List[str]
    document_types: Dict[str, int]
    coverage_areas: List[str]

# --- IMPROVEMENT 1: Enhanced Retrieval System with Balanced Context Enforcement ---
def enhanced_retrieval_v3(db, query_text: str, conversation_history_context: str, k: int = 12) -> Tuple[List, Any, bool]:
    """
    Enhanced retrieval with balanced relevance checking
    Returns: (docs, search_result, has_relevant_context)
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
        
        # BALANCED: Reasonable minimum threshold based on empirical evidence
        # 0.4 threshold allows good matches like the 0.54 score in your example
        BALANCED_RELEVANCE_THRESHOLD = 0.4
        
        filtered_results = [(doc, score) for doc, score in unique_results if score > BALANCED_RELEVANCE_THRESHOLD]
        
        logger.info(f"[ENHANCED_RETRIEVAL] Found {len(unique_results)} unique results, {len(filtered_results)} above balanced threshold {BALANCED_RELEVANCE_THRESHOLD}")
        
        # If no results meet threshold, return empty - DON'T use fallback
        if not filtered_results:
            logger.warning(f"No documents meet balanced relevance threshold of {BALANCED_RELEVANCE_THRESHOLD}")
            return [], {"query_used": query_text, "scores": [], "threshold_used": BALANCED_RELEVANCE_THRESHOLD}, False
        
        docs, scores = zip(*filtered_results)
        
        return list(docs), {
            "query_used": query_text,
            "scores": list(scores),
            "threshold_used": BALANCED_RELEVANCE_THRESHOLD,
            "strategies_used": ["direct", "expanded", "sub_queries"]
        }, True
        
    except Exception as e:
        logger.error(f"[ENHANCED_RETRIEVAL] Search failed: {e}")
        return [], {"error": str(e)}, False

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
        "rights": "rights protections constitutional fundamental",
        "deferred prosecution": "deferred prosecution diversion program",
        "costs": "costs fees expenses reimbursement",
        "indigent": "indigent poverty hardship financial",
        "firearm": "firearm gun weapon possession",
        "domestic violence": "domestic violence protection order restraining",
        "custody": "custody parenting plan child support"
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

def contains_unsupported_claims(response: str, context: str) -> bool:
    """
    Check if response contains information not supported by context
    """
    # Basic checks for common signs of hallucination
    hallucination_indicators = [
        # Citing statutes not in context
        "RCW" in response and "RCW" not in context,
        "U.S.C." in response and "U.S.C." not in context,
        "CFR" in response and "CFR" not in context,
        # Response much longer than context suggests extensive external knowledge
        len(response) > len(context) * 3,
        # Common legal case citations not in context
        " v. " in response and " v. " not in context,
        # Specific legal procedures not mentioned in context
        "pursuant to" in response and "pursuant to" not in context,
        # Specific constitutional amendments not in context
        "Amendment" in response and "Amendment" not in context,
        # Federal regulations not in context
        "Federal Register" in response and "Federal Register" not in context
    ]
    
    return any(hallucination_indicators)

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

# --- IMPROVEMENT 4: Strict Context Enforcement Prompt Template ---
STRICT_CONTEXT_PROMPT_TEMPLATE = """You are a legal research assistant. You MUST follow these rules STRICTLY:

**CRITICAL CONSTRAINTS:**
1. **ONLY use information explicitly found in the provided legal documents below**
2. **If the provided context does not contain sufficient information to answer the question, you MUST say so explicitly**
3. **DO NOT use any legal knowledge from your training data**
4. **DO NOT cite statutes, cases, or regulations not mentioned in the context**
5. **DO NOT provide general legal advice or knowledge not found in the documents**

**RESPONSE REQUIREMENTS:**
- If context is insufficient: "Based on the available documents, I do not have enough information to answer this question about [topic]. The documents provided do not contain information about [specific topic]."
- For partial information: "Based on the available documents, I can only provide limited information about [topic]..."
- Always cite specific document names: [document_name.pdf]
- Structure response clearly but keep it strictly based on provided context

**RESPONSE STYLE:** {response_style}
- Concise: Provide key points only
- Balanced: Structured overview with main points
- Detailed: Comprehensive analysis

CONVERSATION HISTORY:
{conversation_history}

LEGAL DOCUMENT CONTEXT (THIS IS YOUR ONLY SOURCE OF INFORMATION):
{context}

USER QUESTION:
{questions}

Remember: If the context above does not contain information to answer the question, you must explicitly state this limitation. Do not supplement with external legal knowledge.

RESPONSE:"""

# --- Updated Main Processing Function ---
def process_query_with_strict_context(question: str, session_id: str, response_style: str = "balanced") -> QueryResponse:
    """
    Improved query processing with balanced context enforcement
    """
    try:
        # Load Database
        db = load_database()
        
        # Parse Question
        questions = parse_multiple_questions(question)
        combined_query = " ".join(questions)
        
        # Get Conversation History
        conversation_history_context = get_conversation_context(session_id, max_messages=8)
        
        # BALANCED: Enhanced Retrieval with relevance check
        results, search_result, has_relevant_context = enhanced_retrieval_v3(db, combined_query, conversation_history_context, k=12)
        
        # STRICT: If no relevant context found, return clear message
        if not has_relevant_context or not results:
            logger.warning("No sufficiently relevant documents found - refusing to generate response")
            
            # Get available topics for helpful guidance
            available_topics = get_database_topics()
            
            no_info_response = f"""**No Relevant Information Found**

I apologize, but I cannot find any relevant information in the available legal documents to answer your question about: "{question}"

**Why this happened:**
- The documents in my database do not contain information about this specific legal topic
- Your question may require information from sources not included in the current database
- The relevance threshold (0.4) was not met by any available documents

**What you can do:**
1. **Try rephrasing your question** with different keywords
2. **Consult with a qualified attorney** for authoritative legal advice
3. **Check relevant statutes directly** (state/federal)
4. **Ask about topics I do have information on** (see below)

**Available Document Topics:**
{', '.join(available_topics['coverage_areas'][:10])}

**Document Types in Database:**
{', '.join([f"{k}: {v}" for k, v in available_topics['document_types'].items()])}

*If you believe this topic should be covered by the available documents, try rephrasing your question with different keywords.*"""
            
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
        
        # Calculate confidence
        confidence_score = calculate_confidence_score(results, search_result, len(context_text))
        
        # Enhanced Prompt with stronger constraints
        formatted_questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions)) if len(questions) > 1 else questions[0]
        
        formatted_prompt = STRICT_CONTEXT_PROMPT_TEMPLATE.format(
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
        
        # VALIDATION: Check if response contains unsupported claims
        if contains_unsupported_claims(raw_response, context_text):
            logger.warning("Generated response appears to contain information not in context - requesting revision")
            
            # Try once more with even stricter prompt
            revision_prompt = f"""The following response appears to contain information not found in the provided legal documents. Please provide a response that ONLY uses information explicitly stated in the context below.

CONTEXT:
{context_text}

QUESTION: {formatted_questions}

REQUIREMENTS:
- Only use facts explicitly stated in the context above
- If the context doesn't contain enough information, say so explicitly
- Do not add any legal knowledge not found in the context
- Cite specific document names for each claim

REVISED RESPONSE:"""
            
            raw_response = call_openrouter_api(revision_prompt, api_key, api_base)
        
        # Format response based on style
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
    
    # BALANCED: Reasonable minimum for source inclusion - matches retrieval threshold
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

def get_database_topics() -> Dict:
    """Get information about what topics are covered in the database"""
    try:
        db = load_database()
        
        # Sample a variety of documents to understand coverage
        sample_results = db.similarity_search("law legal document", k=50)
        
        document_types = {}
        sample_topics = []
        coverage_areas = set()
        
        for doc in sample_results:
            # Extract file type
            source = doc.metadata.get('source', '')
            if source:
                ext = os.path.splitext(source)[1].lower()
                document_types[ext] = document_types.get(ext, 0) + 1
            
            # Extract topics from content
            content = doc.page_content.lower()
            
            # Common legal topic keywords
            legal_topics = [
                "criminal law", "civil law", "family law", "immigration", "asylum",
                "contract", "tort", "property", "constitutional", "administrative",
                "evidence", "procedure", "appeal", "court", "judge", "jury",
                "sentence", "fine", "damages", "liability", "negligence",
                "deferred prosecution", "costs", "indigent", "firearm", "domestic violence",
                "custody", "child support", "divorce", "probate", "estate"
            ]
            
            for topic in legal_topics:
                if topic in content:
                    coverage_areas.add(topic.title())
            
            # Sample content for topics
            if len(sample_topics) < 10:
                preview = doc.page_content[:100].strip()
                if preview and preview not in sample_topics:
                    sample_topics.append(preview)
        
        return {
            "total_documents": len(sample_results),
            "sample_topics": sample_topics,
            "document_types": document_types,
            "coverage_areas": sorted(list(coverage_areas))
        }
        
    except Exception as e:
        logger.error(f"Failed to get database topics: {e}")
        return {
            "total_documents": 0,
            "sample_topics": [],
            "document_types": {},
            "coverage_areas": ["Unable to determine coverage areas"]
        }

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
    """
    try:
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        db = Chroma(
            collection_name="default",
            embedding_function=embedding_function,
            persist_directory=CHROMA_PATH,
            client_settings=CHROMA_CLIENT_SETTINGS
        )
        logger.debug("Database loaded successfully")
        return db
    except Exception as e:
        logger.error(f"Failed to load database: {e}")
        raise

def get_conversation_context(session_id: str, max_messages: int = 8) -> str:
    """Get recent conversation history as context (shortened for better performance)"""
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
    elif re.search(r'(?:^|\s)\d+[\.\)]\s*.+', query_text, re.MULTILINE):
        numbered_pattern = r'(?:^|\s)\d+[\.\)]\s*(.+?)(?=(?:\s*\d+[\.\)])|$)'
        numbered_matches = re.findall(numbered_pattern, query_text, re.MULTILINE | re.DOTALL)
        for match in numbered_matches:
            match = match.strip()
            if match and len(match) > 5:
                if not match.endswith('?') and '?' not in match:
                    match += '?'
                questions.append(match)
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
                "temperature": 0.5,  # Lower temperature for more consistent responses
                "max_tokens": 2000,
                "top_p": 0.9
            }
            logger.info(f"Attempting model {i+1}/{len(models_to_try)}: {model}")
            
            response = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and result['choices']:
                    content = result['choices'][0]['message']['content']
                    if content and content.strip():
                        logger.info(f"Success with model {model}! Response length: {len(content)}")
                        return content.strip()
        except Exception as e:
            logger.error(f"Error with model {model}: {e}")
            last_exception = e
            continue
    
    # If all models failed
    error_msg = f"All models failed. Last error: {str(last_exception)}"
    logger.error(error_msg)
    
    return "I apologize, but I'm experiencing technical difficulties with the AI service. Please try again in a few moments."

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
    response = process_query_with_strict_context(user_question, session_id, query.response_style)
    return response

# --- NEW ENDPOINTS ---
@app.get("/document-topics", response_model=DocumentTopics)
def get_document_topics():
    """Get information about what topics are covered in the database"""
    topics_info = get_database_topics()
    return DocumentTopics(**topics_info)

@app.get("/debug-db")
def debug_database_enhanced():
    """Enhanced debug endpoint with relevance score testing"""
    if not os.path.exists(CHROMA_PATH):
        return {"error": "Database folder does not exist", "path": CHROMA_PATH}
    
    try:
        # Check database folder contents
        db_contents = os.listdir(CHROMA_PATH)
        
        # Load database using the centralized function
        db = load_database()
        
        # Enhanced test searches with relevance scores
        test_queries = [
            "deferred prosecution costs",
            "firearm rights restoration",
            "RCW statute law",
            "court costs indigent",
            "criminal procedure"
        ]
        
        search_results = {}
        relevance_analysis = {}
        
        for query in test_queries:
            try:
                # Get results with relevance scores
                results_with_scores = db.similarity_search_with_relevance_scores(query, k=5)
                
                results_info = []
                scores = []
                for doc, score in results_with_scores:
                    scores.append(score)
                    results_info.append({
                        "content_preview": doc.page_content[:150] + "...",
                        "relevance_score": score,
                        "source": os.path.basename(doc.metadata.get('source', 'Unknown')),
                        "meets_threshold": score > 0.4
                    })
                
                search_results[query] = {
                    "count": len(results_with_scores),
                    "results": results_info
                }
                
                # Analyze score distribution
                if scores:
                    relevance_analysis[query] = {
                        "avg_score": np.mean(scores),
                        "max_score": max(scores),
                        "min_score": min(scores),
                        "above_threshold_count": sum(1 for s in scores if s > 0.4),
                        "threshold_pass_rate": sum(1 for s in scores if s > 0.4) / len(scores)
                    }
                
            except Exception as e:
                search_results[query] = {"error": str(e)}
                relevance_analysis[query] = {"error": str(e)}
        
        # Overall database health
        total_results = sum(r.get("count", 0) for r in search_results.values() if "count" in r)
        avg_threshold_pass = np.mean([a.get("threshold_pass_rate", 0) for a in relevance_analysis.values() if "threshold_pass_rate" in a])
        
        status = "Database healthy" if total_results > 10 and avg_threshold_pass > 0.2 else "Database may have issues"
        
        return {
            "database_exists": True,
            "database_path": CHROMA_PATH,
            "database_contents": db_contents,
            "search_tests": search_results,
            "relevance_analysis": relevance_analysis,
            "overall_health": {
                "status": status,
                "total_test_results": total_results,
                "avg_threshold_pass_rate": avg_threshold_pass,
                "threshold_used": 0.4
            },
            "recommendations": [
                "Test with your specific RCW question to verify functionality",
                "Monitor logs for hallucination detection triggers",
                "Use /document-topics endpoint to understand coverage"
            ]
        }
        
    except Exception as e:
        logger.error(f"Enhanced database debug failed: {e}", exc_info=True)
        return {
            "error": f"Database test failed: {str(e)}",
            "path": CHROMA_PATH,
            "suggestion": "Try running the ingestion script to recreate the database"
        }

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
    
    # Test database connectivity with relevance scoring
    db_health = "unknown"
    if db_exists:
        try:
            db = load_database()
            test_results = db.similarity_search_with_relevance_scores("test query", k=3)
            if test_results and any(score > 0.1 for _, score in test_results):
                db_health = "healthy"
            else:
                db_health = "no_content"
        except Exception as e:
            db_health = f"error: {str(e)}"
            
    return {
        "status": "healthy" if db_exists and bool(api_key) and db_health == "healthy" else "unhealthy",
        "version": "3.1.0",
        "improvements": [
            "balanced_context_enforcement",
            "enhanced_retrieval_v3", 
            "confidence_scoring",
            "response_styles",
            "hallucination_detection",
            "document_topics_endpoint"
        ],
        "database": {
            "exists": db_exists,
            "path": CHROMA_PATH,
            "contents": db_contents,
            "health": db_health,
            "relevance_threshold": 0.4
        },
        "api_configuration": {
            "key_configured": bool(api_key),
            "base_configured": bool(api_base),
            "base_url": api_base or "https://openrouter.ai/api/v1"
        },
        "runtime_status": {
            "active_conversations": len(conversations),
            "nlp_model_available": nlp is not None,
            "sentence_model_available": sentence_model is not None
        },
        "features": {
            "balanced_threshold": "0.4 (allows good matches like 0.54 RCW example)",
            "strict_context_enforcement": "No fallback results, clear 'no information' messages",
            "response_validation": "Checks for hallucination patterns",
            "enhanced_legal_expansions": "Includes deferred prosecution, costs, indigent terms"
        }
    }

@app.get("/conversation/{session_id}", response_model=ConversationHistory)
def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conv_data = conversations[session_id]
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
    """Root endpoint with API information"""
    return {
        "message": "Enhanced Legal Assistant API is running",
        "version": "3.1.0",
        "features": [
            "Balanced relevance threshold (0.4)",
            "Strict context enforcement",
            "Response validation",
            "Enhanced legal query expansion",
            "Document topics discovery",
            "Conversation management"
        ],
        "endpoints": {
            "ask": "POST /ask - Ask legal questions",
            "document_topics": "GET /document-topics - See available topics",
            "debug_db": "GET /debug-db - Database diagnostics",
            "health": "GET /health - System health check",
            "conversations": "GET /conversations - List active sessions"
        }
    }

# --- Testing and Validation ---
@app.get("/test-relevance/{query}")
def test_query_relevance(query: str, k: int = 5):
    """Test endpoint to check relevance scores for a specific query"""
    try:
        db = load_database()
        results_with_scores = db.similarity_search_with_relevance_scores(query, k=k)
        
        test_results = []
        for doc, score in results_with_scores:
            test_results.append({
                "content_preview": doc.page_content[:200] + "...",
                "relevance_score": score,
                "meets_threshold": score > 0.4,
                "source": os.path.basename(doc.metadata.get('source', 'Unknown')),
                "page": doc.metadata.get('page')
            })
        
        analysis = {
            "query": query,
            "total_results": len(results_with_scores),
            "above_threshold": sum(1 for _, score in results_with_scores if score > 0.4),
            "threshold_used": 0.4,
            "would_generate_response": any(score > 0.4 for _, score in results_with_scores),
            "results": test_results
        }
        
        return analysis
        
    except Exception as e:
        return {"error": str(e), "query": query}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting Enhanced Legal Assistant API v3.1.0 on port {port}")
    logger.info("Key improvements:")
    logger.info("- Balanced relevance threshold (0.4) allows good matches while filtering noise")
    logger.info("- Strict context enforcement prevents hallucination")
    logger.info("- Enhanced legal query expansion for better retrieval")
    logger.info("- Response validation with hallucination detection")
    logger.info("- New /document-topics endpoint for coverage discovery")
    uvicorn.run(app, host="0.0.0.0", port=port)
