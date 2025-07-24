from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import requests
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional, List, Dict
import logging
import uuid
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PATH = os.path.join(os.getcwd(), "my_chroma_db")
logger.info(f"Checking CHROMA_PATH: {CHROMA_PATH}")
if os.path.exists(CHROMA_PATH):
    logger.info(f"Database contents: {os.listdir(CHROMA_PATH)}")
else:
    logger.warning("Database folder not found!")

# In-memory conversation storage (in production, use Redis or a database)
conversations: Dict[str, Dict] = {}

def cleanup_old_conversations():
    """Remove conversations older than 24 hours"""
    cutoff_time = datetime.now() - timedelta(hours=24)
    expired_sessions = [
        session_id for session_id, data in conversations.items()
        if data.get('last_updated', datetime.min) < cutoff_time
    ]
    for session_id in expired_sessions:
        del conversations[session_id]
    logger.info(f"Cleaned up {len(expired_sessions)} expired conversations")

def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get existing session or create new one"""
    cleanup_old_conversations()
    
    if session_id and session_id in conversations:
        conversations[session_id]['last_updated'] = datetime.now()
        return session_id
    
    # Create new session
    new_session_id = str(uuid.uuid4())
    conversations[new_session_id] = {
        'messages': [],
        'created_at': datetime.now(),
        'last_updated': datetime.now()
    }
    logger.info(f"Created new conversation session: {new_session_id}")
    return new_session_id

def add_to_conversation(session_id: str, role: str, content: str, sources: Optional[List] = None):
    """Add a message to the conversation history"""
    if session_id not in conversations:
        conversations[session_id] = {
            'messages': [],
            'created_at': datetime.now(),
            'last_updated': datetime.now()
        }
    
    message = {
        'role': role,
        'content': content,
        'timestamp': datetime.now().isoformat(),
        'sources': sources or []
    }
    
    conversations[session_id]['messages'].append(message)
    conversations[session_id]['last_updated'] = datetime.now()
    
    # Keep only last 20 messages to prevent memory issues
    if len(conversations[session_id]['messages']) > 20:
        conversations[session_id]['messages'] = conversations[session_id]['messages'][-20:]

def get_conversation_context(session_id: str, max_messages: int = 10) -> str:
    """Get recent conversation history as context"""
    if session_id not in conversations:
        return ""
    
    messages = conversations[session_id]['messages'][-max_messages:]
    
    context_parts = []
    for msg in messages:
        role = msg['role'].upper()
        content = msg['content']
        # Truncate very long messages
        if len(content) > 1000:
            content = content[:1000] + "..."
        context_parts.append(f"{role}: {content}")
    
    return "\n".join(context_parts)

def parse_multiple_questions(query_text: str) -> list:
    """Parse multiple questions from a single input"""
    if not query_text or not query_text.strip():
        return [""]
    
    questions = []
    query_text = query_text.strip()
    
    question_marks = query_text.count('?')
    
    if question_marks > 1:
        potential_questions = query_text.split('?')
        for q in potential_questions:
            q = q.strip()
            if q and len(q) > 5:
                if not q.endswith('?'):
                    q += '?'
                questions.append(q)
    elif re.search(r'(?:^|\n)\s*\d+[\.\)]\s*.+', query_text, re.MULTILINE):
        numbered_pattern = r'(?:^|\n)\s*\d+[\.\)]\s*(.+?)(?=(?:\n\s*\d+[\.\)])|$)'
        numbered_matches = re.findall(numbered_pattern, query_text, re.MULTILINE | re.DOTALL)
        
        for match in numbered_matches:
            match = match.strip()
            if match and len(match) > 5:
                if not match.endswith('?') and '?' not in match:
                    match += '?'
                questions.append(match)
    
    if not questions:
        questions = [query_text]
    
    return questions

def enhanced_retrieval(db, query_text: str, k: int = 5):
    """Enhanced retrieval with better scoring and fallback options"""
    try:
        logger.info(f"Searching for: '{query_text}' with k={k}")
        
        results = []
        
        try:
            results = db.similarity_search_with_relevance_scores(query_text, k=k*3)
            logger.info(f"Similarity search returned {len(results)} results")
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            try:
                basic_results = db.similarity_search(query_text, k=k*2)
                results = [(doc, 0.5) for doc in basic_results]
                logger.info(f"Basic search returned {len(results)} results")
            except Exception as e2:
                logger.error(f"Basic search also failed: {e2}")
                return []
        
        if not results:
            logger.warning("No results found, trying broader search")
            words = query_text.split()
            if len(words) > 1:
                for word in words:
                    if len(word) > 3:
                        try:
                            word_results = db.similarity_search_with_relevance_scores(word, k=2)
                            results.extend(word_results)
                            if len(results) >= k:
                                break
                        except:
                            continue
        
        filtered_results = []
        seen_content = set()
        
        for doc, score in results:
            content_preview = doc.page_content[:100].strip()
            content_hash = hash(content_preview)
            
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            if score >= 0.1:
                filtered_results.append((doc, score))
                logger.info(f"Included result with score {score:.3f}: {content_preview[:50]}...")
        
        if not filtered_results and results:
            logger.warning("No results met threshold, including top results anyway")
            filtered_results = results[:k]
        
        final_results = filtered_results[:k]
        logger.info(f"Returning {len(final_results)} filtered results")
        return final_results
        
    except Exception as e:
        logger.error(f"Enhanced retrieval failed completely: {e}")
        return []

def create_enhanced_context(results, questions: list) -> tuple:
    """Create context that's optimized for multiple questions and return source info"""
    if not results:
        return "No relevant context found.", []
    
    context_parts = []
    source_info = []
    
    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get('source', 'Unknown')
        file_name = doc.metadata.get('file_name', os.path.basename(source) if source != 'Unknown' else 'Unknown')
        page = doc.metadata.get('page_number', '')
        page_info = f" (Page {page})" if page else ""
        
        display_source = file_name if file_name != 'Unknown' else source
        
        content = doc.page_content
        if len(content) > 500:
            content = content[:500] + "..."
        
        # Use document name as the identifier instead of SOURCE X
        context_part = f"[{display_source}{page_info}] (Relevance: {score:.2f}):\n{content}"
        context_parts.append(context_part)
        
        source_info.append({
            'id': i+1,
            'file_name': display_source,
            'page': page,
            'relevance': score,
            'full_path': source
        })
    
    context_text = "\n\n" + "\n\n".join(context_parts)
    return context_text, source_info

# Modified prompt template to include conversation history
ENHANCED_PROMPT_TEMPLATE = """You are a helpful assistant engaged in an ongoing conversation. Answer the current question using the provided context sources and conversation history.

IMPORTANT: When citing information, use the document name format shown in brackets, for example [RCW 10.01.240.pdf] or [RCW 10.01.240.pdf (Page 1)] - do NOT use generic SOURCE numbers.

CONVERSATION HISTORY:
{conversation_history}

CURRENT CONTEXT:
{context}

CURRENT QUESTION: {questions}

Please provide a helpful answer based on the context above and the conversation history. When you reference information, cite it using the document name in brackets as shown in the context (e.g., [document_name.pdf] or [document_name.pdf (Page X)]). 

If the user is asking a follow-up question or referring to something mentioned earlier in the conversation, acknowledge that context in your response.

RESPONSE:"""

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

app = FastAPI(title="RAG API", description="Retrieval-Augmented Generation API with Conversation Support", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "RAG API with Conversation Support is running"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    db_exists = os.path.exists(CHROMA_PATH)
    db_contents = []
    
    if db_exists:
        try:
            db_contents = os.listdir(CHROMA_PATH)
        except:
            db_contents = ["Error reading directory"]
    
    return {
        "status": "healthy", 
        "database_exists": db_exists,
        "database_path": CHROMA_PATH,
        "database_contents": db_contents,
        "api_key_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "api_base_configured": bool(os.environ.get("OPENAI_API_BASE")),
        "active_conversations": len(conversations)
    }

def call_openrouter_api(prompt: str, api_key: str, api_base: str) -> str:
    """Make API call to OpenRouter with improved error handling"""
    try:
        if api_base.endswith('/'):
            api_base = api_base.rstrip('/')
        
        if 'openrouter.ai' in api_base and not api_base.endswith('/api/v1'):
            api_base = "https://openrouter.ai/api/v1"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "RAG Application"
        }
        
        models_to_try = [
            "microsoft/phi-3-mini-128k-instruct:free",
            "meta-llama/llama-3.2-3b-instruct:free", 
            "google/gemma-2-9b-it:free",
            "mistralai/mistral-7b-instruct:free",
            "openchat/openchat-7b:free"
        ]
        
        logger.info(f"Trying {len(models_to_try)} models...")
        
        for i, model in enumerate(models_to_try):
            try:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1500,
                    "top_p": 0.9
                }
                
                logger.info(f"Attempting model {i+1}/{len(models_to_try)}: {model}")
                
                response = requests.post(
                    f"{api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=45
                )
                
                logger.info(f"Response status: {response.status_code}")
                
                content_type = response.headers.get('content-type', '')
                if 'text/html' in content_type:
                    logger.warning(f"Got HTML response for {model}")
                    if i == len(models_to_try) - 1:
                        raise HTTPException(status_code=500, detail="All models returned HTML errors")
                    continue
                
                if response.status_code != 200:
                    logger.warning(f"Model {model} failed with status {response.status_code}: {response.text[:200]}")
                    if i == len(models_to_try) - 1:
                        raise HTTPException(status_code=response.status_code, detail=f"All models failed. Last error: {response.text[:200]}")
                    continue
                
                try:
                    result = response.json()
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error for {model}: {e}")
                    if i == len(models_to_try) - 1:
                        raise HTTPException(status_code=500, detail="Invalid JSON response from all models")
                    continue
                
                if 'choices' not in result or not result['choices']:
                    logger.warning(f"No choices in response for {model}")
                    if i == len(models_to_try) - 1:
                        raise HTTPException(status_code=500, detail="No valid response from any model")
                    continue
                
                response_text = result['choices'][0]['message']['content']
                logger.info(f"Success with model {model}! Response length: {len(response_text)}")
                return response_text.strip()
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error for model {model}: {e}")
                if i == len(models_to_try) - 1:
                    raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error for model {model}: {e}")
                if i == len(models_to_try) - 1:
                    raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
                continue
        
        raise HTTPException(status_code=500, detail="All models failed unexpectedly")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API call failed completely: {e}")
        raise HTTPException(status_code=500, detail=f"API call failed: {str(e)}")

@app.post("/ask", response_model=QueryResponse)
def ask_question(query: Query):
    try:
        query_text = query.question.strip() if query.question else ""
        
        logger.info(f"Received query: '{query_text}'")
        
        if not query_text:
            return QueryResponse(
                response=None,
                error="Question cannot be empty",
                context_found=False,
                session_id=query.session_id or ""
            )
        
        # Get or create session
        session_id = get_or_create_session(query.session_id)
        
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        
        if not api_key:
            logger.error("OPENAI_API_KEY not found")
            return QueryResponse(
                response=None,
                error="OPENAI_API_KEY environment variable is required. Please set your OpenRouter API key.",
                context_found=False,
                session_id=session_id
            )
        
        if not os.path.exists(CHROMA_PATH):
            logger.error("Database not found")
            return QueryResponse(
                response=None,
                error="Vector database not found. Please run the document ingestion process first.",
                context_found=False,
                session_id=session_id
            )
        
        # Add user message to conversation history
        add_to_conversation(session_id, "user", query_text)
        
        questions = parse_multiple_questions(query_text)
        logger.info(f"Parsed {len(questions)} questions: {questions}")
        
        try:
            # CRITICAL: Use the same embedding model as ingestion script
            embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            
            # Test database connectivity
            test_results = db.similarity_search("test", k=1)
            logger.info(f"Database loaded successfully with {len(test_results)} test results")
            
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            return QueryResponse(
                response=None,
                error=f"Failed to load vector database: {str(e)}. Make sure you've run the ingestion script first.",
                context_found=False,
                session_id=session_id
            )
        
        combined_query = " ".join(questions)
        results = enhanced_retrieval(db, combined_query, k=5)
        
        logger.info(f"Retrieved {len(results)} results")
        
        if not results:
            logger.warning("No relevant documents found")
            response_text = "I couldn't find any relevant information in the documents to answer your question. Please try rephrasing your question or check if the documents contain information about this topic."
            
            # Add assistant response to conversation history
            add_to_conversation(session_id, "assistant", response_text)
            
            return QueryResponse(
                response=response_text,
                error=None,
                context_found=False,
                session_id=session_id
            )
        
        context_text, source_info = create_enhanced_context(results, questions)
        logger.info(f"Created context with {len(source_info)} sources")
        
        # Get conversation history
        conversation_history = get_conversation_context(session_id, max_messages=8)
        
        if len(questions) > 1:
            formatted_questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        else:
            formatted_questions = questions[0]
        
        formatted_prompt = ENHANCED_PROMPT_TEMPLATE.format(
            conversation_history=conversation_history if conversation_history else "No previous conversation.",
            context=context_text,
            questions=formatted_questions
        )
        
        logger.info(f"Prompt length: {len(formatted_prompt)} characters")
        
        try:
            response_text = call_openrouter_api(formatted_prompt, api_key, api_base)
            
            if not response_text:
                response_text = "I received an empty response. Please try again."
            else:
                # Add sources section if there are sources
                if source_info:
                    response_text += "\n\n**SOURCES:**\n"
                    for source in source_info:
                        page_info = f", Page {source['page']}" if source['page'] else ""
                        response_text += f"• {source['file_name']}{page_info} (Relevance: {source['relevance']:.2f})\n"
            
            # Add assistant response to conversation history
            add_to_conversation(session_id, "assistant", response_text, source_info)
            
            logger.info(f"Successfully generated response of length {len(response_text)}")
            
            return QueryResponse(
                response=response_text,
                error=None,
                context_found=True,
                sources=source_info,
                session_id=session_id
            )
            
        except HTTPException as he:
            logger.error(f"API call failed: {he.detail}")
            error_response = f"API Error: {he.detail}"
            add_to_conversation(session_id, "assistant", error_response)
            
            return QueryResponse(
                response=None,
                error=error_response,
                context_found=True,
                sources=source_info,
                session_id=session_id
            )
        
    except Exception as e:
        logger.error(f"Unexpected error in ask_question: {e}")
        return QueryResponse(
            response=None,
            error=f"An unexpected error occurred: {str(e)}",
            context_found=False,
            session_id=query.session_id or ""
        )

@app.get("/conversation/{session_id}", response_model=ConversationHistory)
def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conv_data = conversations[session_id]
    return ConversationHistory(
        session_id=session_id,
        messages=conv_data['messages'],
        created_at=conv_data['created_at'].isoformat(),
        last_updated=conv_data['last_updated'].isoformat()
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
                "message_count": len(data['messages']),
                "created_at": data['created_at'].isoformat(),
                "last_updated": data['last_updated'].isoformat()
            }
            for session_id, data in conversations.items()
        ]
    }

@app.post("/new-conversation")
def start_new_conversation():
    """Start a new conversation session"""
    session_id = get_or_create_session()
    return {"session_id": session_id, "message": "New conversation started"}

@app.get("/test-api")
def test_api():
    """Test endpoint to check API connectivity"""
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}
    
    try:
        response_text = call_openrouter_api("Hello, this is a test. Please respond with 'Test successful!'", api_key, api_base)
        return {
            "success": True, 
            "response": response_text,
            "api_key_prefix": f"{api_key[:8]}...",
            "api_base": api_base
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/debug-db")
def debug_database():
    """Debug endpoint to check database status"""
    if not os.path.exists(CHROMA_PATH):
        return {"error": "Database folder does not exist", "path": CHROMA_PATH}
    
    try:
        # Check database folder contents
        db_contents = os.listdir(CHROMA_PATH)
        
        # Try to load the database
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Test different search queries
        test_queries = ["test", "document", "content", "information"]
        search_results = {}
        
        for query in test_queries:
            try:
                results = db.similarity_search(query, k=3)
                search_results[query] = {
                    "count": len(results),
                    "previews": [
                        {
                            "content": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                            "metadata": doc.metadata
                        } for doc in results[:2]  # Show first 2 results
                    ]
                }
            except Exception as e:
                search_results[query] = {"error": str(e)}
        
        return {
            "database_exists": True,
            "database_path": CHROMA_PATH,
            "database_contents": db_contents,
            "search_tests": search_results,
            "status": "Database appears to be working" if any(r.get("count", 0) > 0 for r in search_results.values()) else "Database exists but no search results found"
        }
        
    except Exception as e:
        return {
            "error": f"Database test failed: {str(e)}", 
            "path": CHROMA_PATH,
            "suggestion": "Try running the ingestion script to recreate the database"
        }

@app.get("/sources")
def get_sources_info():
    """Get information about available sources in the database"""
    if not os.path.exists(CHROMA_PATH):
        return {"error": "Database not found", "path": CHROMA_PATH}
    
    try:
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        sample_docs = db.similarity_search("document content", k=50)
        
        sources = {}
        total_docs = len(sample_docs)
        
        for doc in sample_docs:
            source = doc.metadata.get('source', 'Unknown')
            file_name = doc.metadata.get('file_name', os.path.basename(source) if source != 'Unknown' else 'Unknown')
            page = doc.metadata.get('page_number', 'N/A')
            
            if file_name not in sources:
                sources[file_name] = {
                    'full_path': source,
                    'pages': set(),
                    'chunks': 0,
                    'sample_content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
            
            if page != 'N/A':
                sources[file_name]['pages'].add(str(page))
            sources[file_name]['chunks'] += 1
        
        for source_info in sources.values():
            source_info['pages'] = sorted(source_info['pages'], key=lambda x: int(x) if x.isdigit() else 0)
        
        return {
            "total_documents_sampled": total_docs,
            "total_sources": len(sources),
            "sources": sources
        }
    
    except Exception as e:
        return {"error": f"Failed to analyze sources: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
