#app.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import requests
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PATH = os.path.join(os.getcwd(), "my_chroma_db")
logger.info(f"Checking CHROMA_PATH: {CHROMA_PATH}")
if os.path.exists(CHROMA_PATH):
    logger.info(f"Database contents: {os.listdir(CHROMA_PATH)}")
else:
    logger.warning("Database folder not found!")

# Helper functions for enhanced query processing
def parse_multiple_questions(query_text: str) -> list:
    """Parse multiple questions from a single input"""
    if not query_text or not query_text.strip():
        return [""]
    
    questions = []
    query_text = query_text.strip()
    
    # Count actual question marks to determine if it's truly multiple questions
    question_marks = query_text.count('?')
    
    # Only try numbered pattern if there are multiple question marks OR clear numbered format
    if question_marks > 1:
        # Split by question marks and filter
        potential_questions = query_text.split('?')
        for q in potential_questions:
            q = q.strip()
            if q and len(q) > 5:  # Reduced minimum length
                if not q.endswith('?'):
                    q += '?'
                questions.append(q)
    elif re.search(r'(?:^|\n)\s*\d+[\.\)]\s*.+', query_text, re.MULTILINE):
        # Numbered questions pattern
        numbered_pattern = r'(?:^|\n)\s*\d+[\.\)]\s*(.+?)(?=(?:\n\s*\d+[\.\)])|$)'
        numbered_matches = re.findall(numbered_pattern, query_text, re.MULTILINE | re.DOTALL)
        
        for match in numbered_matches:
            match = match.strip()
            if match and len(match) > 5:
                if not match.endswith('?') and '?' not in match:
                    match += '?'
                questions.append(match)
    
    # If no clear multiple questions found, treat as single question
    if not questions:
        questions = [query_text]
    
    return questions

def enhanced_retrieval(db, query_text: str, k: int = 5):
    """Enhanced retrieval with better scoring and fallback options"""
    try:
        logger.info(f"Searching for: '{query_text}' with k={k}")
        
        # Try multiple retrieval strategies
        results = []
        
        # Strategy 1: Similarity search with scores
        try:
            results = db.similarity_search_with_relevance_scores(query_text, k=k*3)
            logger.info(f"Similarity search returned {len(results)} results")
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            # Fallback to basic similarity search
            try:
                basic_results = db.similarity_search(query_text, k=k*2)
                results = [(doc, 0.5) for doc in basic_results]  # Assign default score
                logger.info(f"Basic search returned {len(results)} results")
            except Exception as e2:
                logger.error(f"Basic search also failed: {e2}")
                return []
        
        if not results:
            logger.warning("No results found, trying broader search")
            # Try with individual words if no results
            words = query_text.split()
            if len(words) > 1:
                for word in words:
                    if len(word) > 3:  # Skip short words
                        try:
                            word_results = db.similarity_search_with_relevance_scores(word, k=2)
                            results.extend(word_results)
                            if len(results) >= k:
                                break
                        except:
                            continue
        
        # Filter and deduplicate results
        filtered_results = []
        seen_content = set()
        
        for doc, score in results:
            # Create a hash of the first 100 characters to avoid duplicates
            content_preview = doc.page_content[:100].strip()
            content_hash = hash(content_preview)
            
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            # Much lower threshold to be more inclusive
            if score >= 0.1:  # Reduced from 0.3
                filtered_results.append((doc, score))
                logger.info(f"Included result with score {score:.3f}: {content_preview[:50]}...")
        
        # If still no results, include some anyway with very low threshold
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
        # Add source information for better reference
        source = doc.metadata.get('source', 'Unknown')
        file_name = doc.metadata.get('file_name', os.path.basename(source) if source != 'Unknown' else 'Unknown')
        page = doc.metadata.get('page_number', '')
        page_info = f" (Page {page})" if page else ""
        
        # Clean up source path for display
        display_source = file_name if file_name != 'Unknown' else source
        
        # Limit content length to avoid token limits
        content = doc.page_content
        if len(content) > 500:
            content = content[:500] + "..."
        
        context_part = f"[SOURCE {i+1}] {display_source}{page_info} (Relevance: {score:.2f}):\n{content}"
        context_parts.append(context_part)
        
        # Store source information for citation
        source_info.append({
            'id': i+1,
            'file_name': display_source,
            'page': page,
            'relevance': score,
            'full_path': source
        })
    
    context_text = "\n\n" + "\n\n".join(context_parts)
    return context_text, source_info

# Simplified prompt template
ENHANCED_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using the provided context sources. Cite your sources using [SOURCE X] format.

CONTEXT:
{context}

QUESTION: {questions}

Please provide a helpful answer based on the context above. If you reference information, include the source citation like [SOURCE 1].

RESPONSE:"""

class Query(BaseModel):
    question: str

class QueryResponse(BaseModel):
    response: Optional[str] = None
    error: Optional[str] = None
    context_found: bool = False
    sources: Optional[list] = None

# Create FastAPI app
app = FastAPI(title="RAG API", description="Retrieval-Augmented Generation API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "RAG API is running"}

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
        "api_base_configured": bool(os.environ.get("OPENAI_API_BASE"))
    }

def call_openrouter_api(prompt: str, api_key: str, api_base: str) -> str:
    """Make API call to OpenRouter with improved error handling"""
    try:
        # Clean the API base URL
        if api_base.endswith('/'):
            api_base = api_base.rstrip('/')
        
        # Ensure correct OpenRouter URL
        if 'openrouter.ai' in api_base and not api_base.endswith('/api/v1'):
            api_base = "https://openrouter.ai/api/v1"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "RAG Application"
        }
        
        # Updated model list with more reliable options
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
                
                # Check for HTML error pages
                content_type = response.headers.get('content-type', '')
                if 'text/html' in content_type:
                    logger.warning(f"Got HTML response for {model}")
                    if i == len(models_to_try) - 1:  # Last model
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
                
                # Success!
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
        
        # Shouldn't reach here
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
                context_found=False
            )
        
        # Check environment variables first
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        
        if not api_key:
            logger.error("OPENAI_API_KEY not found")
            return QueryResponse(
                response=None,
                error="OPENAI_API_KEY environment variable is required. Please set your OpenRouter API key.",
                context_found=False
            )
        
        # Check if database exists
        if not os.path.exists(CHROMA_PATH):
            logger.error("Database not found")
            return QueryResponse(
                response=None,
                error="Vector database not found. Please run the document ingestion process first.",
                context_found=False
            )
        
        # Parse questions
        questions = parse_multiple_questions(query_text)
        logger.info(f"Parsed {len(questions)} questions: {questions}")
        
        # Initialize embedding function and database
        try:
            embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            logger.info("Database loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            return QueryResponse(
                response=None,
                error=f"Failed to load vector database: {str(e)}",
                context_found=False
            )
        
        # Retrieve relevant documents
        combined_query = " ".join(questions)
        results = enhanced_retrieval(db, combined_query, k=5)
        
        logger.info(f"Retrieved {len(results)} results")
        
        if not results:
            logger.warning("No relevant documents found")
            return QueryResponse(
                response="I couldn't find any relevant information in the documents to answer your question. Please try rephrasing your question or check if the documents contain information about this topic.",
                error=None,
                context_found=False
            )
        
        # Create context
        context_text, source_info = create_enhanced_context(results, questions)
        logger.info(f"Created context with {len(source_info)} sources")
        
        # Format prompt
        if len(questions) > 1:
            formatted_questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        else:
            formatted_questions = questions[0]
        
        formatted_prompt = ENHANCED_PROMPT_TEMPLATE.format(
            context=context_text,
            questions=formatted_questions
        )
        
        logger.info(f"Prompt length: {len(formatted_prompt)} characters")
        
        # Make API call
        try:
            response_text = call_openrouter_api(formatted_prompt, api_key, api_base)
            
            if not response_text:
                return QueryResponse(
                    response="I received an empty response. Please try again.",
                    error=None,
                    context_found=True,
                    sources=source_info
                )
            
            # Add source information
            if source_info:
                response_text += "\n\n**SOURCES:**\n"
                for source in source_info:
                    page_info = f", Page {source['page']}" if source['page'] else ""
                    response_text += f"[SOURCE {source['id']}] {source['file_name']}{page_info} (Relevance: {source['relevance']:.2f})\n"
            
            logger.info(f"Successfully generated response of length {len(response_text)}")
            
            return QueryResponse(
                response=response_text,
                error=None,
                context_found=True,
                sources=source_info
            )
            
        except HTTPException as he:
            logger.error(f"API call failed: {he.detail}")
            return QueryResponse(
                response=None,
                error=f"API Error: {he.detail}",
                context_found=True,
                sources=source_info
            )
        
    except Exception as e:
        logger.error(f"Unexpected error in ask_question: {e}")
        return QueryResponse(
            response=None,
            error=f"An unexpected error occurred: {str(e)}",
            context_found=False
        )

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
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Test a simple search
        test_results = db.similarity_search("test", k=3)
        
        return {
            "database_exists": True,
            "database_path": CHROMA_PATH,
            "database_contents": os.listdir(CHROMA_PATH),
            "test_search_results": len(test_results),
            "sample_documents": [
                {
                    "content_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                    "metadata": doc.metadata
                } for doc in test_results
            ]
        }
    except Exception as e:
        return {"error": f"Database test failed: {str(e)}", "path": CHROMA_PATH}

@app.get("/sources")
def get_sources_info():
    """Get information about available sources in the database"""
    if not os.path.exists(CHROMA_PATH):
        return {"error": "Database not found", "path": CHROMA_PATH}
    
    try:
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Get more documents to analyze sources
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
        
        # Convert sets to sorted lists for JSON
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
    uvicorn.run(app, host="0.0.0.0", port=port):
                if not match.endswith('?') and '?' not in match:
                    match += '?'
                questions.append(match)
    
    # If no clear multiple questions found, treat as single question
    if not questions:
        questions = [query_text]
    
    return questions

def enhanced_retrieval(db, query_text: str, k: int = 5):
    """Enhanced retrieval with better scoring and fallback options"""
    try:
        logger.info(f"Searching for: '{query_text}' with k={k}")
        
        # Try multiple retrieval strategies
        results = []
        
        # Strategy 1: Similarity search with scores
        try:
            results = db.similarity_search_with_relevance_scores(query_text, k=k*3)
            logger.info(f"Similarity search returned {len(results)} results")
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            # Fallback to basic similarity search
            try:
                basic_results = db.similarity_search(query_text, k=k*2)
                results = [(doc, 0.5) for doc in basic_results]  # Assign default score
                logger.info(f"Basic search returned {len(results)} results")
            except Exception as e2:
                logger.error(f"Basic search also failed: {e2}")
                return []
        
        if not results:
            logger.warning("No results found, trying broader search")
            # Try with individual words if no results
            words = query_text.split()
            if len(words) > 1:
                for word in words:
                    if len(word) > 3:  # Skip short words
                        try:
                            word_results = db.similarity_search_with_relevance_scores(word, k=2)
                            results.extend(word_results)
                            if len(results) >= k:
                                break
                        except:
                            continue
        
        # Filter and deduplicate results
        filtered_results = []
        seen_content = set()
        
        for doc, score in results:
            # Create a hash of the first 100 characters to avoid duplicates
            content_preview = doc.page_content[:100].strip()
            content_hash = hash(content_preview)
            
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            # Much lower threshold to be more inclusive
            if score >= 0.1:  # Reduced from 0.3
                filtered_results.append((doc, score))
                logger.info(f"Included result with score {score:.3f}: {content_preview[:50]}...")
        
        # If still no results, include some anyway with very low threshold
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
        # Add source information for better reference
        source = doc.metadata.get('source', 'Unknown')
        file_name = doc.metadata.get('file_name', os.path.basename(source) if source != 'Unknown' else 'Unknown')
        page = doc.metadata.get('page_number', '')
        page_info = f" (Page {page})" if page else ""
        
        # Clean up source path for display
        display_source = file_name if file_name != 'Unknown' else source
        
        # Limit content length to avoid token limits
        content = doc.page_content
        if len(content) > 500:
            content = content[:500] + "..."
        
        context_part = f"[SOURCE {i+1}] {display_source}{page_info} (Relevance: {score:.2f}):\n{content}"
        context_parts.append(context_part)
        
        # Store source information for citation
        source_info.append({
            'id': i+1,
            'file_name': display_source,
            'page': page,
            'relevance': score,
            'full_path': source
        })
    
    context_text = "\n\n" + "\n\n".join(context_parts)
    return context_text, source_info

# Simplified prompt template
ENHANCED_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using the provided context sources. Cite your sources using [SOURCE X] format.

CONTEXT:
{context}

QUESTION: {questions}

Please provide a helpful answer based on the context above. If you reference information, include the source citation like [SOURCE 1].

RESPONSE:"""

class Query(BaseModel):
    question: str

class QueryResponse(BaseModel):
    response: Optional[str] = None
    error: Optional[str] = None
    context_found: bool = False
    sources: Optional[list] = None

# Create FastAPI app
app = FastAPI(title="RAG API", description="Retrieval-Augmented Generation API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "RAG API is running"}

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
        "api_base_configured": bool(os.environ.get("OPENAI_API_BASE"))
    }

def call_openrouter_api(prompt: str, api_key: str, api_base: str) -> str:
    """Make API call to OpenRouter with improved error handling"""
    try:
        # Clean the API base URL
        if api_base.endswith('/'):
            api_base = api_base.rstrip('/')
        
        # Ensure correct OpenRouter URL
        if 'openrouter.ai' in api_base and not api_base.endswith('/api/v1'):
            api_base = "https://openrouter.ai/api/v1"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "RAG Application"
        }
        
        # Updated model list with more reliable options
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
                
                # Check for HTML error pages
                content_type = response.headers.get('content-type', '')
                if 'text/html' in content_type:
                    logger.warning(f"Got HTML response for {model}")
                    if i == len(models_to_try) - 1:  # Last model
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
                
                # Success!
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
        
        # Shouldn't reach here
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
                context_found=False
            )
        
        # Check environment variables first
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        
        if not api_key:
            logger.error("OPENAI_API_KEY not found")
            return QueryResponse(
                response=None,
                error="OPENAI_API_KEY environment variable is required. Please set your OpenRouter API key.",
                context_found=False
            )
        
        # Check if database exists
        if not os.path.exists(CHROMA_PATH):
            logger.error("Database not found")
            return QueryResponse(
                response=None,
                error="Vector database not found. Please run the document ingestion process first.",
                context_found=False
            )
        
        # Parse questions
        questions = parse_multiple_questions(query_text)
        logger.info(f"Parsed {len(questions)} questions: {questions}")
        
        # Initialize embedding function and database
        try:
            embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            logger.info("Database loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            return QueryResponse(
                response=None,
                error=f"Failed to load vector database: {str(e)}",
                context_found=False
            )
        
        # Retrieve relevant documents
        combined_query = " ".join(questions)
        results = enhanced_retrieval(db, combined_query, k=5)
        
        logger.info(f"Retrieved {len(results)} results")
        
        if not results:
            logger.warning("No relevant documents found")
            return QueryResponse(
                response="I couldn't find any relevant information in the documents to answer your question. Please try rephrasing your question or check if the documents contain information about this topic.",
                error=None,
                context_found=False
            )
        
        # Create context
        context_text, source_info = create_enhanced_context(results, questions)
        logger.info(f"Created context with {len(source_info)} sources")
        
        # Format prompt
        if len(questions) > 1:
            formatted_questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        else:
            formatted_questions = questions[0]
        
        formatted_prompt = ENHANCED_PROMPT_TEMPLATE.format(
            context=context_text,
            questions=formatted_questions
        )
        
        logger.info(f"Prompt length: {len(formatted_prompt)} characters")
        
        # Make API call
        try:
            response_text = call_openrouter_api(formatted_prompt, api_key, api_base)
            
            if not response_text:
                return QueryResponse(
                    response="I received an empty response. Please try again.",
                    error=None,
                    context_found=True,
                    sources=source_info
                )
            
            # Add source information
            if source_info:
                response_text += "\n\n**SOURCES:**\n"
                for source in source_info:
                    page_info = f", Page {source['page']}" if source['page'] else ""
                    response_text += f"[SOURCE {source['id']}] {source['file_name']}{page_info} (Relevance: {source['relevance']:.2f})\n"
            
            logger.info(f"Successfully generated response of length {len(response_text)}")
            
            return QueryResponse(
                response=response_text,
                error=None,
                context_found=True,
                sources=source_info
            )
            
        except HTTPException as he:
            logger.error(f"API call failed: {he.detail}")
            return QueryResponse(
                response=None,
                error=f"API Error: {he.detail}",
                context_found=True,
                sources=source_info
            )
        
    except Exception as e:
        logger.error(f"Unexpected error in ask_question: {e}")
        return QueryResponse(
            response=None,
            error=f"An unexpected error occurred: {str(e)}",
            context_found=False
        )

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
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Test a simple search
        test_results = db.similarity_search("test", k=3)
        
        return {
            "database_exists": True,
            "database_path": CHROMA_PATH,
            "database_contents": os.listdir(CHROMA_PATH),
            "test_search_results": len(test_results),
            "sample_documents": [
                {
                    "content_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                    "metadata": doc.metadata
                } for doc in test_results
            ]
        }
    except Exception as e:
        return {"error": f"Database test failed: {str(e)}", "path": CHROMA_PATH}

@app.get("/sources")
def get_sources_info():
    """Get information about available sources in the database"""
    if not os.path.exists(CHROMA_PATH):
        return {"error": "Database not found", "path": CHROMA_PATH}
    
    try:
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Get more documents to analyze sources
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
        
        # Convert sets to sorted lists for JSON
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
        
        context_part = f"[SOURCE {i+1}] {display_source}{page_info} (Relevance: {score:.2f}):\n{doc.page_content}"
        context_parts.append(context_part)
        
        # Store source information for citation
        source_info.append({
            'id': i+1,
            'file_name': display_source,
            'page': page,
            'relevance': score,
            'full_path': source
        })
    
    context_text = "\n\n" + "="*50 + "\n\n".join(context_parts)
    return context_text, source_info

ENHANCED_PROMPT_TEMPLATE = """
You are a legal expert. Answer the question below using ONLY the provided context sources. You MUST cite sources using the exact format [SOURCE X] where X matches the source number from the context.

CONTEXT:
{context}

QUESTION: {questions}

CRITICAL CITATION RULES:
- Every factual claim MUST include a source citation like [SOURCE 1] or [SOURCE 2]
- Use the exact [SOURCE X] format - do not write "RCW 10.01.160" or document names
- When referencing specific statutes or rules, include the source citation immediately after
- Example: "According to the statute [SOURCE 1], costs may be imposed..."
- Example: "The definition of indigent [SOURCE 2] includes compelling circumstances..."

GUIDELINES:
- Provide a direct, comprehensive answer based solely on the provided sources
- Use professional legal terminology and maintain a formal tone
- Organize information clearly with bullet points when appropriate
- If information is insufficient, state what additional sources would be needed
- Do not reference external cases or general legal knowledge unless specifically mentioned in the sources
- Keep responses focused and avoid unnecessary repetition

RESPONSE:
"""

class Query(BaseModel):
    question: str

class QueryResponse(BaseModel):
    response: Optional[str] = None
    error: Optional[str] = None
    context_found: bool = False
    sources: Optional[list] = None

# Create FastAPI app instance ONCE
app = FastAPI(title="RAG API", description="Retrieval-Augmented Generation API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:3000"] for stricter
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "RAG API is running"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "database_exists": os.path.exists(CHROMA_PATH),
        "database_path": CHROMA_PATH
    }

def call_openrouter_api(prompt: str, api_key: str, api_base: str) -> str:
    """Make API call to OpenRouter with proper error handling"""
    try:
        # Clean the API base URL
        if api_base.endswith('/'):
            api_base = api_base.rstrip('/')
        
        # Ensure we're using the correct OpenRouter URL
        if 'openrouter.ai' in api_base and not api_base.endswith('/api/v1'):
            api_base = "https://openrouter.ai/api/v1"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",  # Required for OpenRouter
            "X-Title": "RAG Application"  # Optional: for OpenRouter
        }
        
        # Try different model names that are known to work
        models_to_try = [
            "meta-llama/llama-3.2-3b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free",
            "google/gemma-2-9b-it:free",
            "deepseek/deepseek-chat"
        ]
        
        for model in models_to_try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 1000  # Increased from 500 to prevent cutoff
            }
            
            print(f"Trying model: {model}")
            print(f"Making request to: {api_base}/chat/completions")
            print(f"API key starts with: {api_key[:10]}...")
            
            response = requests.post(
                f"{api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            # If we get HTML, it's likely an error page - let's see what it says
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                print(f"Got HTML response for model {model}")
                print(f"Response content (first 1000 chars): {response.text[:1000]}")
                
                # If it's the last model to try, raise the error
                if model == models_to_try[-1]:
                    # Try to extract error message from HTML
                    error_msg = "HTML response received instead of JSON"
                    if "error" in response.text.lower():
                        # Simple extraction - look for common error patterns
                        import re
                        error_pattern = r'error["\s:]*([^"<>\n]{10,100})'
                        match = re.search(error_pattern, response.text, re.IGNORECASE)
                        if match:
                            error_msg = f"API Error: {match.group(1).strip()}"
                    
                    raise HTTPException(
                        status_code=500,
                        detail=f"{error_msg}. Status: {response.status_code}"
                    )
                continue
            
            if response.status_code != 200:
                print(f"Error response for model {model}: {response.text}")
                if model == models_to_try[-1]:  # Last model to try
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"All models failed. Last error: {response.text}"
                    )
                continue
            
            # Check if response is JSON
            if 'application/json' not in content_type:
                print(f"Unexpected content type for model {model}: {content_type}")
                if model == models_to_try[-1]:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Expected JSON response, got {content_type}"
                    )
                continue
            
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                print(f"JSON decode error for model {model}: {e}")
                if model == models_to_try[-1]:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Invalid JSON response: {str(e)}"
                    )
                continue
            
            if 'choices' not in result or not result['choices']:
                print(f"No choices in response for model {model}")
                if model == models_to_try[-1]:
                    raise HTTPException(
                        status_code=500,
                        detail="No choices in API response"
                    )
                continue
            
            # Success! Return the response
            response_text = result['choices'][0]['message']['content']
            print(f"Success with model {model}: {response_text[:100]}...")
            return response_text
        
        # This shouldn't be reached, but just in case
        raise HTTPException(
            status_code=500,
            detail="All models failed to generate a response"
        )
        
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="API request timed out")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/ask", response_model=QueryResponse)
def ask_question(query: Query):
    try:
        query_text = query.question
        
        if not query_text or query_text.strip() == "":
            return QueryResponse(
                response=None,
                error="Question cannot be empty",
                context_found=False
            )
        
        # Parse multiple questions
        questions = parse_multiple_questions(query_text)
        print(f"Parsed {len(questions)} questions: {questions}")
        
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Check if Chroma database exists
        if not os.path.exists(CHROMA_PATH):
            return QueryResponse(
                response=None,
                error="Database not found",
                context_found=False
            )
        
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Combine all questions for retrieval
        combined_query = " ".join(questions)
        results = enhanced_retrieval(db, combined_query, k=5)
        
        if not results:
            return QueryResponse(
                response="No relevant information found in the documents.",
                error=None,
                context_found=False
            )
        
        # Create enhanced context with source tracking
        context_text, source_info = create_enhanced_context(results, questions)
        
        # Format questions for the prompt
        if len(questions) > 1:
            formatted_questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
            # Use multi-question template with stronger source citation requirements
            multi_question_template = """
You are a legal expert. Answer the questions below using ONLY the provided context sources. You MUST cite sources using the exact format [SOURCE X] where X matches the source number from the context.

CONTEXT:
{context}

QUESTIONS: 
{questions}

CRITICAL CITATION RULES:
- Every factual claim MUST include a source citation like [SOURCE 1] or [SOURCE 2]
- Use the exact [SOURCE X] format - do not write statute names or document names without citations
- When referencing specific statutes or rules, include the source citation immediately after
- Example: "According to the statute [SOURCE 1], costs may be imposed..."

GUIDELINES:
- Answer each question separately and clearly
- Number your responses to match the question numbers
- Provide comprehensive answers based solely on the provided sources
- Use professional legal terminology and maintain a formal tone
- If information is insufficient for any question, state what additional sources would be needed
- Organize information clearly with bullet points when appropriate

RESPONSE:
"""
            formatted_prompt = multi_question_template.format(
                context=context_text,
                questions=formatted_questions
            )
        else:
            # Single question - use simpler template
            formatted_questions = questions[0]
            formatted_prompt = ENHANCED_PROMPT_TEMPLATE.format(
                context=context_text,
                questions=formatted_questions
            )
        
        # Debug: Print the formatted prompt type and length
        print(f"Formatted prompt type: {type(formatted_prompt)}")
        print(f"Formatted prompt length: {len(formatted_prompt)}")
        
        # Check for required environment variables
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        
        if not api_key:
            return QueryResponse(
                response=None,
                error="OPENAI_API_KEY environment variable is required",
                context_found=True
            )
        
        if not api_base:
            return QueryResponse(
                response=None,
                error="OPENAI_API_BASE environment variable is required",
                context_found=True
            )
        
        # Make the API call using requests
        try:
            response_text = call_openrouter_api(formatted_prompt, api_key, api_base)
            
            # Append source information to the response
            if response_text and source_info:
                response_text += "\n\n**SOURCES:**\n"
                for source in source_info:
                    page_info = f", Page {source['page']}" if source['page'] else ""
                    response_text += f"[SOURCE {source['id']}] {source['file_name']}{page_info} (Relevance: {source['relevance']:.2f})\n"
            
            print(f"Successfully got response: {response_text[:100]}...")
            
            return QueryResponse(
                response=response_text if response_text else "No response generated",
                error=None,
                context_found=True,
                sources=source_info
            )
            
        except HTTPException as he:
            return QueryResponse(
                response=None,
                error=str(he.detail),
                context_found=True,
                sources=source_info
            )
        except Exception as e:
            print(f"API call failed: {str(e)}")
            print(f"Error type: {type(e)}")
            return QueryResponse(
                response=None,
                error=f"API call failed: {str(e)}",
                context_found=True,
                sources=source_info
            )
        
    except Exception as e:
        return QueryResponse(
            response=None,
            error=f"An error occurred: {str(e)}",
            context_found=False
        )

@app.get("/test-api")
def test_api():
    """Test endpoint to check API connectivity"""
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}
    
    try:
        response_text = call_openrouter_api("Hello, this is a test.", api_key, api_base)
        return {"success": True, "response": response_text}
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug-api")
def debug_api():
    """Debug endpoint to check API configuration"""
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    
    # Make a simple request to see what we get
    headers = {
        "Authorization": f"Bearer {api_key}" if api_key else "Bearer MISSING",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "RAG Application Debug"
    }
    
    try:
        response = requests.get(
            f"{api_base}/models",  # Get available models
            headers=headers,
            timeout=10
        )
        
        return {
            "api_key_set": bool(api_key),
            "api_key_prefix": api_key[:10] + "..." if api_key else "Not set",
            "api_base": api_base,
            "models_endpoint_status": response.status_code,
            "models_content_type": response.headers.get('content-type'),
            "models_response_preview": response.text[:500] if response.text else "No content"
        }
    except Exception as e:
        return {
            "api_key_set": bool(api_key),
            "api_base": api_base,
            "error": str(e)
        }

@app.get("/sources")
def get_sources_info():
    """Get information about available sources in the database"""
    if not os.path.exists(CHROMA_PATH):
        return {"error": "Database not found"}
    
    try:
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Get a sample of documents to show available sources
        sample_docs = db.similarity_search("sample query", k=20)
        
        sources = {}
        for doc in sample_docs:
            source = doc.metadata.get('source', 'Unknown')
            file_name = doc.metadata.get('file_name', os.path.basename(source) if source != 'Unknown' else 'Unknown')
            page = doc.metadata.get('page_number', 'N/A')
            
            if file_name not in sources:
                sources[file_name] = {
                    'full_path': source,
                    'pages': set(),
                    'chunks': 0
                }
            
            if page != 'N/A':
                sources[file_name]['pages'].add(str(page))
            sources[file_name]['chunks'] += 1
        
        # Convert sets to sorted lists for JSON serialization
        for source_info in sources.values():
            source_info['pages'] = sorted(source_info['pages'], key=lambda x: int(x) if x.isdigit() else 0)
        
        return {
            "total_sources": len(sources),
            "sources": sources
        }
    
    except Exception as e:
        return {"error": f"Failed to get sources info: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Use Render's port or fallback to 8000 locally
    uvicorn.run(app, host="0.0.0.0", port=port)
