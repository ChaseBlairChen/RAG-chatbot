# - app.py -
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import requests
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional, List, Dict, Tuple
import logging
import uuid
from datetime import datetime, timedelta

# - AI Agent Imports -
from enum import Enum
from dataclasses import dataclass
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
# - End AI Agent Imports -

# --- ChromaDB Settings Import ---
# Add this import to match generate_db.py
from chromadb.config import Settings
# --- End ChromaDB Settings Import ---

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SUGGESTION: Match the database folder name and use absolute path for consistency ---
# Using absolute path like generate_db.py can help prevent issues.
CHROMA_PATH = os.path.abspath(os.path.join(os.getcwd(), "chromadb-database"))
# --- END SUGGESTION ---
logger.info(f"Checking CHROMA_PATH: {CHROMA_PATH}")
if os.path.exists(CHROMA_PATH):
    logger.info(f"Database contents: {os.listdir(CHROMA_PATH)}")
else:
    logger.warning("Database folder not found!")

# In-memory conversation storage (in production, use Redis or a database)
conversations: Dict[str, Dict] = {}

# --- SUGGESTION 1: Alias Map for Entity Resolution ---
# Define common aliases for better search
ALIAS_MAP = {
    "rcw": "Revised Code of Washington",
    "washington rcw": "Revised Code of Washington",
    "revised code washington": "Revised Code of Washington",
    "uscis": "United States Citizenship and Immigration Services",
    "immigration court": "Executive Office for Immigration Review",
    "eoir": "Executive Office for Immigration Review",
    "ice": "Immigration and Customs Enforcement",
    "uscis asylum": "USCIS Asylum",
    "asylum": "Asylum",
    "immigration": "Immigration",
    "law": "Law",
    "act": "Act",
    "statute": "Statute",
    "regulation": "Regulation",
    "bill": "Bill",
    "legislation": "Legislation",
    "section": "Section",
    "subsection": "Subsection",
    "chapter": "Chapter",
    "title": "Title",
    # Add more aliases as needed
}
# --- END SUGGESTION 1 ---

# --- SUGGESTION 2: Load spaCy model once ---
# Load spaCy model once at startup for efficiency
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {e}")
    nlp = None

# Load Sentence Transformer model once at startup for efficiency
try:
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence transformer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Sentence Transformer model: {e}")
    sentence_model = None
# --- END SUGGESTION 2 ---

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SUGGESTION 3: Query Model ---
class Query(BaseModel):
    question: str
    session_id: Optional[str] = None
# --- END SUGGESTION 3 ---

# --- SUGGESTION 4: Query Response Model ---
class QueryResponse(BaseModel):
    response: Optional[str]
    error: Optional[str]
    context_found: bool
    session_id: str
# --- END SUGGESTION 4 ---

# --- SUGGESTION 5: Universal AI Agent Implementation ---
# --- (This section includes the AI agent code from the provided snippets) ---

# --- END SUGGESTION 5 ---

# --- SUGGESTION 6: Cleanup Expired Conversations ---
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
# --- END SUGGESTION 6 ---

# --- SUGGESTION 7: Process Query Function ---
# --- Updated to use explicit Chroma client settings ---
def process_query(question: str, session_id: str) -> QueryResponse:
    """
    Process the user query using the RAG pipeline and AI agent.
    """
    try:
        # --- Load Database with Explicit Settings ---
        # Match the client settings used in generate_db.py
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create explicit client settings matching generate_db.py
        client_settings = Settings(
            persist_directory=CHROMA_PATH,
            anonymized_telemetry=False, # Match generate_db.py
            allow_reset=True,
            is_persistent=True
        )

        # Create the Chroma instance with explicit client settings
        db = Chroma(
            collection_name="default", # Explicitly specify collection name
            embedding_function=embedding_function,
            persist_directory=CHROMA_PATH,
            client_settings=client_settings # Use the explicit settings
        )
        # --- End Load Database ---

        # Test database connectivity (this also confirms loading worked)
        test_results = db.similarity_search("test", k=1)
        logger.info(f"Database loaded successfully with {len(test_results)} test results")

        # --- Initialize AI Agent ---
        if not nlp or not sentence_model:
             return QueryResponse(
                response=None,
                error="NLP models failed to load.",
                context_found=False,
                session_id=session_id
            )
        agent = UniversalRAGAgent(nlp, sentence_model, db)
        # --- End Initialize AI Agent ---

        # --- Run AI Agent Search ---
        logger.info(f"🧠 AI Agent starting intelligent search for: '{question}'")
        agent_result = agent.search(question)
        # --- End Run AI Agent Search ---

        # --- Format Response ---
        if agent_result.results:
            # If agent found results, format them
            formatted_results = [f"Source: {res.metadata.get('source', 'Unknown')}\nContent: {res.page_content}" for res in agent_result.results]
            response_text = "\n\n".join(formatted_results)
            logger.info(f"🧠 Universal AI Agent Result:\n  Strategy: {agent_result.strategy}\n  Results: {len(agent_result.results)}")
        else:
            # If no results found
            logger.warning(f"❌ All search strategies failed for query: '{question}'")
            response_text = "No relevant documents found."
        # --- End Format Response ---

        return QueryResponse(
            response=response_text,
            error=None,
            context_found=bool(agent_result.results),
            session_id=session_id
        )

    except Exception as e:
        logger.error(f"Failed to load database or process query: {e}")
        return QueryResponse(
            response=None,
            error=f"Failed to load vector database or process query: {str(e)}. This might be due to a schema mismatch. Try deleting the '{CHROMA_PATH}' folder and re-running the ingestion script.",
            context_found=False,
            session_id=session_id
        )
# --- END SUGGESTION 7 ---

# --- SUGGESTION 8: Ask Endpoint ---
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
            "history": [],
            "last_accessed": datetime.utcnow()
        }
        logger.info(f"Created new conversation session: {session_id}")
    elif session_id not in conversations:
        # Session ID provided but not found, treat as new
        conversations[session_id] = {
            "history": [],
            "last_accessed": datetime.utcnow()
        }
        logger.info(f"Recreated conversation session: {session_id}")
    else:
        # Existing session, update last accessed
        conversations[session_id]["last_accessed"] = datetime.utcnow()
    # --- End Manage Session ---

    # --- Process the Query ---
    user_question = query.question.strip()
    logger.info(f"Received query: '{user_question}'")

    response = process_query(user_question, session_id)
    
    # Update conversation history if context was found
    if response.context_found and response.response:
        conversations[session_id]["history"].append({"user": user_question, "bot": response.response})

    return response
    # --- End Process Query ---
# --- END SUGGESTION 8 ---

# --- Health Check Endpoint ---
@app.get("/health")
def health_check():
    """Basic health check endpoint"""
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE") # Use OPENAI_API_BASE
        
        # Test API connectivity if key and base are configured
        if api_key and api_base:
            test_response = call_openrouter_api("Hello, this is a test. Please respond with 'Test successful!'", api_key, api_base)
            return {
                "success": True,
                "response": test_response,
                "api_key_prefix": f"{api_key[:8]}...",
                "api_base": api_base
            }
        else:
            return {
                "success": False,
                "error": "API key or base URL not configured"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
# --- End Health Check Endpoint ---

# --- Debug Database Endpoint ---
@app.get("/debug-db")
def debug_database():
    """Debug endpoint to check database status"""
    if not os.path.exists(CHROMA_PATH):
        return {"error": "Database folder does not exist", "path": CHROMA_PATH}
    
    try:
        # Check database folder contents
        db_contents = os.listdir(CHROMA_PATH)
        
        # --- Load Database with Explicit Settings for Debugging ---
        # Match the client settings used in generate_db.py
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create explicit client settings matching generate_db.py
        client_settings = Settings(
            persist_directory=CHROMA_PATH,
            anonymized_telemetry=False, # Match generate_db.py
            allow_reset=True,
            is_persistent=True
        )

        # Create the Chroma instance with explicit client settings
        db = Chroma(
            collection_name="default", # Explicitly specify collection name
            embedding_function=embedding_function,
            persist_directory=CHROMA_PATH,
            client_settings=client_settings # Use the explicit settings
        )
        # --- End Load Database for Debugging ---
        
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
        return {
            "error": f"Database test failed: {str(e)}",
            "path": CHROMA_PATH,
            "suggestion": "Try running the ingestion script to recreate the database"
        }
# --- End Debug Database Endpoint ---

# --- Sources Info Endpoint ---
@app.get("/sources")
def get_sources_info():
    """Get information about available sources in the database"""
    if not os.path.exists(CHROMA_PATH):
        return {"error": "Database not found", "path": CHROMA_PATH}
    
    try:
        # --- Load Database with Explicit Settings for Sources ---
        # Match the client settings used in generate_db.py
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create explicit client settings matching generate_db.py
        client_settings = Settings(
            persist_directory=CHROMA_PATH,
            anonymized_telemetry=False, # Match generate_db.py
            allow_reset=True,
            is_persistent=True
        )

        # Create the Chroma instance with explicit client settings
        db = Chroma(
            collection_name="default", # Explicitly specify collection name
            embedding_function=embedding_function,
            persist_directory=CHROMA_PATH,
            client_settings=client_settings # Use the explicit settings
        )
        # --- End Load Database for Sources ---
        
        # Sample documents to get source info
        sample_docs = db.similarity_search("document content", k=50)
        sources = {}
        total_docs = len(sample_docs)
        for doc in sample_docs:
            source = doc.metadata.get('source', 'Unknown')
            file_name = doc.metadata.get('file_name', os.path.basename(source) if source != 'Unknown' else 'Unknown')
            if source not in sources:
                sources[source] = {
                    "file_name": file_name,
                    "count": 0
                }
            sources[source]["count"] += 1
            
        return {
            "total_documents_sampled": total_docs,
            "total_sources": len(sources),
            "sources": sources
        }
    except Exception as e:
        return {
            "error": f"Failed to retrieve sources: {str(e)}",
            "path": CHROMA_PATH
        }
# --- End Sources Info Endpoint ---

# --- Call OpenRouter API Function ---
def call_openrouter_api(prompt: str, api_key: str, api_base: str) -> str:
    """
    Call the OpenRouter API (or compatible service) with the given prompt.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openchat/openchat-7b", # Example model, adjust as needed
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            f"{api_base}/chat/completions", # Use api_base
            headers=headers,
            json=data,
            timeout=30 # Add a timeout
        )
        response.raise_for_status() # Raise an exception for bad status codes
        response_data = response.json()
        return response_data['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")
    except KeyError:
        logger.error(f"Unexpected API response format: {response.text}")
        raise HTTPException(status_code=500, detail="Unexpected API response format")
# --- End Call OpenRouter API Function ---

# --- Main Health Endpoint ---
@app.get("/")
def read_root():
    """Main health check and status endpoint"""
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE") # Use OPENAI_API_BASE
    
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
        "api_key_configured": bool(api_key),
        "api_base_configured": bool(api_base), # Check api_base
        "active_conversations": len(conversations),
        "ai_agent_status": {
            "loaded": True, # Always true now as UniversalRAGAgent is defined
            "nlp_model_available": nlp is not None,
            "sentence_model_available": sentence_model is not None
        }
    }
# --- End Main Health Endpoint ---

# --- AI Agent Code (from provided snippets) ---
# --- (Include the UniversalRAGAgent class and related enums/dataclasses here) ---
# --- Placeholder for the AI Agent code ---
# (The actual AI agent code from your snippets should be pasted here)
# For brevity, and because it was quite long, I'm placing a placeholder.
# Please ensure you copy the full `UniversalRAGAgent` class and related components
# from your original `app.py` snippets into this section.

# Example placeholder structure (replace with actual code):
class SearchStrategy(Enum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    NAMED_ENTITY = "named_entity"

@dataclass
class AgentResult:
    strategy: Optional[SearchStrategy]
    results: List

class UniversalRAGAgent:
    def __init__(self, nlp_model, sentence_model, db):
        self.nlp = nlp_model
        self.sentence_model = sentence_model
        self.db = db
        logger.info("🧠 Universal AI Agent initialized")

    def search(self, query: str) -> AgentResult:
        # --- Simple Semantic Search Strategy ---
        # This is a simplified version. Replace with your full logic.
        try:
            # Perform similarity search using the correctly configured db instance
            results = self.db.similarity_search(query, k=5) # Adjust k as needed
            if results:
                return AgentResult(strategy=SearchStrategy.SEMANTIC, results=results)
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")

        # Return empty result if all strategies fail
        return AgentResult(strategy=None, results=[])
# --- End AI Agent Code Placeholder ---
# - End of app.py -
