# Enhanced AI Agent with Document Analysis Capabilities
from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
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

# Third-party library imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from chromadb.config import Settings

# Document processing imports
try:
    import PyPDF2
    import docx
    from pdfplumber import PDF
except ImportError:
    print("Document processing libraries not installed. Run: pip install PyPDF2 python-docx pdfplumber")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI(
    title="Enhanced Legal Document Analysis Agent", 
    description="Comprehensive Legal Analysis with Document Processing", 
    version="4.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables and configuration
CHROMA_PATH = "chroma"
CHROMA_CLIENT_SETTINGS = Settings(
    persist_directory=CHROMA_PATH,
    anonymized_telemetry=False,
    allow_reset=True,
    is_persistent=True
)

# In-memory conversation storage
conversations = {}

# Document processing utilities
class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF file"""
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""

    @staticmethod
    def extract_text_from_docx(file_content: bytes) -> str:
        """Extract text from DOCX file"""
        try:
            doc_file = io.BytesIO(file_content)
            doc = docx.Document(doc_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            return ""

    @staticmethod
    def extract_text_from_txt(file_content: bytes) -> str:
        """Extract text from TXT file"""
        try:
            return file_content.decode('utf-8', errors='ignore').strip()
        except Exception as e:
            logger.error(f"Error extracting TXT text: {e}")
            return ""

    @staticmethod
    def process_document(file: UploadFile) -> Tuple[str, str]:
        """Process uploaded document and extract text"""
        try:
            file_content = file.file.read()
            file.file.seek(0)  # Reset file pointer
            
            filename = file.filename.lower()
            
            if filename.endswith('.pdf'):
                text = DocumentProcessor.extract_text_from_pdf(file_content)
                doc_type = "PDF"
            elif filename.endswith('.docx'):
                text = DocumentProcessor.extract_text_from_docx(file_content)
                doc_type = "Word Document"
            elif filename.endswith('.txt'):
                text = DocumentProcessor.extract_text_from_txt(file_content)
                doc_type = "Text Document"
            else:
                raise ValueError(f"Unsupported file type: {filename}")
            
            if not text.strip():
                raise ValueError("No text could be extracted from the document")
                
            return text, doc_type
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            raise HTTPException(status_code=400, detail=f"Document processing failed: {str(e)}")

# Enhanced Pydantic Models
class LegalQuery(BaseModel):
    question: str
    session_id: Optional[str] = None
    response_style: Optional[str] = "balanced"
    analysis_type: Optional[str] = "comprehensive"
    jurisdiction: Optional[str] = "federal"
    case_types: Optional[List[str]] = None
    time_period: Optional[str] = "all"

class AskQuery(BaseModel):
    question: str
    session_id: Optional[str] = None

class DocumentAnalysisRequest(BaseModel):
    analysis_type: str
    prompt_template: str
    session_id: Optional[str] = None
    response_style: Optional[str] = "balanced"

class LegalAnalysisResponse(BaseModel):
    response: Optional[str] = None
    error: Optional[str] = None
    context_found: bool = False
    sources: Optional[List[Dict]] = None
    session_id: str
    confidence_score: float = 0.0
    expand_available: bool = False
    document_info: Optional[Dict] = None
    analysis_metadata: Optional[Dict] = None

# Case Law Analysis Data Models (from original code)
class CaseType(Enum):
    CRIMINAL = "criminal"
    CIVIL = "civil"
    CONSTITUTIONAL = "constitutional"
    ADMINISTRATIVE = "administrative"
    IMMIGRATION = "immigration"
    CORPORATE = "corporate"
    FAMILY = "family"
    PROPERTY = "property"
    CONTRACT = "contract"
    TORT = "tort"

class JurisdictionLevel(Enum):
    SUPREME_COURT = "supreme_court"
    APPELLATE = "appellate"
    FEDERAL_DISTRICT = "federal_district"
    STATE_SUPREME = "state_supreme"
    STATE_APPELLATE = "state_appellate"
    STATE_TRIAL = "state_trial"
    ADMINISTRATIVE = "administrative"

@dataclass
class CaseCitation:
    case_name: str
    citation: str
    year: int
    court: str
    jurisdiction_level: JurisdictionLevel
    case_type: CaseType
    relevance_score: float = 0.0

# Document Analysis Engine
class DocumentAnalysisEngine:
    def __init__(self):
        self.predefined_prompts = {
            'summarize': 'Summarize this legal document in plain English, keeping the legal tone intact. Highlight purpose, parties involved, and key terms.',
            'extract-clauses': 'Extract and list the clauses related to termination, indemnification, liability, governing law, and confidentiality.',
            'missing-clauses': 'Analyze this contract and flag any commonly expected legal clauses that are missing, such as limitation of liability or dispute resolution.',
            'risk-flagging': 'Identify any clauses that may pose legal risks to the signing party, such as unilateral termination, broad indemnity, or vague obligations.',
            'contract-comparison': 'Compare these two versions of a contract and list all changes by clause, highlighting any that introduce additional legal risk or obligation.',
            'timeline-extraction': 'Extract and list all dates, deadlines, renewal periods, and notice periods mentioned in this document.',
            'obligations': 'List all actions or obligations the signing party is required to perform, along with associated deadlines or conditions.',
            'client-friendly': 'Rewrite this document summary for a client who is not a lawyer, using clear and simple language.'
        }

    def analyze_document(self, document_text: str, analysis_type: str, custom_prompt: str = None) -> Dict:
        """Analyze document with specified analysis type"""
        prompt = custom_prompt or self.predefined_prompts.get(analysis_type, '')
        
        if not prompt:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        # Combine prompt with document text
        full_prompt = f"{prompt}\n\nDocument to analyze:\n{document_text}"
        
        # Extract document metadata
        metadata = self.extract_document_metadata(document_text)
        
        return {
            'prompt': full_prompt,
            'analysis_type': analysis_type,
            'document_length': len(document_text),
            'word_count': len(document_text.split()),
            'metadata': metadata
        }

    def extract_document_metadata(self, text: str) -> Dict:
        """Extract metadata from document text"""
        # Simple metadata extraction
        word_count = len(text.split())
        char_count = len(text)
        
        # Look for common legal document indicators
        has_parties = bool(re.search(r'(party|parties|between|among)', text, re.IGNORECASE))
        has_dates = bool(re.search(r'\b(19|20)\d{2}\b', text))
        has_signatures = bool(re.search(r'(signature|signed|execute)', text, re.IGNORECASE))
        
        # Estimate document type
        doc_type = "Unknown"
        if re.search(r'(contract|agreement|terms)', text, re.IGNORECASE):
            doc_type = "Contract/Agreement"
        elif re.search(r'(policy|procedure|guidelines)', text, re.IGNORECASE):
            doc_type = "Policy Document"
        elif re.search(r'(lease|rent|tenant)', text, re.IGNORECASE):
            doc_type = "Lease Agreement"
        elif re.search(r'(employment|employee|employer)', text, re.IGNORECASE):
            doc_type = "Employment Document"
        
        return {
            'word_count': word_count,
            'character_count': char_count,
            'estimated_type': doc_type,
            'has_parties': has_parties,
            'has_dates': has_dates,
            'has_signatures': has_signatures,
            'complexity_score': min(1.0, word_count / 5000)  # Simple complexity metric
        }

# Initialize analysis engine
document_analyzer = DocumentAnalysisEngine()

# Database and utility functions (keeping from original)
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

def get_conversation_context(session_id: str, max_messages: int = 10) -> str:
    """Get conversation context"""
    if session_id not in conversations:
        return ""
    messages = conversations[session_id]['messages'][-max_messages:]
    context_parts = []
    for msg in messages:
        role = msg['role'].upper()
        content = msg['content'][:600] + "..." if len(msg['content']) > 600 else msg['content']
        context_parts.append(f"{role}: {content}")
    return "\n".join(context_parts) if context_parts else ""

def add_to_conversation(session_id: str, role: str, content: str, sources: Optional[List] = None, document_info: Optional[Dict] = None):
    """Add message to conversation history"""
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
        'sources': sources or [],
        'document_info': document_info
    }
    conversations[session_id]['messages'].append(message)
    conversations[session_id]['last_accessed'] = datetime.utcnow()
    if len(conversations[session_id]['messages']) > 50:
        conversations[session_id]['messages'] = conversations[session_id]['messages'][-50:]

def call_openrouter_api(prompt: str, api_key: str, api_base: str = "https://openrouter.ai/api/v1") -> str:
    """Call OpenRouter API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Legal Document Analysis Assistant"
    }
    
    models_to_try = [
        "anthropic/claude-3-sonnet",
        "openai/gpt-4-turbo-preview",
        "deepseek/deepseek-chat-v3-0324:free",
        "microsoft/phi-3-mini-128k-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free"
    ]
    
    for model in models_to_try:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 4000,
                "top_p": 0.9
            }
            response = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=90)
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

def cleanup_expired_conversations():
    """Remove expired conversations"""
    now = datetime.utcnow()
    expired_sessions = [
        session_id for session_id, data in conversations.items()
        if now - data['last_accessed'] > timedelta(hours=2)
    ]
    for session_id in expired_sessions:
        del conversations[session_id]

# API Endpoints

@app.post("/document-analysis", response_model=LegalAnalysisResponse)
async def document_analysis_endpoint(
    file: UploadFile = File(...),
    analysis_type: str = Form(...),
    session_id: Optional[str] = Form(None),
    response_style: str = Form("balanced")
):
    """Analyze uploaded legal document"""
    cleanup_expired_conversations()
    
    session_id = session_id or str(uuid.uuid4())
    if session_id not in conversations:
        conversations[session_id] = {
            "messages": [],
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow()
        }
    else:
        conversations[session_id]["last_accessed"] = datetime.utcnow()

    try:
        # Process the document and extract text
        document_text, doc_type = DocumentProcessor.process_document(file)
        
        # Get the analysis prompt template
        analysis_data = document_analyzer.analyze_document(document_text, analysis_type)
        
        # Create the complete prompt with document content
        complete_prompt = f"""You are a legal document analysis assistant. {analysis_data['prompt']}

DOCUMENT CONTENT TO ANALYZE:
{document_text}

Please provide a detailed analysis based on the request above. Focus on being accurate, thorough, and helpful."""
        
        # Get API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return LegalAnalysisResponse(
                response=None,
                error="API configuration error. Please contact administrator.",
                context_found=False,
                sources=[],
                session_id=session_id,
                confidence_score=0.0,
                expand_available=False
            )

        # Call LLM with the complete prompt including document content
        api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        raw_response = call_openrouter_api(complete_prompt, api_key, api_base)
        
        # Document info for response
        document_info = {
            'filename': file.filename,
            'file_type': doc_type,
            'file_size': len(document_text),
            'metadata': analysis_data['metadata']
        }
        
        # Analysis metadata
        analysis_metadata = {
            'analysis_type': analysis_type,
            'processing_time': datetime.utcnow().isoformat(),
            'document_stats': {
                'word_count': analysis_data['word_count'],
                'character_count': analysis_data['document_length']
            }
        }
        
        # Calculate confidence score based on document length and successful processing
        confidence_score = min(1.0, max(0.3, len(document_text) / 10000))
        if len(document_text) > 1000:  # Good amount of text extracted
            confidence_score = max(confidence_score, 0.8)
        
        # Update conversation
        add_to_conversation(session_id, "user", f"Document Analysis Request: {analysis_type} for {file.filename}", document_info=document_info)
        add_to_conversation(session_id, "assistant", raw_response, document_info=document_info)
        
        return LegalAnalysisResponse(
            response=raw_response,
            error=None,
            context_found=True,
            sources=[],
            session_id=session_id,
            confidence_score=confidence_score,
            expand_available=True,
            document_info=document_info,
            analysis_metadata=analysis_metadata
        )
        
    except Exception as e:
        logger.error(f"Document analysis failed: {e}", exc_info=True)
        return LegalAnalysisResponse(
            response=None,
            error=f"Document analysis failed: {str(e)}",
            context_found=False,
            sources=[],
            session_id=session_id,
            confidence_score=0.0,
            expand_available=False
        )

@app.get("/analysis-types")
def get_analysis_types():
    """Get available analysis types"""
    return {
        "analysis_types": list(document_analyzer.predefined_prompts.keys()),
        "descriptions": {
            "summarize": "Legal Document Summarization",
            "extract-clauses": "Key Clause Extraction", 
            "missing-clauses": "Missing Clause Detection",
            "risk-flagging": "Legal Risk Flagging",
            "contract-comparison": "Contract Version Comparison",
            "timeline-extraction": "Timeline & Deadline Extraction",
            "obligations": "Obligation Summary",
            "client-friendly": "Client-Friendly Summary"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting Enhanced Legal Document Analysis Agent on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
