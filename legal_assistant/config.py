"""Configuration and environment variables"""
import os
from typing import List

# API Configuration
OPENROUTER_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

# Database Paths
DEFAULT_CHROMA_PATH = os.path.abspath(os.path.join(os.getcwd(), "chromadb-database"))
USER_CONTAINERS_PATH = os.path.abspath(os.path.join(os.getcwd(), "user-containers"))

# File Processing
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
LEGAL_EXTENSIONS = {'.pdf', '.txt', '.docx', '.rtf'}

# External Database Configuration
LEXISNEXIS_API_KEY = os.environ.get("LEXISNEXIS_API_KEY")
LEXISNEXIS_API_ENDPOINT = os.environ.get("LEXISNEXIS_API_ENDPOINT")
WESTLAW_API_KEY = os.environ.get("WESTLAW_API_KEY")
WESTLAW_API_ENDPOINT = os.environ.get("WESTLAW_API_ENDPOINT")

# Harvard Legal Library Configuration (NEW)
HARVARD_LEGAL_API_KEY = os.environ.get("HARVARD_LEGAL_API_KEY")
HARVARD_LEGAL_API_ENDPOINT = os.environ.get("HARVARD_LEGAL_API_ENDPOINT", "https://api.case.law/v1")

# Model Names
EMBEDDING_MODELS = [
    "nlpaueb/legal-bert-base-uncased",
    "law-ai/InCaseLawBERT", 
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-roberta-large-v1",
    "microsoft/DialoGPT-medium",
    "sentence-transformers/all-MiniLM-L12-v2",
    "all-MiniLM-L6-v2"
]

FAST_EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
]

# AI Models
AI_MODELS = [
    "moonshotai/kimi-k2:free",
    "deepseek/deepseek-chat",
    "microsoft/phi-3-mini-128k-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free",
    "openchat/openchat-7b:free"
]

# Chunk Sizes - OPTIMIZED
DEFAULT_CHUNK_SIZE = 1000     # Reduced from 1500
LEGISLATIVE_CHUNK_SIZE = 1500 # Reduced from 2000
DEFAULT_CHUNK_OVERLAP = 200   # Reduced from 300
LEGISLATIVE_CHUNK_OVERLAP = 300  # Reduced from 500

# Search Settings - IMPROVED
DEFAULT_SEARCH_K = 15  # Increased from 10
ENHANCED_SEARCH_K = 20  # Increased from 12
COMPREHENSIVE_SEARCH_K = 30  # Increased from 20
MIN_RELEVANCE_SCORE = 0.1  # CHANGED from 0.3 to be more permissive

# Confidence Score Weights - REBALANCED
CONFIDENCE_WEIGHTS = {
    "relevance": 0.5,     # Increased weight on relevance
    "document_count": 0.2, # Decreased weight on count
    "consistency": 0.2,
    "completeness": 0.1
}

# Add new search configuration
SEARCH_CONFIG = {
    "rerank_enabled": True,
    "hybrid_search_enabled": False,  # CHANGED to False to disable hybrid search
    "keyword_weight": 0.3,
    "semantic_weight": 0.7,
    "max_results_to_rerank": 50,
    "query_expansion_enabled": True,
    "min_score_threshold": 0.1,  # CHANGED from 0.3 to be more permissive
    "boost_exact_matches": True,
    "boost_factor": 1.5
}

# Feature Flags
class FeatureFlags:
    AI_ENABLED: bool = bool(OPENROUTER_API_KEY)
    AIOHTTP_AVAILABLE: bool = False
    OPEN_SOURCE_NLP_AVAILABLE: bool = False
    PYMUPDF_AVAILABLE: bool = False
    PDFPLUMBER_AVAILABLE: bool = False
    UNSTRUCTURED_AVAILABLE: bool = False
    OCR_AVAILABLE: bool = False
    HYBRID_SEARCH_AVAILABLE: bool = False  # CHANGED to False to disable hybrid search

def initialize_feature_flags():
    """Initialize feature flags by checking for available dependencies"""
    
    # Check OCR
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
        FeatureFlags.OCR_AVAILABLE = True
        print("✅ OCR support available - can process scanned PDFs")
    except ImportError:
        FeatureFlags.OCR_AVAILABLE = False
        print("⚠️ OCR not available - install pytesseract and pdf2image")

    # Hybrid search is now disabled by default
    FeatureFlags.HYBRID_SEARCH_AVAILABLE = False
    print("⚠️ Hybrid search disabled for better performance")
    
    # Check existing features
    try:
        import aiohttp
        FeatureFlags.AIOHTTP_AVAILABLE = True
        print("✅ Async HTTP support available")
    except ImportError:
        FeatureFlags.AIOHTTP_AVAILABLE = False
        print("⚠️ Async HTTP not available - install aiohttp")
    
    try:
        import spacy
        import nltk
        FeatureFlags.OPEN_SOURCE_NLP_AVAILABLE = True
        print("✅ Open source NLP available")
    except ImportError:
        FeatureFlags.OPEN_SOURCE_NLP_AVAILABLE = False
        print("⚠️ Open source NLP not available - install spacy and nltk")
    
    try:
        import fitz  # PyMuPDF
        FeatureFlags.PYMUPDF_AVAILABLE = True
        print("✅ PyMuPDF available for PDF processing")
    except ImportError:
        FeatureFlags.PYMUPDF_AVAILABLE = False
        print("⚠️ PyMuPDF not available - install PyMuPDF")
    
    try:
        import pdfplumber
        FeatureFlags.PDFPLUMBER_AVAILABLE = True
        print("✅ PDFPlumber available for advanced PDF parsing")
    except ImportError:
        FeatureFlags.PDFPLUMBER_AVAILABLE = False
        print("⚠️ PDFPlumber not available - install pdfplumber")
    
    try:
        import unstructured
        FeatureFlags.UNSTRUCTURED_AVAILABLE = True
        print("✅ Unstructured available for document parsing")
    except ImportError:
        FeatureFlags.UNSTRUCTURED_AVAILABLE = False
        print("⚠️ Unstructured not available - install unstructured")

# Free Legal Database Configuration
COURTLISTENER_API_KEY = os.environ.get("COURTLISTENER_API_KEY", "")  # Optional - increases rate limits

# Initialize feature flags when module is imported
initialize_feature_flags()

