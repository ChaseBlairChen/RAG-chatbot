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

# Chunk Sizes
DEFAULT_CHUNK_SIZE = 1500
LEGISLATIVE_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 300
LEGISLATIVE_CHUNK_OVERLAP = 500

# Search Settings
DEFAULT_SEARCH_K = 10
ENHANCED_SEARCH_K = 12
COMPREHENSIVE_SEARCH_K = 20
MIN_RELEVANCE_SCORE = 0.15

# Confidence Score Weights
CONFIDENCE_WEIGHTS = {
    "relevance": 0.4,
    "document_count": 0.3,
    "consistency": 0.2,
    "completeness": 0.1
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
    HYBRID_SEARCH_AVAILABLE: bool = False

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

    # Check hybrid search
    try:
        import rank_bm25
        FeatureFlags.HYBRID_SEARCH_AVAILABLE = True
        print("✅ Hybrid search available - better retrieval accuracy")
    except ImportError:
        FeatureFlags.HYBRID_SEARCH_AVAILABLE = False
        print("⚠️ Hybrid search not available - install rank-bm25")
    
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

# Initialize feature flags when module is imported
initialize_feature_flags()
