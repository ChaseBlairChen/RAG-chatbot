"""Configuration and environment variables - Enhanced for SMB/NGO/Environmental Focus"""
import os
from typing import List
from cryptography.fernet import Fernet

# API Configuration
OPENROUTER_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

# Database Paths
DEFAULT_CHROMA_PATH = os.path.abspath(os.path.join(os.getcwd(), "chromadb-database"))
USER_CONTAINERS_PATH = os.path.abspath(os.path.join(os.getcwd(), "user-containers"))

# File Processing
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
LEGAL_EXTENSIONS = {'.pdf', '.txt', '.docx', '.rtf'}

# Existing Premium Database Configuration
LEXISNEXIS_API_KEY = os.environ.get("LEXISNEXIS_API_KEY")
LEXISNEXIS_API_ENDPOINT = os.environ.get("LEXISNEXIS_API_ENDPOINT")
WESTLAW_API_KEY = os.environ.get("WESTLAW_API_KEY")
WESTLAW_API_ENDPOINT = os.environ.get("WESTLAW_API_ENDPOINT")

# === FREE & AFFORDABLE LEGAL DATABASES (HIGH PRIORITY) ===

# CourtListener - FREE comprehensive legal database
COURTLISTENER_API_KEY = os.environ.get("COURTLISTENER_API_KEY", "")  # Optional - increases rate limits
COURTLISTENER_API_ENDPOINT = "https://www.courtlistener.com/api/rest/v3/"

# Harvard Caselaw Access Project - FREE
HARVARD_LEGAL_API_KEY = os.environ.get("HARVARD_LEGAL_API_KEY")
HARVARD_LEGAL_API_ENDPOINT = os.environ.get("HARVARD_LEGAL_API_ENDPOINT", "https://api.case.law/v1")

# Justia - FREE legal database
JUSTIA_API_ENDPOINT = "https://law.justia.com/api/"  # No key required for basic access

# Google Scholar - Web scraping (be respectful of rate limits)
GOOGLE_SCHOLAR_ENABLED = os.environ.get("GOOGLE_SCHOLAR_ENABLED", "true").lower() == "true"

# === GOVERNMENT & REGULATORY APIS (FREE) ===

# Federal Register API - FREE
FEDERAL_REGISTER_API_ENDPOINT = "https://www.federalregister.gov/api/v1/"

# Congress.gov API - FREE
CONGRESS_API_KEY = os.environ.get("CONGRESS_API_KEY")  # Required
CONGRESS_API_ENDPOINT = "https://api.congress.gov/v3/"

# SEC EDGAR API - FREE
SEC_EDGAR_API_ENDPOINT = "https://data.sec.gov/api/"

# EPA Environmental Data - FREE
EPA_API_ENDPOINT = "https://www.epa.gov/enviro/web-services"

# === ENVIRONMENTAL & CLIMATE LAW SPECIFIC ===

# Climate Policy Initiative API
CLIMATE_POLICY_API_ENDPOINT = "https://climatepolicyinitiative.org/api/"

# Environmental Law Institute (if they have an API)
ELI_API_KEY = os.environ.get("ELI_API_KEY")
ELI_API_ENDPOINT = os.environ.get("ELI_API_ENDPOINT")

# Carbon Disclosure Project API
CDP_API_KEY = os.environ.get("CDP_API_KEY")
CDP_API_ENDPOINT = "https://data.cdp.net/api/"

# === INTERNATIONAL & NGO RESOURCES ===

# UN Treaty Collection
UN_TREATY_API_ENDPOINT = "https://treaties.un.org/api/"

# World Bank Open Data API - FREE
WORLD_BANK_API_ENDPOINT = "https://api.worldbank.org/v2/"

# === SMB-FOCUSED RESOURCES ===

# SBA (Small Business Administration) API - FREE
SBA_API_ENDPOINT = "https://api.sba.gov/"

# USPTO (Patent/Trademark) API - FREE
USPTO_API_ENDPOINT = "https://developer.uspto.gov/api-catalog"

# Business.gov API (if available)
BUSINESS_GOV_API_ENDPOINT = os.environ.get("BUSINESS_GOV_API_ENDPOINT")

# === STATE & LOCAL GOVERNMENT ===

# Many states have open data portals - configure as needed
STATE_APIS = {
    "california": "https://data.ca.gov/api/",
    "new_york": "https://data.ny.gov/api/",
    "texas": "https://data.texas.gov/api/",
    # Add more states as needed
}

# Immigration Features
SUPPORTED_LANGUAGES = ["en", "es", "zh", "ar", "fr", "hi", "pt", "ru", "bn", "ur"]
AUTO_TRANSLATION_ENABLED = False  # Set to True when translation API is configured
RFE_STANDARD_DEADLINE_DAYS = 87
ASYLUM_ONE_YEAR_DEADLINE_DAYS = 365

# Security for Immigration Data
ENCRYPTION_KEY = os.environ.get("IMMIGRATION_ENCRYPTION_KEY", Fernet.generate_key())
DATA_RETENTION_DAYS = 2555  # 7 years per immigration law requirements
HIPAA_COMPLIANT_STORAGE = True

# Batch Processing
MAX_BATCH_SIZE = 50
BATCH_PROCESSING_TIMEOUT = 300  # 5 minutes

# Mobile Optimization
MOBILE_CHUNK_SIZE = 500  # Smaller chunks for mobile
LOW_BANDWIDTH_MODE = True

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

# === API PRIORITY CONFIGURATION ===
# Define which APIs to prioritize based on user tier/subscription
API_TIERS = {
    "free": [
        "COURTLISTENER", "HARVARD_LEGAL", "JUSTIA", "FEDERAL_REGISTER", 
        "CONGRESS", "SEC_EDGAR", "EPA", "WORLD_BANK", "SBA", "USPTO"
    ],
    "basic": [
        "free", "CLIMATE_POLICY", "CDP", "UN_TREATY"
    ],
    "premium": [
        "basic", "LEXISNEXIS", "WESTLAW", "ELI"
    ]
}

# Rate limiting configuration for each API
API_RATE_LIMITS = {
    "COURTLISTENER": {"requests_per_hour": 100, "burst": 10},
    "HARVARD_LEGAL": {"requests_per_hour": 500, "burst": 20},
    "FEDERAL_REGISTER": {"requests_per_hour": 1000, "burst": 50},
    "CONGRESS": {"requests_per_hour": 5000, "burst": 100},
    "SEC_EDGAR": {"requests_per_hour": 500, "burst": 25},
    "EPA": {"requests_per_hour": 200, "burst": 10},
    "GOOGLE_SCHOLAR": {"requests_per_hour": 20, "burst": 2},  # Be very conservative
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
    
    # New API availability flags
    COURTLISTENER_AVAILABLE: bool = True  # Always available (free)
    HARVARD_LEGAL_AVAILABLE: bool = bool(HARVARD_LEGAL_API_KEY)
    CONGRESS_AVAILABLE: bool = bool(CONGRESS_API_KEY)
    ENVIRONMENTAL_APIs_AVAILABLE: bool = True  # Most are free

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

    # Check API availability
    print("🔍 Checking API availability...")
    if FeatureFlags.COURTLISTENER_AVAILABLE:
        print("✅ CourtListener API available (free legal database)")
    if FeatureFlags.HARVARD_LEGAL_AVAILABLE:
        print("✅ Harvard Legal API configured")
    if FeatureFlags.CONGRESS_AVAILABLE:
        print("✅ Congress.gov API configured")
    if FeatureFlags.ENVIRONMENTAL_APIs_AVAILABLE:
        print("✅ Environmental APIs available (EPA, Federal Register)")

# Initialize feature flags when module is imported
initialize_feature_flags()
