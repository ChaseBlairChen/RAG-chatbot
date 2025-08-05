"""Dependency injection and initialization"""
import logging
from typing import Optional
import spacy
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from ..config import EMBEDDING_MODELS, FAST_EMBEDDING_MODELS, FeatureFlags

logger = logging.getLogger(__name__)

# Global instances
nlp: Optional[spacy.Language] = None
sentence_model: Optional[SentenceTransformer] = None
embeddings: Optional[HuggingFaceEmbeddings] = None
sentence_model_name: Optional[str] = None

def initialize_nlp_models():
    """Initialize NLP models with comprehensive error handling"""
    global nlp, sentence_model, embeddings, sentence_model_name
    
    logger.info("🔄 Initializing NLP models...")
    
    # Load spaCy with graceful fallback
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("✅ spaCy model loaded successfully")
    except IOError:
        logger.warning("⚠️ spaCy model 'en_core_web_sm' not found")
        logger.info("   Install with: python -m spacy download en_core_web_sm")
        nlp = None
    except Exception as e:
        logger.error(f"❌ Failed to load spaCy model: {e}")
        nlp = None
    
    # Load sentence transformer with priority fallback
    logger.info("🔄 Loading sentence transformer model...")
    for model_name in EMBEDDING_MODELS:
        try:
            logger.info(f"   Trying {model_name}...")
            sentence_model = SentenceTransformer(model_name)
            sentence_model_name = model_name
            logger.info(f"✅ Loaded powerful sentence model: {model_name}")
            break
        except Exception as e:
            logger.warning(f"   Failed to load {model_name}: {e}")
            continue
    
    if sentence_model is None:
        logger.error("❌ Failed to load any sentence transformer model")
        sentence_model_name = "none"
    
    # Load embeddings with comprehensive fallback
    logger.info("🔄 Loading embeddings...")
    try:
        if sentence_model_name and sentence_model_name != "none":
            embeddings = HuggingFaceEmbeddings(model_name=sentence_model_name)
            logger.info(f"✅ Loaded embeddings with: {sentence_model_name}")
        else:
            # Try fallback models
            for fallback_model in FAST_EMBEDDING_MODELS:
                try:
                    embeddings = HuggingFaceEmbeddings(model_name=fallback_model)
                    logger.info(f"✅ Using fallback embeddings: {fallback_model}")
                    break
                except Exception as e:
                    logger.warning(f"   Fallback {fallback_model} failed: {e}")
            
            if embeddings is None:
                # Final fallback
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                logger.warning("⚠️ Using final fallback embeddings: all-MiniLM-L6-v2")
                
    except Exception as e:
        logger.error(f"❌ Failed to load embeddings: {e}")
        embeddings = None
    
    # Summary
    models_loaded = sum([
        nlp is not None,
        sentence_model is not None, 
        embeddings is not None
    ])
    
    logger.info(f"📊 NLP Initialization Summary: {models_loaded}/3 models loaded successfully")
    if models_loaded < 3:
        logger.warning(f"⚠️ Some NLP features may be limited. Consider installing missing dependencies.")

def initialize_feature_flags():
    """Initialize feature availability flags"""
    logger.info("🔍 Checking feature availability...")
    
    # Check aiohttp for AI features
    try:
        import aiohttp
        FeatureFlags.AIOHTTP_AVAILABLE = True
        logger.info("✅ aiohttp available - HTTP client features enabled")
    except ImportError:
        FeatureFlags.AIOHTTP_AVAILABLE = False
        logger.warning("⚠️ aiohttp not available - install with: pip install aiohttp")
    
    # Check open source NLP
    try:
        import torch
        from transformers import pipeline
        FeatureFlags.OPEN_SOURCE_NLP_AVAILABLE = True
        logger.info("✅ Open-source NLP models available")
    except ImportError as e:
        FeatureFlags.OPEN_SOURCE_NLP_AVAILABLE = False
        logger.warning(f"⚠️ Open-source NLP models not available: {e}")
        logger.info("   Install with: pip install transformers torch")
    
    # Check PDF processors
    try:
        import fitz
        FeatureFlags.PYMUPDF_AVAILABLE = True
        logger.info("✅ PyMuPDF available - enhanced PDF extraction enabled")
    except ImportError:
        FeatureFlags.PYMUPDF_AVAILABLE = False
        logger.warning("⚠️ PyMuPDF not available - install with: pip install PyMuPDF")
    
    try:
        import pdfplumber
        FeatureFlags.PDFPLUMBER_AVAILABLE = True
        logger.info("✅ pdfplumber available - table extraction enabled")
    except ImportError:
        FeatureFlags.PDFPLUMBER_AVAILABLE = False
        logger.warning("⚠️ pdfplumber not available - install with: pip install pdfplumber")
    
    try:
        from unstructured.partition.auto import partition
        FeatureFlags.UNSTRUCTURED_AVAILABLE = True
        logger.info("✅ Unstructured.io available - advanced document processing enabled")
    except ImportError:
        FeatureFlags.UNSTRUCTURED_AVAILABLE = False
        logger.warning("⚠️ Unstructured.io not available - install with: pip install unstructured[all-docs]")
    
    # Check OCR capabilities
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
        FeatureFlags.OCR_AVAILABLE = True
        logger.info("✅ OCR support available - can process scanned documents")
    except ImportError:
        FeatureFlags.OCR_AVAILABLE = False
        logger.warning("⚠️ OCR not available - install pytesseract and pdf2image for scanned PDFs")
    
    # Update AI enabled flag
    FeatureFlags.AI_ENABLED = bool(FeatureFlags.AIOHTTP_AVAILABLE and OPENROUTER_API_KEY)
    
    # Summary
    features_enabled = sum([
        FeatureFlags.AI_ENABLED,
        FeatureFlags.PYMUPDF_AVAILABLE,
        FeatureFlags.PDFPLUMBER_AVAILABLE,
        FeatureFlags.UNSTRUCTURED_AVAILABLE,
        FeatureFlags.OCR_AVAILABLE
    ])
    
    logger.info(f"📊 Features Summary: {features_enabled}/5 advanced features available")
    logger.info(f"🤖 AI Status: {'ENABLED' if FeatureFlags.AI_ENABLED else 'DISABLED - Set OPENAI_API_KEY to enable'}")
    logger.info(f"📄 Document Processing: PyMuPDF={FeatureFlags.PYMUPDF_AVAILABLE}, "
               f"pdfplumber={FeatureFlags.PDFPLUMBER_AVAILABLE}, "
               f"Unstructured={FeatureFlags.UNSTRUCTURED_AVAILABLE}")

def get_nlp():
    """Get NLP instance"""
    return nlp

def get_sentence_model():
    """Get sentence model instance"""
    return sentence_model

def get_embeddings():
    """Get embeddings instance"""
    return embeddings

def get_sentence_model_name():
    """Get sentence model name"""
    return sentence_model_name
