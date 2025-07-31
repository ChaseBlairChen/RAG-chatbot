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
    """Initialize NLP models"""
    global nlp, sentence_model, embeddings, sentence_model_name
    
    # Load spaCy
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("✅ spaCy model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load spaCy model: {e}")
        nlp = None
    
    # Load sentence transformer
    for model_name in EMBEDDING_MODELS:
        try:
            sentence_model = SentenceTransformer(model_name)
            sentence_model_name = model_name
            logger.info(f"✅ Loaded powerful sentence model: {model_name}")
            break
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            continue
    
    if sentence_model is None:
        logger.error("❌ Failed to load any sentence transformer model")
        sentence_model_name = "none"
    
    # Load embeddings
    try:
        if sentence_model_name and sentence_model_name != "none":
            embeddings = HuggingFaceEmbeddings(model_name=sentence_model_name)
            logger.info(f"✅ Loaded embeddings with: {sentence_model_name}")
        else:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            logger.info("⚠️ Using fallback embeddings: all-MiniLM-L6-v2")
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        embeddings = None

def initialize_feature_flags():
    """Initialize feature availability flags"""
    # Check aiohttp
    try:
        import aiohttp
        FeatureFlags.AIOHTTP_AVAILABLE = True
    except ImportError:
        FeatureFlags.AIOHTTP_AVAILABLE = False
        print("⚠️ aiohttp not available - AI features disabled. Install with: pip install aiohttp")
    
    # Check open source NLP
    try:
        import torch
        from transformers import pipeline
        FeatureFlags.OPEN_SOURCE_NLP_AVAILABLE = True
        logger.info("✅ Open-source NLP models available")
    except ImportError as e:
        logger.warning(f"⚠️ Open-source NLP models not available: {e}")
        print("Install with: pip install transformers torch")
    
    # Check PDF processors
    try:
        import fitz
        FeatureFlags.PYMUPDF_AVAILABLE = True
        print("✅ PyMuPDF available - using enhanced PDF extraction")
    except ImportError as e:
        print(f"⚠️ PyMuPDF not available: {e}")
        print("Install with: pip install PyMuPDF")
    
    try:
        import pdfplumber
        FeatureFlags.PDFPLUMBER_AVAILABLE = True
        print("✅ pdfplumber available - using enhanced PDF extraction")
    except ImportError as e:
        print(f"⚠️ pdfplumber not available: {e}")
        print("Install with: pip install pdfplumber")
    
    try:
        from unstructured.partition.auto import partition
        FeatureFlags.UNSTRUCTURED_AVAILABLE = True
        print("✅ Unstructured.io available - using advanced document processing")
    except ImportError as e:
        print(f"⚠️ Unstructured.io not available: {e}")
        print("Install with: pip install unstructured[all-docs]")
    
    # Update AI enabled flag
    FeatureFlags.AI_ENABLED = bool(FeatureFlags.AIOHTTP_AVAILABLE and FeatureFlags.AI_ENABLED)
    
    print(f"Document processing status: PyMuPDF={FeatureFlags.PYMUPDF_AVAILABLE}, pdfplumber={FeatureFlags.PDFPLUMBER_AVAILABLE}, Unstructured={FeatureFlags.UNSTRUCTURED_AVAILABLE}")

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
