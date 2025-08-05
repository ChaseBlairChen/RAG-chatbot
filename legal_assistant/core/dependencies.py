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
    """Initialize NLP models with error handling"""
    global nlp, sentence_model, embeddings, sentence_model_name
    
    # Load spaCy with fallback
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("✅ spaCy model loaded successfully")
    except Exception as e:
        logger.warning(f"spaCy model not available: {e}")
        nlp = None
    
    # Load sentence transformer with fallback
    for model_name in EMBEDDING_MODELS:
        try:
            sentence_model = SentenceTransformer(model_name)
            sentence_model_name = model_name
            logger.info(f"✅ Loaded sentence model: {model_name}")
            break
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            continue
    
    if sentence_model is None:
        logger.error("❌ Failed to load any sentence transformer model")
        sentence_model_name = "none"
    
    # Load embeddings with fallback
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

# ===== legal_assistant/services/ai_service.py =====
"""AI/LLM Integration Service"""
import logging
import requests
from typing import Optional, Dict, Any, List

from ..config import AI_MODELS, OPENROUTER_API_KEY, OPENAI_API_BASE, APP_REFERER, APP_TITLE

logger = logging.getLogger(__name__)

class OpenRouterService:
    """A service for making API calls to OpenRouter with fallback models"""

    def __init__(
        self,
        api_key: str = None,
        api_base: str = None,
        models: List[str] = None,
        timeout: int = 60
    ):
        """Initialize the OpenRouter service"""
        self.api_key = api_key or OPENROUTER_API_KEY
        self.api_base = api_base or OPENAI_API_BASE
        self.models = models or AI_MODELS
        self.timeout = timeout
        
        if not self.api_key:
            logger.warning("OpenRouter API key not configured")
            
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": APP_REFERER,
            "X-Title": APP_TITLE
        }
        
        self._session = requests.Session()
        self._session.headers.update(self.headers)

    def _create_payload(self, prompt: str, model: str) -> Dict[str, Any]:
        """Creates the JSON payload for the chat completion request"""
        return {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a legal assistant. Respond only to the current query."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.5,
            "max_tokens": 2000
        }

    def call_api(self, prompt: str) -> Optional[str]:
        """Call the OpenRouter API with fallback models"""
        if not self.api_key:
            return "API key not configured. Please set OPENAI_API_KEY environment variable."
        
        for model in self.models:
            logger.info(f"Attempting API call with model: {model}")
            try:
                payload = self._create_payload(prompt, model)
                response = self._session.post(
                    f"{self.api_base}/chat/completions",
                    json=payload,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                result = response.json()
                
                if 'choices' in result and result['choices']:
                    ai_response = result['choices'][0]['message']['content'].strip()
                    logger.info(f"✅ Response received from model: {model}")
                    return ai_response
                else:
                    logger.warning(f"API call to {model} succeeded but returned no choices")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error with model {model}: {e}")
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Malformed response from model {model}: {e}")

        logger.error("❌ All configured models failed")
        return None

# Backward compatible function
def call_openrouter_api(prompt: str, api_key: str = None, api_base: str = None) -> str:
    """Backward compatible function for existing code"""
    try:
        service = OpenRouterService(
            api_key=api_key or OPENROUTER_API_KEY,
            api_base=api_base or OPENAI_API_BASE
        )
        response = service.call_api(prompt)
        return response or "I apologize, but I couldn't generate a response at this time."
    except Exception as e:
        logger.error(f"AI service call failed: {e}")
        return f"Error generating response: {str(e)}"
