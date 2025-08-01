"""AI/LLM integration service"""
import logging
import requests
from typing import Optional
from ..config import AI_MODELS, OPENROUTER_API_KEY, OPENAI_API_BASE

logger = logging.getLogger(__name__)

def call_openrouter_api(prompt: str, api_key: str = None, api_base: str = None) -> str:
    """Call OpenRouter API with fallback models"""
    if not api_key:
        api_key = OPENROUTER_API_KEY
    if not api_base:
        api_base = OPENAI_API_BASE
    
    if not api_key:
        return "I apologize, but AI features are not configured. Please set OPENAI_API_KEY environment variable."
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Legal Assistant"
    }
    
    for model in AI_MODELS:
        try:
            # Create a fresh payload for each call - no state contamination
            payload = {
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
            
            response = requests.post(api_base + "/chat/completions", headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                logger.info(f"✅ Successfully used model: {model}")
                return result['choices'][0]['message']['content'].strip()
                
        except Exception as e:
            logger.error(f"Error with model {model}: {e}")
            continue
    
    return "I'm experiencing technical difficulties. Please try again later."
