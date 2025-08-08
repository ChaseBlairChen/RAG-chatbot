"""
AI/LLM Integration Service

This module provides a robust and flexible service for interacting with
the OpenRouter API. It includes a fallback mechanism to try multiple
models and handles various API errors gracefully.
"""
import logging
import requests
from typing import Optional, Dict, Any, List

from ..config import AI_MODELS, OPENROUTER_API_KEY, OPENAI_API_BASE, APP_REFERER, APP_TITLE

# Configure logging for the module
logger = logging.getLogger(__name__)

class OpenRouterService:
    """
    A service for making API calls to OpenRouter with a fallback model mechanism.
    """

    def __init__(
        self,
        api_key: str = None,
        api_base: str = None,
        models: List[str] = None,
        referer: str = None,
        title: str = None,
        timeout: int = 60
    ):
        """
        Initializes the OpenRouterService with configuration.

        Args:
            api_key: The API key for OpenRouter.
            api_base: The base URL for the OpenRouter API.
            models: A list of models to try in order.
            referer: The HTTP referer for the API call.
            title: The X-Title header for the API call.
            timeout: The request timeout in seconds.
        """
        self.api_key = api_key or OPENROUTER_API_KEY
        self.api_base = api_base or OPENAI_API_BASE
        self.models = models or AI_MODELS
        self.timeout = timeout
        
        if not self.api_key:
            logger.warning("API key is not configured. Please set the OPENROUTER_API_KEY.")
            
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": referer or APP_REFERER,
            "X-Title": title or APP_TITLE
        }
        
        self._session = requests.Session()
        self._session.headers.update(self.headers)

    def _create_payload(self, prompt: str, model: str, query_type: str = "general", response_style: str = "balanced") -> Dict[str, Any]:
        """
        Creates the JSON payload for the chat completion request with advanced prompt engineering.
        """
        # Enhanced system prompts based on query type
        system_prompts = {
            "legal_analysis": """You are an expert legal assistant with deep knowledge of US law, regulations, and legal procedures. 
Your responses should be:
- Accurate and well-reasoned based on legal sources
- Clear and accessible to both legal professionals and laypersons
- Comprehensive but concise
- Properly cited when referencing specific laws or cases
- Cautious about providing legal advice (disclaimers when appropriate)

Always structure your responses with clear headings, bullet points for key points, and actionable insights when possible.""",
            
            "immigration": """You are a specialized immigration law expert with extensive knowledge of:
- USCIS forms and procedures
- Immigration court processes
- Asylum and refugee law
- Family-based immigration
- Employment-based immigration
- Removal proceedings

Provide practical, accurate guidance while being clear about limitations and the need for professional legal counsel.""",
            
            "statutory": """You are a statutory interpretation expert. When analyzing statutes:
- Focus on plain meaning first
- Consider legislative intent and history
- Reference relevant case law interpretations
- Identify key terms and definitions
- Note any ambiguities or areas requiring judicial interpretation
- Provide practical applications when possible""",
            
            "comprehensive_analysis": """You are conducting a comprehensive legal analysis. Structure your response with:
1. Executive Summary
2. Key Legal Issues
3. Relevant Laws and Regulations
4. Analysis and Interpretation
5. Practical Implications
6. Recommendations
7. Risk Assessment
8. Next Steps

Be thorough but organized, and always cite your sources.""",
            
            "general": """You are a knowledgeable legal assistant. Provide helpful, accurate information while:
- Being clear about the limitations of your advice
- Suggesting when professional legal counsel is needed
- Providing practical, actionable information
- Citing relevant sources when possible
- Maintaining a helpful, professional tone"""
        }
        
        # Select appropriate system prompt
        system_prompt = system_prompts.get(query_type, system_prompts["general"])
        
        # Adjust parameters based on response style
        style_params = {
            "concise": {"temperature": 0.3, "max_tokens": 1000},
            "balanced": {"temperature": 0.5, "max_tokens": 2000},
            "detailed": {"temperature": 0.7, "max_tokens": 3000},
            "comprehensive": {"temperature": 0.4, "max_tokens": 4000}
        }
        
        params = style_params.get(response_style, style_params["balanced"])
        
        return {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": params["temperature"],
            "max_tokens": params["max_tokens"],
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        }

    def call_api(self, prompt: str) -> Optional[str]:
        """
        Calls the OpenRouter API with a prompt, attempting a list of models in order.

        Args:
            prompt: The user's query.

        Returns:
            The AI's response as a string, or None if all models fail.
        """
        if not self.api_key:
            return "API key not configured. Please set OPENAI_API_KEY environment variable."
        
        for model in self.models:
            logger.info(f"Attempting to call API with model: {model}")
            try:
                payload = self._create_payload(prompt, model)
                response = self._session.post(
                    f"{self.api_base}/chat/completions",
                    json=payload,
                    timeout=self.timeout
                )
                
                # Check for HTTP errors
                response.raise_for_status()

                result = response.json()
                
                # Validate the response structure
                if 'choices' in result and result['choices']:
                    ai_response = result['choices'][0]['message']['content'].strip()
                    logger.info(f"✅ Successfully received response from model: {model}")
                    return ai_response
                else:
                    logger.warning(f"API call to {model} succeeded but returned no choices.")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Network or HTTP error with model {model}: {e}")
                
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Malformed response from model {model}: {e}")

        logger.error("❌ All configured models failed to return a valid response.")
        return None

# FIXED: Add backward compatible function that was missing
def call_openrouter_api(prompt: str, api_key: str = None, api_base: str = None, timeout: int = 60) -> str:
    """
    Backward compatible function for existing code
    This function was missing and causing import errors
    """
    try:
        service = OpenRouterService(
            api_key=api_key or OPENROUTER_API_KEY,
            api_base=api_base or OPENAI_API_BASE,
            timeout=timeout
        )
        response = service.call_api(prompt)
        return response or "I apologize, but I couldn't generate a response at this time."
    except Exception as e:
        logger.error(f"AI service call failed: {e}")
        return f"Error generating response: {str(e)}"

# --- Usage Example ---
if __name__ == "__main__":
    # In a real application, this would be handled by a dependency injection framework
    # or by a factory function.
    try:
        service = OpenRouterService()
        user_prompt = "What are the key differences between a contract and an agreement?"
        response = service.call_api(user_prompt)

        if response:
            print("AI Response:")
            print(response)
        else:
            print("Failed to get a response from the AI service.")
            
    except ValueError as e:
        print(f"Configuration error: {e}")
