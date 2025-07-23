# app.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import requests
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional

CHROMA_PATH = os.path.join(os.getcwd(), "my_chroma_db")
print("Checking CHROMA_PATH:", CHROMA_PATH)
if os.path.exists(CHROMA_PATH):
    print("Contents:", os.listdir(CHROMA_PATH))
else:
    print("Folder not found")

PROMPT_TEMPLATE = """
As a legal expert, provide a comprehensive answer to the question below using the provided context as your primary source.

CONTEXT:
{context}

QUESTION: {question}

GUIDELINES:
- Prioritize information from the provided context above all else
- Only supplement with general legal knowledge if it directly supports or clarifies the context
- Clearly distinguish between what's stated in the documents vs. general legal principles
- Use professional legal terminology and maintain a formal tone
- Provide detailed explanations with specific references to the context
- If the context is insufficient, explicitly state what additional information would be needed
- Use space when asked to create bullet points to make things easy to read 

RESPONSE:
"""

class Query(BaseModel):
    question: str

class QueryResponse(BaseModel):
    response: Optional[str] = None
    error: Optional[str] = None
    context_found: bool = False

# Create FastAPI app instance ONCE
app = FastAPI(title="RAG API", description="Retrieval-Augmented Generation API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:3000"] for stricter
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "RAG API is running"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "database_exists": os.path.exists(CHROMA_PATH),
        "database_path": CHROMA_PATH
    }

def call_openrouter_api(prompt: str, api_key: str, api_base: str) -> str:
    """Make API call to OpenRouter with proper error handling"""
    try:
        # Clean the API base URL
        if api_base.endswith('/'):
            api_base = api_base.rstrip('/')
        
        # Ensure we're using the correct OpenRouter URL
        if 'openrouter.ai' in api_base and not api_base.endswith('/api/v1'):
            api_base = "https://openrouter.ai/api/v1"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",  # Required for OpenRouter
            "X-Title": "RAG Application"  # Optional: for OpenRouter
        }
        
        # Try different model names that are known to work
        models_to_try = [
            "meta-llama/llama-3.2-3b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free",
            "google/gemma-2-9b-it:free",
            "deepseek/deepseek-chat"
        ]
        
        for model in models_to_try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 500
            }
            
            print(f"Trying model: {model}")
            print(f"Making request to: {api_base}/chat/completions")
            print(f"API key starts with: {api_key[:10]}...")
            
            response = requests.post(
                f"{api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            # If we get HTML, it's likely an error page - let's see what it says
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                print(f"Got HTML response for model {model}")
                print(f"Response content (first 1000 chars): {response.text[:1000]}")
                
                # If it's the last model to try, raise the error
                if model == models_to_try[-1]:
                    # Try to extract error message from HTML
                    error_msg = "HTML response received instead of JSON"
                    if "error" in response.text.lower():
                        # Simple extraction - look for common error patterns
                        import re
                        error_pattern = r'error["\s:]*([^"<>\n]{10,100})'
                        match = re.search(error_pattern, response.text, re.IGNORECASE)
                        if match:
                            error_msg = f"API Error: {match.group(1).strip()}"
                    
                    raise HTTPException(
                        status_code=500,
                        detail=f"{error_msg}. Status: {response.status_code}"
                    )
                continue
            
            if response.status_code != 200:
                print(f"Error response for model {model}: {response.text}")
                if model == models_to_try[-1]:  # Last model to try
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"All models failed. Last error: {response.text}"
                    )
                continue
            
            # Check if response is JSON
            if 'application/json' not in content_type:
                print(f"Unexpected content type for model {model}: {content_type}")
                if model == models_to_try[-1]:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Expected JSON response, got {content_type}"
                    )
                continue
            
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                print(f"JSON decode error for model {model}: {e}")
                if model == models_to_try[-1]:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Invalid JSON response: {str(e)}"
                    )
                continue
            
            if 'choices' not in result or not result['choices']:
                print(f"No choices in response for model {model}")
                if model == models_to_try[-1]:
                    raise HTTPException(
                        status_code=500,
                        detail="No choices in API response"
                    )
                continue
            
            # Success! Return the response
            response_text = result['choices'][0]['message']['content']
            print(f"Success with model {model}: {response_text[:100]}...")
            return response_text
        
        # This shouldn't be reached, but just in case
        raise HTTPException(
            status_code=500,
            detail="All models failed to generate a response"
        )
        
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="API request timed out")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/ask", response_model=QueryResponse)
def ask_question(query: Query):
    try:
        query_text = query.question
        
        if not query_text or query_text.strip() == "":
            return QueryResponse(
                response=None,
                error="Question cannot be empty",
                context_found=False
            )
        
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Check if Chroma database exists
        if not os.path.exists(CHROMA_PATH):
            return QueryResponse(
                response=None,
                error="Database not found",
                context_found=False
            )
        
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        
        if not results or results[0][1] < 0.5:
            return QueryResponse(
                response="Nothing in the records.",
                error=None,
                context_found=False
            )
        
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        
        # Simple string formatting - avoid ChatPromptTemplate complications
        formatted_prompt = PROMPT_TEMPLATE.format(
            context=context_text, 
            question=query_text
        )
        
        # Debug: Print the formatted prompt type and length
        print(f"Formatted prompt type: {type(formatted_prompt)}")
        print(f"Formatted prompt length: {len(formatted_prompt)}")
        
        # Check for required environment variables
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        
        if not api_key:
            return QueryResponse(
                response=None,
                error="OPENAI_API_KEY environment variable is required",
                context_found=True
            )
        
        if not api_base:
            return QueryResponse(
                response=None,
                error="OPENAI_API_BASE environment variable is required",
                context_found=True
            )
        
        # Make the API call using requests
        try:
            response_text = call_openrouter_api(formatted_prompt, api_key, api_base)
            
            print(f"Successfully got response: {response_text[:100]}...")
            
            return QueryResponse(
                response=response_text if response_text else "No response generated",
                error=None,
                context_found=True
            )
            
        except HTTPException as he:
            return QueryResponse(
                response=None,
                error=str(he.detail),
                context_found=True
            )
        except Exception as e:
            print(f"API call failed: {str(e)}")
            print(f"Error type: {type(e)}")
            return QueryResponse(
                response=None,
                error=f"API call failed: {str(e)}",
                context_found=True
            )
        
    except Exception as e:
        return QueryResponse(
            response=None,
            error=f"An error occurred: {str(e)}",
            context_found=False
        )

@app.get("/test-api")
def test_api():
    """Test endpoint to check API connectivity"""
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}
    
    try:
        response_text = call_openrouter_api("Hello, this is a test.", api_key, api_base)
        return {"success": True, "response": response_text}
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug-api")
def debug_api():
    """Debug endpoint to check API configuration"""
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    
    # Make a simple request to see what we get
    headers = {
        "Authorization": f"Bearer {api_key}" if api_key else "Bearer MISSING",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "RAG Application Debug"
    }
    
    try:
        response = requests.get(
            f"{api_base}/models",  # Get available models
            headers=headers,
            timeout=10
        )
        
        return {
            "api_key_set": bool(api_key),
            "api_key_prefix": api_key[:10] + "..." if api_key else "Not set",
            "api_base": api_base,
            "models_endpoint_status": response.status_code,
            "models_content_type": response.headers.get('content-type'),
            "models_response_preview": response.text[:500] if response.text else "No content"
        }
    except Exception as e:
        return {
            "api_key_set": bool(api_key),
            "api_base": api_base,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Use Render's port or fallback to 8000 locally
    uvicorn.run(app, host="0.0.0.0", port=port)
