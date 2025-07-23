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
Answer the question based only on the following factual context and use a little bit of your understanding as well:
{context}

Question: {question}
Answer using as many words as possible with a serious tone like a lawyer:
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
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",  # Optional: for OpenRouter
            "X-Title": "RAG Application"  # Optional: for OpenRouter
        }
        
        payload = {
            "model": "deepseek/deepseek-chat",  # Updated model name
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 500
        }
        
        print(f"Making request to: {api_base}")
        print(f"Headers: {headers}")
        print(f"Payload model: {payload['model']}")
        
        response = requests.post(
            f"{api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code != 200:
            print(f"Error response content: {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"API call failed: {response.text}"
            )
        
        # Check if response is JSON
        content_type = response.headers.get('content-type', '')
        if 'application/json' not in content_type:
            print(f"Unexpected content type: {content_type}")
            print(f"Response content: {response.text[:500]}...")
            raise HTTPException(
                status_code=500,
                detail=f"Expected JSON response, got {content_type}"
            )
        
        result = response.json()
        
        if 'choices' not in result or not result['choices']:
            raise HTTPException(
                status_code=500,
                detail="No choices in API response"
            )
        
        return result['choices'][0]['message']['content']
        
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Use Render's port or fallback to 8000 locally
    uvicorn.run(app, host="0.0.0.0", port=port)
