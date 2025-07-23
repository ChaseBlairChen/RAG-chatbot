# app.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.prompts import ChatPromptTemplate  # Not needed anymore
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
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
        api_base = os.environ.get("OPENAI_API_BASE")
        
        if not api_key or not api_base:
            return QueryResponse(
                response=None,
                error="API configuration missing",
                context_found=True
            )
        
        # Use OpenAI client directly instead of LangChain wrapper
        try:
            import openai
            
            # Set up the OpenAI client for OpenRouter
            client = openai.OpenAI(
                api_key=api_key,
                base_url=api_base
            )
            
            print(f"Making API call to {api_base} with model deepseek/deepseek-chat-v3-0324:free")
            print(f"API key starts with: {api_key[:10]}...")
            
            # Make the API call directly
            response = client.chat.completions.create(
                model="deepseek/deepseek-chat-v3-0324:free",
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )
            
            print(f"Response type: {type(response)}")
            print(f"Response: {response}")
            
            # Handle different response types
            if hasattr(response, 'choices') and response.choices:
                response_text = response.choices[0].message.content
                print(f"Extracted response text: {response_text[:100]}...")
            elif isinstance(response, dict) and 'choices' in response:
                response_text = response['choices'][0]['message']['content']
                print(f"Dict response - Extracted text: {response_text[:100]}...")
            elif isinstance(response, str):
                response_text = response
                print(f"String response: {response_text[:100]}...")
            else:
                print(f"Unexpected response format: {response}")
                response_text = str(response)
            
        except ImportError:
            return QueryResponse(
                response=None,
                error="OpenAI package not installed. Please install: pip install openai",
                context_found=True
            )
        except Exception as e:
            print(f"Direct OpenAI call failed: {str(e)}")
            print(f"Error type: {type(e)}")
            return QueryResponse(
                response=None,
                error=f"API call failed: {str(e)}",
                context_found=True
            )
        
        return QueryResponse(
            response=response_text if response_text else None,
            error=None,
            context_found=True
        )
        
    except Exception as e:
        return QueryResponse(
            response=None,
            error=f"An error occurred: {str(e)}",
            context_found=False
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Use Render's port or fallback to 8000 locally
    uvicorn.run(app, host="0.0.0.0", port=port)
