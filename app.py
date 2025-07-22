# app.py
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import os
import json
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import Optional

CHROMA_PATH = os.path.join(os.path.expanduser("~"), "my_chroma_db")

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

app = FastAPI()

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

        embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
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
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
            context=context_text, 
            question=query_text
        )

        # Check for required environment variables
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE")
        
        if not api_key or not api_base:
            return QueryResponse(
                response=None,
                error="API configuration missing",
                context_found=True
            )

        llm = ChatOpenAI(
            model="deepseek/deepseek-chat-v3-0324:free",
            temperature=0.2,
            max_tokens=200,
            openai_api_key=api_key,
            openai_api_base=api_base
        )
        
        response = llm.predict(prompt)
        
        return QueryResponse(
            response=response if response else None,
            error=None,
            context_found=True
        )
        
    except Exception as e:
        return QueryResponse(
            response=None,
            error=f"An error occurred: {str(e)}",
            context_found=False
        )

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "database_exists": os.path.exists(CHROMA_PATH)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
