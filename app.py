# app.py - Combined FastAPI with Enhanced RAG System
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import requests
import re
import logging
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional, List, Tuple, Dict, Set
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PATH = os.path.join(os.getcwd(), "my_chroma_db")
print("Checking CHROMA_PATH:", CHROMA_PATH)
if os.path.exists(CHROMA_PATH):
    print("Contents:", os.listdir(CHROMA_PATH))
else:
    print("Folder not found")

@dataclass
class SearchResult:
    """Enhanced search result with metadata"""
    content: str
    score: float
    source: str = ""
    chunk_type: str = "standard"

class EnhancedLegalRAG:
    """Enhanced RAG system specifically designed for legal scenario analysis"""
    
    def __init__(self, db, embedding_function, logger=None):
        self.db = db
        self.embedding_function = embedding_function
        self.logger = logger or logging.getLogger(__name__)
        
        # Legal domain knowledge
        self.legal_keywords = {
            'contract_law': ['contract', 'agreement', 'breach', 'performance', 'consideration', 
                            'offer', 'acceptance', 'capacity', 'legality', 'discharge'],
            'tort_law': ['negligence', 'liability', 'damages', 'duty of care', 'causation', 
                         'foreseeability', 'reasonable person', 'proximate cause'],
            'employment_law': ['employment', 'termination', 'discrimination', 'harassment', 
                               'wrongful termination', 'at-will', 'benefits', 'wage', 'overtime'],
            'property_law': ['property', 'lease', 'rent', 'tenant', 'landlord', 'title', 
                             'deed', 'easement', 'zoning', 'eminent domain'],
            'corporate_law': ['corporation', 'LLC', 'partnership', 'merger', 'acquisition', 
                              'fiduciary duty', 'board of directors', 'shareholders'],
            'ip_law': ['intellectual property', 'copyright', 'trademark', 'patent', 'license', 
                       'trade secret', 'infringement', 'fair use'],
            'regulatory': ['compliance', 'regulation', 'statute', 'rule', 'violation', 
                           'penalty', 'enforcement', 'administrative law']
        }
        
        self.legal_phrases = [
            'force majeure', 'due diligence', 'good faith', 'reasonable care',
            'material breach', 'liquidated damages', 'indemnification',
            'non-disclosure', 'confidentiality', 'fiduciary duty', 'statute of limitations',
            'jurisdiction', 'governing law', 'dispute resolution', 'termination clause',
            'trade secrets', 'non-compete', 'employment at will', 'prima facie',
            'burden of proof', 'preponderance of evidence', 'beyond reasonable doubt'
        ]

    def enhanced_similarity_search(self, query_text: str, k: int = 5) -> List[SearchResult]:
        """Multi-strategy search approach for comprehensive document retrieval"""
        all_results = []
        
        # Strategy 1: Direct semantic search
        try:
            direct_results = self.db.similarity_search_with_relevance_scores(query_text, k=k)
            for doc, score in direct_results:
                all_results.append(SearchResult(
                    content=doc.page_content,
                    score=score,
                    source=getattr(doc, 'metadata', {}).get('source', 'unknown'),
                    chunk_type='direct'
                ))
        except Exception as e:
            self.logger.warning(f"Direct search failed: {e}")

        # Strategy 2: Legal domain-specific search
        legal_domain = self._identify_legal_domain(query_text)
        if legal_domain:
            domain_keywords = self.legal_keywords.get(legal_domain, [])
            for keyword in domain_keywords[:3]:
                try:
                    domain_results = self.db.similarity_search_with_relevance_scores(keyword, k=2)
                    for doc, score in domain_results:
                        all_results.append(SearchResult(
                            content=doc.page_content,
                            score=score * 0.8,
                            source=getattr(doc, 'metadata', {}).get('source', 'unknown'),
                            chunk_type='domain_keyword'
                        ))
                except Exception as e:
                    self.logger.warning(f"Domain search for {keyword} failed: {e}")

        # Strategy 3: Entity-based search
        entities = self._extract_entities(query_text)
        for entity in entities[:3]:
            try:
                entity_results = self.db.similarity_search_with_relevance_scores(entity, k=2)
                for doc, score in entity_results:
                    all_results.append(SearchResult(
                        content=doc.page_content,
                        score=score * 0.9,
                        source=getattr(doc, 'metadata', {}).get('source', 'unknown'),
                        chunk_type='entity'
                    ))
            except Exception as e:
                self.logger.warning(f"Entity search for {entity} failed: {e}")

        # Strategy 4: Legal phrase search
        legal_phrases_found = self._extract_legal_phrases(query_text)
        for phrase in legal_phrases_found:
            try:
                phrase_results = self.db.similarity_search_with_relevance_scores(phrase, k=2)
                for doc, score in phrase_results:
                    all_results.append(SearchResult(
                        content=doc.page_content,
                        score=score * 0.95,
                        source=getattr(doc, 'metadata', {}).get('source', 'unknown'),
                        chunk_type='legal_phrase'
                    ))
            except Exception as e:
                self.logger.warning(f"Phrase search for {phrase} failed: {e}")

        return self._deduplicate_and_rank_results(all_results, k * 2)

    def _identify_legal_domain(self, text: str) -> Optional[str]:
        """Identify the primary legal domain of the query"""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in self.legal_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        return max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else None

    def _extract_entities(self, text: str) -> List[str]:
        """Extract legal entities from text"""
        entities = []
        
        patterns = {
            'companies': r'\b[A-Z][a-zA-Z\s&,]+(?:Inc\.?|LLC|Corp\.?|Ltd\.?|Company|Co\.)\b',
            'amounts': r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand))?',
            'dates': r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})\b',
            'percentages': r'\d+(?:\.\d+)?%',
            'case_numbers': r'\b\d{4}\s*WL\s*\d+|\b\d+\s*F\.\s*(?:2d|3d)\s*\d+',
            'statutes': r'\b\d+\s*U\.?S\.?C\.?\s*§?\s*\d+|\b\w+\s*Code\s*§?\s*\d+'
        }
        
        for pattern_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend(matches[:2])
        
        return entities

    def _extract_legal_phrases(self, text: str) -> List[str]:
        """Extract legal phrases from text"""
        text_lower = text.lower()
        found_phrases = []
        
        for phrase in self.legal_phrases:
            if phrase in text_lower:
                found_phrases.append(phrase)
        
        return found_phrases

    def _deduplicate_and_rank_results(self, results: List[SearchResult], max_results: int) -> List[SearchResult]:
        """Remove duplicates and rank results by relevance"""
        seen_content = {}
        for result in results:
            content_key = result.content.strip()
            if content_key not in seen_content or seen_content[content_key].score < result.score:
                seen_content[content_key] = result
        
        unique_results = list(seen_content.values())
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        return unique_results[:max_results]

    def is_scenario_question(self, text: str) -> bool:
        """Enhanced scenario detection"""
        scenario_indicators = [
            r'\bif\b', r'\bsuppose\b', r'\bassume\b', r'\bconsider\b', r'\bgiven\b',
            r'what would happen', r'what if', r'in the case', r'scenario', r'situation',
            r'company has', r'employee does', r'client wants', r'party fails',
            r'person is', r'individual has', r'organization does',
            r'breach occurs', r'contract states', r'agreement provides', r'law requires',
            r'violation of', r'dispute over', r'claim that', r'lawsuit', r'legal action'
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in scenario_indicators)

    def generate_enhanced_prompt(self, query_text: str, results: List[SearchResult]) -> str:
        """Generate context-aware prompt based on query type and results"""
        
        if self.is_scenario_question(query_text):
            return self._generate_scenario_prompt(query_text, results)
        else:
            return self._generate_standard_prompt(query_text, results)

    def _generate_scenario_prompt(self, query_text: str, results: List[SearchResult]) -> str:
        """Generate a prompt optimized for direct and professional scenario analysis."""
        context_sections = []
        
        # Group results for clarity
        direct_matches = [r for r in results if r.chunk_type == 'direct']
        supporting_context = [r for r in results if r.chunk_type != 'direct']
        
        # Primary context from the most relevant documents
        if direct_matches:
            primary_context = "\n\n---\n\n".join([r.content for r in direct_matches[:4]])
            context_sections.append(f"RELEVANT LEGAL PROVISIONS:\n{primary_context}")
        
        # Additional context for broader understanding
        if supporting_context:
            support_context = "\n\n---\n\n".join([r.content for r in supporting_context[:3]])
            context_sections.append(f"ADDITIONAL CONTEXTUAL PROVISIONS:\n{support_context}")
            
        full_context = "\n\n" + "="*50 + "\n\n".join(context_sections)

        # This new prompt is more direct and demands a conclusive answer.
        return f"""You are a legal analyst AI. Your task is to answer a specific question based on the provided legal text.

**LEGAL CONTEXT:**
{full_context}

**QUESTION:**
{query_text}

**INSTRUCTIONS:**
1.  **Direct Answer:** Begin with a direct answer to the question (e.g., "Yes, RCW 10.01.140 allows..." or "No, based on the text...").
2.  **Governing Rule:** Identify and quote the specific language from the legal context that governs the answer.
3.  **Application:** Briefly explain how the rule applies to the question.
4.  **Conclusion:** Conclude with a concise summary of the legal determination.

**RESPONSE REQUIREMENTS:**
-   **Rely exclusively on the provided `LEGAL CONTEXT`.** Do not use external knowledge.
-   If the context directly answers the question, state the answer decisively.
-   If the context does not contain the information needed to answer, state that "The provided text does not address this specific issue."
-   **Do not ask for more information or suggest alternative questions.**
-   Maintain a formal, objective, and professional tone.

**LEGAL ANALYSIS:**
"""

    def _generate_standard_prompt(self, query_text: str, results: List[SearchResult]) -> str:
        """Generate prompt for standard legal questions"""
        context_text = "\n\n---\n\n".join([r.content for r in results[:5]])
        
        return f"""As a legal expert, provide a comprehensive answer using the provided context as your primary source.

CONTEXT FROM LEGAL DOCUMENTS:
{context_text}

QUESTION: {query_text}

GUIDELINES:
- Prioritize information from the provided context above all else
- Only supplement with general legal knowledge if it directly supports or clarifies the context
- Clearly distinguish between what's stated in the documents vs. general legal principles
- Use professional legal terminology and maintain a formal tone
- Provide detailed explanations with specific references to the context
- If the context is insufficient, explicitly state what additional information would be needed

RESPONSE:"""

# Pydantic models
class Query(BaseModel):
    question: str
    use_enhanced_search: Optional[bool] = True  # Option to use enhanced search

class QueryResponse(BaseModel):
    response: Optional[str] = None
    error: Optional[str] = None
    context_found: bool = False
    search_strategy: Optional[str] = None
    result_count: Optional[int] = None
    legal_domain: Optional[str] = None

# Create FastAPI app instance
app = FastAPI(title="Enhanced Legal RAG API", description="Legal RAG with Enhanced Search", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Enhanced Legal RAG API is running"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "database_exists": os.path.exists(CHROMA_PATH),
        "database_path": CHROMA_PATH,
        "features": ["enhanced_search", "scenario_detection", "legal_domain_analysis"]
    }

def call_openrouter_api(prompt: str, api_key: str, api_base: str) -> str:
    """Make API call to OpenRouter with proper error handling"""
    try:
        if api_base.endswith('/'):
            api_base = api_base.rstrip('/')
        
        if 'openrouter.ai' in api_base and not api_base.endswith('/api/v1'):
            api_base = "https://openrouter.ai/api/v1"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Enhanced Legal RAG Application"
        }
        
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
                "max_tokens": 800  # Increased for detailed legal analysis
            }
            
            logger.info(f"Trying model: {model}")
            
            response = requests.post(
                f"{api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            content_type = response.headers.get('content-type', '')
            if 'text/html' in content_type:
                if model == models_to_try[-1]:
                    error_msg = "HTML response received instead of JSON"
                    if "error" in response.text.lower():
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
                if model == models_to_try[-1]:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"All models failed. Last error: {response.text}"
                    )
                continue
            
            if 'application/json' not in content_type:
                if model == models_to_try[-1]:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Expected JSON response, got {content_type}"
                    )
                continue
            
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                if model == models_to_try[-1]:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Invalid JSON response: {str(e)}"
                    )
                continue
            
            if 'choices' not in result or not result['choices']:
                if model == models_to_try[-1]:
                    raise HTTPException(
                        status_code=500,
                        detail="No choices in API response"
                    )
                continue
            
            response_text = result['choices'][0]['message']['content']
            logger.info(f"Success with model {model}")
            return response_text
        
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
    """Enhanced question answering with multi-strategy search"""
    try:
        query_text = query.question
        
        if not query_text or query_text.strip() == "":
            return QueryResponse(
                response=None,
                error="Question cannot be empty",
                context_found=False
            )
        
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        if not os.path.exists(CHROMA_PATH):
            return QueryResponse(
                response=None,
                error="Database not found",
                context_found=False
            )
        
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        # Initialize enhanced RAG system
        rag_system = EnhancedLegalRAG(db, embedding_function, logger)
        
        # Choose search strategy based on user preference
        if query.use_enhanced_search:
            results = rag_system.enhanced_similarity_search(query_text, k=6)
            search_strategy = "enhanced_multi_strategy"
            
            # Determine minimum score threshold based on query type
            min_score = 0.3 if rag_system.is_scenario_question(query_text) else 0.4
            legal_domain = rag_system._identify_legal_domain(query_text)
            
        else:
            # Fallback to original simple search
            simple_results = db.similarity_search_with_relevance_scores(query_text, k=3)
            results = [SearchResult(
                content=doc.page_content,
                score=score,
                source=getattr(doc, 'metadata', {}).get('source', 'unknown'),
                chunk_type='simple'
            ) for doc, score in simple_results]
            search_strategy = "simple_semantic"
            min_score = 0.5
            legal_domain = None
        
        if not results or results[0].score < min_score:
            return QueryResponse(
                response="The available documents don't contain sufficient information to analyze this question. Please provide more specific details or check if the relevant documents are in the database.",
                error=None,
                context_found=False,
                search_strategy=search_strategy,
                result_count=0,
                legal_domain=legal_domain
            )
        
        # Generate appropriate prompt
        if query.use_enhanced_search:
            formatted_prompt = rag_system.generate_enhanced_prompt(query_text, results)
        else:
            # Use original prompt template
            context_text = "\n\n---\n\n".join([r.content for r in results])
            formatted_prompt = f"""As a legal expert, provide a comprehensive answer using the provided context as your primary source.

CONTEXT FROM LEGAL DOCUMENTS:
{context_text}

QUESTION: {query_text}

GUIDELINES:
- Prioritize information from the provided context above all else
- Only supplement with general legal knowledge if it directly supports or clarifies the context
- Clearly distinguish between what's stated in the documents vs. general legal principles
- Use professional legal terminology and maintain a formal tone
- Provide detailed explanations with specific references to the context
- If the context is insufficient, explicitly state what additional information would be needed

RESPONSE:"""
        
        # Get API credentials
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        
        if not api_key:
            return QueryResponse(
                response=None,
                error="OPENAI_API_KEY environment variable is required",
                context_found=True,
                search_strategy=search_strategy,
                result_count=len(results),
                legal_domain=legal_domain
            )
        
        # Make the API call
        try:
            response_text = call_openrouter_api(formatted_prompt, api_key, api_base)
            
            logger.info(f"Successfully generated response using {search_strategy} strategy")
            
            return QueryResponse(
                response=response_text if response_text else "No response generated",
                error=None,
                context_found=True,
                search_strategy=search_strategy,
                result_count=len(results),
                legal_domain=legal_domain
            )
            
        except HTTPException as he:
            return QueryResponse(
                response=None,
                error=str(he.detail),
                context_found=True,
                search_strategy=search_strategy,
                result_count=len(results),
                legal_domain=legal_domain
            )
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return QueryResponse(
                response=None,
                error=f"API call failed: {str(e)}",
                context_found=True,
                search_strategy=search_strategy,
                result_count=len(results),
                legal_domain=legal_domain
            )
        
    except Exception as e:
        logger.error(f"Unexpected error in ask_question: {str(e)}")
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
        response_text = call_openrouter_api("Hello, this is a test of the enhanced legal RAG system.", api_key, api_base)
        return {"success": True, "response": response_text}
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug-search")
def debug_search(query: str = "contract breach"):
    """Debug endpoint to test search strategies"""
    if not os.path.exists(CHROMA_PATH):
        return {"error": "Database not found"}
    
    try:
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        rag_system = EnhancedLegalRAG(db, embedding_function, logger)
        
        # Test both search strategies
        enhanced_results = rag_system.enhanced_similarity_search(query, k=5)
        simple_results = db.similarity_search_with_relevance_scores(query, k=3)
        
        return {
            "query": query,
            "is_scenario": rag_system.is_scenario_question(query),
            "legal_domain": rag_system._identify_legal_domain(query),
            "entities": rag_system._extract_entities(query),
            "legal_phrases": rag_system._extract_legal_phrases(query),
            "enhanced_results": {
                "count": len(enhanced_results),
                "top_scores": [r.score for r in enhanced_results[:3]],
                "chunk_types": [r.chunk_type for r in enhanced_results[:5]]
            },
            "simple_results": {
                "count": len(simple_results),
                "top_scores": [score for _, score in simple_results]
            }
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
