# --- app.py ---
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import requests
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional, List, Dict, Tuple
import logging
import uuid
from datetime import datetime, timedelta
# --- AI Agent Imports ---
from enum import Enum
from dataclasses import dataclass
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
# --- End AI Agent Imports ---
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# --- SUGGESTION: Match the new database folder name ---
CHROMA_PATH = os.path.join(os.getcwd(), "chromadb-database") # Changed from "my_chroma_db"
# --- END SUGGESTION ---
logger.info(f"Checking CHROMA_PATH: {CHROMA_PATH}")
if os.path.exists(CHROMA_PATH):
    logger.info(f"Database contents: {os.listdir(CHROMA_PATH)}")
else:
    logger.warning("Database folder not found!")
# In-memory conversation storage (in production, use Redis or a database)
conversations: Dict[str, Dict] = {}
# --- SUGGESTION 1: Alias Map for Entity Resolution ---
# Define known aliases or colloquial terms and their formal counterparts
ENTITY_ALIASES = {
    "one big beautiful bill": "Inflation Reduction Act",
    "inflation reduction act": "Inflation Reduction Act", # Ensure consistent casing
    "ira": "Inflation Reduction Act",
    "lower energy costs act": "H.R.1 - Lower Energy Costs Act",
    # Add more aliases as discovered
}
# --- END SUGGESTION 1 ---
def cleanup_old_conversations():
    """Remove conversations older than 24 hours"""
    cutoff_time = datetime.now() - timedelta(hours=24)
    expired_sessions = [
        session_id for session_id, data in conversations.items()
        if data.get('last_updated', datetime.min) < cutoff_time
    ]
    for session_id in expired_sessions:
        del conversations[session_id]
    logger.info(f"Cleaned up {len(expired_sessions)} expired conversations")
def get_or_create_session(session_id: Optional[str] = None) -> str:
    """Get existing session or create new one"""
    cleanup_old_conversations()
    if session_id and session_id in conversations:
        conversations[session_id]['last_updated'] = datetime.now()
        return session_id
    # Create new session
    new_session_id = str(uuid.uuid4())
    conversations[new_session_id] = {
        'messages': [],
        'created_at': datetime.now(),
        'last_updated': datetime.now()
    }
    logger.info(f"Created new conversation session: {new_session_id}")
    return new_session_id
def add_to_conversation(session_id: str, role: str, content: str, sources: Optional[List] = None):
    """Add a message to the conversation history"""
    if session_id not in conversations:
        conversations[session_id] = {
            'messages': [],
            'created_at': datetime.now(),
            'last_updated': datetime.now()
        }
    message = {
        'role': role,
        'content': content,
        'timestamp': datetime.now().isoformat(),
        'sources': sources or []
    }
    conversations[session_id]['messages'].append(message)
    conversations[session_id]['last_updated'] = datetime.now()
    # Keep only last 20 messages to prevent memory issues
    if len(conversations[session_id]['messages']) > 20:
        conversations[session_id]['messages'] = conversations[session_id]['messages'][-20:]
def get_conversation_context(session_id: str, max_messages: int = 10) -> str:
    """Get recent conversation history as context"""
    if session_id not in conversations:
        return ""
    messages = conversations[session_id]['messages'][-max_messages:]
    context_parts = []
    for msg in messages:
        role = msg['role'].upper()
        content = msg['content']
        # Truncate very long messages
        if len(content) > 1000:
            content = content[:1000] + "..."
        context_parts.append(f"{role}: {content}")
    return "\n".join(context_parts)
def parse_multiple_questions(query_text: str) -> list:
    """Parse multiple questions from a single input"""
    if not query_text or not query_text.strip():
        return [""]
    questions = []
    query_text = query_text.strip()
    question_marks = query_text.count('?')
    if question_marks > 1:
        potential_questions = query_text.split('?')
        for q in potential_questions:
            q = q.strip()
            if q and len(q) > 5:
                if not q.endswith('?'):
                    q += '?'
                questions.append(q)
    elif re.search(r'(?:^|\n)\s*\d+[\.\)]\s*.+', query_text, re.MULTILINE):
        numbered_pattern = r'(?:^|\n)\s*\d+[\.\)]\s*(.+?)(?=(?:\n\s*\d+[\.\)])|$)'
        numbered_matches = re.findall(numbered_pattern, query_text, re.MULTILINE | re.DOTALL)
        for match in numbered_matches:
            match = match.strip()
            if match and len(match) > 5:
                if not match.endswith('?') and '?' not in match:
                    match += '?'
                questions.append(match)
    if not questions:
        questions = [query_text]
    return questions
def enhanced_retrieval(db, query_text: str, k: int = 5):
    """Enhanced retrieval with better scoring and fallback options"""
    try:
        logger.info(f"Searching for: '{query_text}' with k={k}")
        results = []
        try:
            results = db.similarity_search_with_relevance_scores(query_text, k=k*3)
            logger.info(f"Similarity search returned {len(results)} results")
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            try:
                basic_results = db.similarity_search(query_text, k=k*2)
                results = [(doc, 0.5) for doc in basic_results]
                logger.info(f"Basic search returned {len(results)} results")
            except Exception as e2:
                logger.error(f"Basic search also failed: {e2}")
                return []
        if not results:
            logger.warning("No results found, trying broader search")
            words = query_text.split()
            if len(words) > 1:
                for word in words:
                    if len(word) > 3:
                        try:
                            word_results = db.similarity_search_with_relevance_scores(word, k=2)
                            results.extend(word_results)
                            if len(results) >= k:
                                break
                        except:
                            continue
        filtered_results = []
        seen_content = set()
        for doc, score in results:
            content_preview = doc.page_content[:100].strip()
            content_hash = hash(content_preview)
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            # --- SUGGESTION 4: Increase relevance threshold ---
            if score >= 0.3: # Increased from 0.1
                filtered_results.append((doc, score))
                logger.info(f"Included result with score {score:.3f}: {content_preview[:50]}...")
        if not filtered_results and results:
            logger.warning("No results met threshold, including top results anyway")
            filtered_results = results[:k]
        final_results = filtered_results[:k]
        logger.info(f"Returning {len(final_results)} filtered results")
        return final_results
    except Exception as e:
        logger.error(f"Enhanced retrieval failed completely: {e}")
        return []
def create_enhanced_context(results, questions: list) -> tuple:
    """Create context that's optimized for multiple questions and return source info"""
    if not results:
        return "No relevant context found.", []
    context_parts = []
    source_info = []
    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get('source', 'Unknown')
        file_name = doc.metadata.get('file_name', os.path.basename(source) if source != 'Unknown' else 'Unknown')
        page = doc.metadata.get('page_number', '')
        page_info = f" (Page {page})" if page else ""
        display_source = file_name if file_name != 'Unknown' else source
        content = doc.page_content
        if len(content) > 500:
            content = content[:500] + "..."
        # Use document name as the identifier instead of SOURCE X
        context_part = f"[{display_source}{page_info}] (Relevance: {score:.2f}):\n{content}"
        context_parts.append(context_part)
        source_info.append({
            'id': i+1,
            'file_name': display_source,
            'page': page,
            'relevance': score,
            'full_path': source
        })
    context_text = "\n" + "\n".join(context_parts)
    return context_text, source_info
# --- SUGGESTION 3 & 4: Enhanced Prompt Template ---
# Modified prompt template to include conversation history and instructions for clarification and relevance filtering
ENHANCED_PROMPT_TEMPLATE = """You are a helpful assistant engaged in an ongoing conversation. Answer the current question using the provided context sources and conversation history.
IMPORTANT INSTRUCTIONS FOR RESPONSE:
1.  **Answer Strictly from Context:** Base your answer primarily and strictly on the provided CURRENT CONTEXT and CONVERSATION HISTORY. If the context contains no information related to the query, explicitly state that.
2.  **Clarification Dialogue:** If the question is ambiguous or refers to a term that could have multiple meanings (e.g., 'the bill'), and the context contains information about multiple potential referents, ask a clarifying question to the user before providing a specific answer. For example:
    User: "What does the bill say about tax credits?"
    Assistant: "Could you please specify which bill you are referring to? Are you asking about the Inflation Reduction Act, the Infrastructure Investment and Jobs Act, or another specific bill?"
3.  **Citation Format:** When citing information, use the document name format shown in brackets, for example [RCW 10.01.240.pdf] or [RCW 10.01.240.pdf (Page 1)] - do NOT use generic SOURCE numbers.
CONVERSATION HISTORY:
{conversation_history}
CURRENT CONTEXT:
{context}
CURRENT QUESTION: {questions}
Please provide a helpful answer based on the context above and the conversation history. When you reference information, cite it using the document name in brackets as shown in the context (e.g., [document_name.pdf] or [document_name.pdf (Page X)]).
If the user is asking a follow-up question or referring to something mentioned earlier in the conversation, acknowledge that context in your response.
RESPONSE:"""
# --- END SUGGESTION 3 & 4 ---
class Query(BaseModel):
    question: str
    session_id: Optional[str] = None
class QueryResponse(BaseModel):
    response: Optional[str] = None
    error: Optional[str] = None
    context_found: bool = False
    sources: Optional[list] = None
    session_id: str
class ConversationHistory(BaseModel):
    session_id: str
    messages: List[Dict]
    created_at: str
    last_updated: str

# ==============================================================================
# STEP 2: ADD THE UNIVERSAL AI AGENT CODE
# ==============================================================================
# Add this after your imports and before the FastAPI app creation
# (Already imported above: import re, import spacy, from sentence_transformers import SentenceTransformer, import numpy as np, from typing import List, Dict, Tuple, Optional, Set, from dataclasses import dataclass, from enum import Enum)

class SearchStrategy(Enum):
    DIRECT = "direct"
    DECOMPOSED = "decomposed" 
    EXPANDED = "expanded"
    CONTEXTUAL = "contextual"
    FUZZY = "fuzzy"
    CONCEPTUAL = "conceptual"

@dataclass
class SearchAttempt:
    query: str
    strategy: SearchStrategy
    results_count: int
    max_relevance: float
    success: bool

@dataclass
class IntelligentSearchResult:
    original_query: str
    successful_strategy: Optional[SearchStrategy]
    final_query: str
    results: List[Tuple]
    search_attempts: List[SearchAttempt]
    confidence: float
    explanation: str

class UniversalRAGAgent:
    """Universal AI Agent that makes ANY RAG system smarter"""
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy model not found - some features will be limited")
            self.nlp = None
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            self.sentence_model = None
        # Universal patterns for understanding query intent
        self.intent_patterns = {
            'definition': [r'\bwhat is\b', r'\bdefine\b', r'\bdefinition of\b', r'\bmeaning of\b'],
            'procedure': [r'\bhow to\b', r'\bsteps\b', r'\bprocess\b', r'\bprocedure\b'],
            'comparison': [r'\bcompare\b', r'\bdifference\b', r'\bversus\b', r'\bvs\b', r'\brather than\b'],
            'causal': [r'\bwhy\b', r'\bcause\b', r'\breason\b', r'\bresult\b', r'\bdue to\b'],
            'temporal': [r'\bwhen\b', r'\bbefore\b', r'\bafter\b', r'\bduring\b', r'\btimeline\b'],
            'location': [r'\bwhere\b', r'\blocation\b', r'\bplace\b'],
            'quantitative': [r'\bhow many\b', r'\bhow much\b', r'\bamount\b', r'\bnumber\b'],
            'conditional': [r'\bif\b', r'\bunless\b', r'\bwhen.*then\b', r'\bunder.*circumstances\b']
        }

    def intelligent_search(self, db, query: str, k: int = 5) -> IntelligentSearchResult:
        """Universal intelligent search that tries multiple strategies adaptively"""
        logger.info(f"🧠 AI Agent starting intelligent search for: '{query}'")
        search_attempts = []
        # Strategy 1: Direct search (baseline)
        direct_results = self._try_direct_search(db, query, k)
        attempt = SearchAttempt(
            query=query,
            strategy=SearchStrategy.DIRECT,
            results_count=len(direct_results),
            max_relevance=max([score for _, score in direct_results], default=0.0),
            success=len(direct_results) > 0 and max([score for _, score in direct_results], default=0.0) > 0.3
        )
        search_attempts.append(attempt)
        if attempt.success:
            logger.info(f"✅ Direct search succeeded with {len(direct_results)} results")
            return IntelligentSearchResult(
                original_query=query,
                successful_strategy=SearchStrategy.DIRECT,
                final_query=query,
                results=direct_results,
                search_attempts=search_attempts,
                confidence=0.9,
                explanation="Direct search found relevant results"
            )
        # Strategy 2: Query decomposition for complex questions
        decomposed_results = self._try_decomposed_search(db, query, k)
        attempt = SearchAttempt(
            query="[decomposed queries]",
            strategy=SearchStrategy.DECOMPOSED,
            results_count=len(decomposed_results),
            max_relevance=max([score for _, score in decomposed_results], default=0.0),
            success=len(decomposed_results) > 0 and max([score for _, score in decomposed_results], default=0.0) > 0.3
        )
        search_attempts.append(attempt)
        if attempt.success:
            logger.info(f"✅ Decomposed search succeeded with {len(decomposed_results)} results")
            return IntelligentSearchResult(
                original_query=query,
                successful_strategy=SearchStrategy.DECOMPOSED,
                final_query="[multiple sub-queries]",
                results=decomposed_results,
                search_attempts=search_attempts,
                confidence=0.8,
                explanation="Query was broken down into components for better matching"
            )
        # Strategy 3: Intelligent query expansion
        expanded_results = self._try_expanded_search(db, query, k)
        attempt = SearchAttempt(
            query="[expanded with synonyms/related terms]",
            strategy=SearchStrategy.EXPANDED,
            results_count=len(expanded_results),
            max_relevance=max([score for _, score in expanded_results], default=0.0),
            success=len(expanded_results) > 0 and max([score for _, score in expanded_results], default=0.0) > 0.25
        )
        search_attempts.append(attempt)
        if attempt.success:
            logger.info(f"✅ Expanded search succeeded with {len(expanded_results)} results")
            return IntelligentSearchResult(
                original_query=query,
                successful_strategy=SearchStrategy.EXPANDED,
                final_query="[query expanded with related terms]",
                results=expanded_results,
                search_attempts=search_attempts,
                confidence=0.7,
                explanation="Query was expanded with related terms and synonyms"
            )
        # Strategy 4: Contextual search
        contextual_results = self._try_contextual_search(db, query, k)
        attempt = SearchAttempt(
            query="[with context]",
            strategy=SearchStrategy.CONTEXTUAL,
            results_count=len(contextual_results),
            max_relevance=max([score for _, score in contextual_results], default=0.0),
            success=len(contextual_results) > 0 and max([score for _, score in contextual_results], default=0.0) > 0.2
        )
        search_attempts.append(attempt)
        if attempt.success:
            logger.info(f"✅ Contextual search succeeded with {len(contextual_results)} results")
            return IntelligentSearchResult(
                original_query=query,
                successful_strategy=SearchStrategy.CONTEXTUAL,
                final_query="[with contextual understanding]",
                results=contextual_results,
                search_attempts=search_attempts,
                confidence=0.6,
                explanation="Used contextual clues to find relevant information"
            )
        # Strategy 5: Fuzzy/partial matching
        fuzzy_results = self._try_fuzzy_search(db, query, k)
        attempt = SearchAttempt(
            query="[fuzzy matching]",
            strategy=SearchStrategy.FUZZY,
            results_count=len(fuzzy_results),
            max_relevance=max([score for _, score in fuzzy_results], default=0.0),
            success=len(fuzzy_results) > 0
        )
        search_attempts.append(attempt)
        if attempt.success:
            logger.info(f"✅ Fuzzy search succeeded with {len(fuzzy_results)} results")
            return IntelligentSearchResult(
                original_query=query,
                successful_strategy=SearchStrategy.FUZZY,
                final_query="[fuzzy/partial matching]",
                results=fuzzy_results,
                search_attempts=search_attempts,
                confidence=0.5,
                explanation="Found results using fuzzy matching for potential typos or variations"
            )
        # Strategy 6: Conceptual search
        conceptual_results = self._try_conceptual_search(db, query, k)
        attempt = SearchAttempt(
            query="[conceptual similarity]",
            strategy=SearchStrategy.CONCEPTUAL,
            results_count=len(conceptual_results),
            max_relevance=max([score for _, score in conceptual_results], default=0.0),
            success=len(conceptual_results) > 0
        )
        search_attempts.append(attempt)
        if conceptual_results:
            logger.info(f"⚠️ Only conceptual search found results - low confidence")
            return IntelligentSearchResult(
                original_query=query,
                successful_strategy=SearchStrategy.CONCEPTUAL,
                final_query="[conceptual matching]",
                results=conceptual_results,
                search_attempts=search_attempts,
                confidence=0.3,
                explanation="Found potentially related content using conceptual similarity"
            )
        # All strategies failed
        logger.warning(f"❌ All search strategies failed for query: '{query}'")
        return IntelligentSearchResult(
            original_query=query,
            successful_strategy=None,
            final_query=query,
            results=[],
            search_attempts=search_attempts,
            confidence=0.0,
            explanation="No relevant content found despite trying multiple search strategies"
        )

    def _try_direct_search(self, db, query: str, k: int) -> List[Tuple]:
        try:
            results = db.similarity_search_with_relevance_scores(query, k=k)
            logger.debug(f"Direct search: {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Direct search failed: {e}")
            return []

    def _try_decomposed_search(self, db, query: str, k: int) -> List[Tuple]:
        try:
            sub_queries = self._decompose_query(query)
            if len(sub_queries) <= 1:
                return []
            all_results = []
            for sub_query in sub_queries:
                try:
                    sub_results = db.similarity_search_with_relevance_scores(sub_query, k=max(1, k//len(sub_queries)))
                    all_results.extend(sub_results)
                    logger.debug(f"Sub-query '{sub_query}': {len(sub_results)} results")
                except Exception as e:
                    logger.debug(f"Sub-query '{sub_query}' failed: {e}")
                    continue
            unique_results = self._deduplicate_results(all_results)
            return unique_results[:k]
        except Exception as e:
            logger.error(f"Decomposed search failed: {e}")
            return []

    def _decompose_query(self, query: str) -> List[str]:
        sub_queries = []
        # Split on conjunctions
        if ' and ' in query.lower():
            parts = re.split(r'\s+and\s+', query, flags=re.IGNORECASE)
            sub_queries.extend([part.strip() for part in parts if part.strip()])
        # Split on question patterns
        if '?' in query:
            questions = [q.strip() + '?' for q in query.split('?') if q.strip()]
            sub_queries.extend(questions)
        # Extract key noun phrases if spaCy is available
        if self.nlp:
            try:
                doc = self.nlp(query)
                noun_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 4]
                sub_queries.extend(noun_phrases)
            except Exception as e:
                logger.debug(f"spaCy decomposition failed: {e}")
        return list(set([sq for sq in sub_queries if len(sq) > 3]))

    def _try_expanded_search(self, db, query: str, k: int) -> List[Tuple]:
        try:
            expanded_queries = self._expand_query(query)
            all_results = []
            for exp_query in expanded_queries:
                try:
                    results = db.similarity_search_with_relevance_scores(exp_query, k=max(1, k//len(expanded_queries)))
                    all_results.extend(results)
                    logger.debug(f"Expanded query '{exp_query}': {len(results)} results")
                except Exception as e:
                    logger.debug(f"Expanded query '{exp_query}' failed: {e}")
                    continue
            unique_results = self._deduplicate_results(all_results)
            return unique_results[:k]
        except Exception as e:
            logger.error(f"Expanded search failed: {e}")
            return []

    def _expand_query(self, query: str) -> List[str]:
        expansions = [query]
        intent = self._detect_intent(query)
        if intent == 'definition':
            key_terms = self._extract_key_terms(query)
            for term in key_terms[:2]:
                expansions.extend([
                    f"definition {term}",
                    f"what is {term}",
                    f"{term} means",
                    f"explanation of {term}"
                ])
        elif intent == 'procedure':
            key_terms = self._extract_key_terms(query)
            for term in key_terms[:2]:
                expansions.extend([
                    f"how to {term}",
                    f"steps for {term}",
                    f"process of {term}",
                    f"{term} procedure"
                ])
        elif intent == 'comparison':
            key_terms = self._extract_key_terms(query)
            if len(key_terms) >= 2:
                expansions.extend([
                    f"difference between {key_terms[0]} {key_terms[1]}",
                    f"{key_terms[0]} versus {key_terms[1]}",
                    f"compare {key_terms[0]} {key_terms[1]}"
                ])
        keywords = self._extract_keywords(query)
        if len(keywords) > 1:
            expansions.append(' '.join(keywords[:4]))
        return list(set(expansions))[:8]

    def _try_contextual_search(self, db, query: str, k: int) -> List[Tuple]:
        try:
            contextual_query = self._resolve_context(query)
            if contextual_query != query:
                results = db.similarity_search_with_relevance_scores(contextual_query, k=k)
                logger.debug(f"Contextual search '{contextual_query}': {len(results)} results")
                return results
            return []
        except Exception as e:
            logger.error(f"Contextual search failed: {e}")
            return []

    def _resolve_context(self, query: str) -> str:
        resolved = query
        # Would integrate conversation history here
        return resolved

    def _try_fuzzy_search(self, db, query: str, k: int) -> List[Tuple]:
        try:
            fuzzy_queries = self._generate_fuzzy_variations(query)
            all_results = []
            for fuzzy_query in fuzzy_queries:
                try:
                    results = db.similarity_search_with_relevance_scores(fuzzy_query, k=max(1, k//len(fuzzy_queries)))
                    all_results.extend(results)
                    logger.debug(f"Fuzzy query '{fuzzy_query}': {len(results)} results")
                except Exception as e:
                    continue
            unique_results = self._deduplicate_results(all_results)
            return unique_results[:k]
        except Exception as e:
            logger.error(f"Fuzzy search failed: {e}")
            return []

    def _generate_fuzzy_variations(self, query: str) -> List[str]:
        variations = []
        # Remove/add common punctuation
        variations.append(re.sub(r'[^\w\s]', '', query))
        variations.append(re.sub(r'[^\w\s]', ' ', query))
        # Handle common abbreviations
        words = query.split()
        for i, word in enumerate(words):
            if word.lower() in ['hr', 'h.r.', 'h.r']:
                new_words = words.copy()
                new_words[i] = 'house resolution'
                variations.append(' '.join(new_words))
        return list(set(variations))

    def _try_conceptual_search(self, db, query: str, k: int) -> List[Tuple]:
        try:
            conceptual_queries = self._generate_conceptual_queries(query)
            all_results = []
            for concept_query in conceptual_queries:
                try:
                    results = db.similarity_search_with_relevance_scores(concept_query, k=max(1, k//len(conceptual_queries)))
                    adjusted_results = [(doc, score * 0.7) for doc, score in results]
                    all_results.extend(adjusted_results)
                    logger.debug(f"Conceptual query '{concept_query}': {len(results)} results")
                except Exception as e:
                    continue
            unique_results = self._deduplicate_results(all_results)
            return unique_results[:k]
        except Exception as e:
            logger.error(f"Conceptual search failed: {e}")
            return []

    def _generate_conceptual_queries(self, query: str) -> List[str]:
        conceptual_queries = []
        if any(word in query.lower() for word in ['bill', 'act', 'law', 'legislation']):
            conceptual_queries.extend(['legislation', 'law', 'statute', 'regulation'])
        if any(word in query.lower() for word in ['court', 'judge', 'trial']):
            conceptual_queries.extend(['legal proceeding', 'judicial', 'court system'])
        if any(word in query.lower() for word in ['penalty', 'fine', 'punishment']):
            conceptual_queries.extend(['enforcement', 'sanctions', 'violations'])
        key_terms = self._extract_key_terms(query)
        conceptual_queries.extend(key_terms)
        return list(set(conceptual_queries))[:5]

    def _detect_intent(self, query: str) -> str:
        query_lower = query.lower()
        for intent, patterns in self.intent_patterns.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                return intent
        return 'general'

    def _extract_key_terms(self, query: str) -> List[str]:
        stop_words = {'what', 'how', 'when', 'where', 'why', 'who', 'which', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but'}
        if self.nlp:
            try:
                doc = self.nlp(query)
                key_terms = []
                for ent in doc.ents:
                    key_terms.append(ent.text)
                for token in doc:
                    if token.pos_ in ['NOUN', 'PROPN'] and token.text.lower() not in stop_words:
                        key_terms.append(token.text)
                return list(set(key_terms))[:5]
            except Exception as e:
                logger.debug(f"spaCy key term extraction failed: {e}")
        # Fallback
        words = re.findall(r'\b\w+\b', query.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        return key_terms[:5]

    def _extract_keywords(self, query: str) -> List[str]:
        stop_words = {'what', 'how', 'when', 'where', 'why', 'who', 'which', 'is', 'are', 'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', query.lower())
        return [word for word in words if word not in stop_words and len(word) > 2]

    def _deduplicate_results(self, results: List[Tuple]) -> List[Tuple]:
        if not results:
            return []
        unique_results = []
        seen_content_hashes = set()
        for doc, score in results:
            content_hash = hash(doc.page_content[:200].strip())
            if content_hash not in seen_content_hashes:
                seen_content_hashes.add(content_hash)
                unique_results.append((doc, score))
        unique_results.sort(key=lambda x: x[1], reverse=True)
        return unique_results

# ==============================================================================
# STEP 3: REPLACE THE MAIN SEARCH FUNCTION
# ==============================================================================
def enhanced_retrieval_with_universal_agent(db, query_text: str, conversation_history: List[Dict] = None, k: int = 5):
    """
    Universal AI agent that makes RAG smarter for ANY query
    """
    try:
        # Use the universal agent
        agent = UniversalRAGAgent()
        # Perform intelligent search
        search_result = agent.intelligent_search(db, query_text, k)
        logger.info(f"🧠 Universal AI Agent Result:")
        logger.info(f"  Strategy: {search_result.successful_strategy}")
        logger.info(f"  Results: {len(search_result.results)}")
        logger.info(f"  Confidence: {search_result.confidence}")
        logger.info(f"  Explanation: {search_result.explanation}")
        # Return results in the expected format
        return search_result.results, search_result
    except Exception as e:
        logger.error(f"Universal AI agent failed: {e}")
        # Fallback to basic search
        try:
            basic_results = db.similarity_search_with_relevance_scores(query_text, k=k)
            return basic_results, None
        except Exception as e2:
            logger.error(f"Even basic search failed: {e2}")
            return [], None

# ==============================================================================
# STEP 4: REPLACE THE CONTEXT CREATION FUNCTION
# ==============================================================================
def create_universal_context(results, search_result: IntelligentSearchResult, questions: list) -> tuple:
    """
    Create context with universal AI agent insights
    """
    if not results:
        if search_result and search_result.explanation:
            return f"No relevant context found. {search_result.explanation}", []
        return "No relevant context found.", []
    context_parts = []
    source_info = []
    # Add AI agent explanation as context
    if search_result and search_result.successful_strategy:
        context_parts.append(f"[AI Search Strategy: {search_result.successful_strategy.value} - {search_result.explanation}]")
    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get('source', 'Unknown')
        file_name = doc.metadata.get('file_name', os.path.basename(source) if source != 'Unknown' else 'Unknown')
        page = doc.metadata.get('page_number', '')
        page_info = f" (Page {page})" if page else ""
        display_source = file_name if file_name != 'Unknown' else source
        content = doc.page_content
        if len(content) > 550:
            content = content[:550] + "..."
        context_part = f"[{display_source}{page_info}] (Relevance: {score:.2f}):\n{content}"
        context_parts.append(context_part)
        source_info.append({
            'id': i+1,
            'file_name': display_source,
            'page': page,
            'relevance': score,
            'full_path': source,
            'ai_strategy': search_result.successful_strategy.value if search_result and search_result.successful_strategy else None
        })
    context_text = "\n" + "\n".join(context_parts)
    return context_text, source_info

# ==============================================================================
# STEP 6: REMOVE THE OLD AI AGENT INITIALIZATION
# ==============================================================================
# Removed: ai_analyzer = EnhancedQuestionAnalyzer()
# The Universal AI Agent creates instances as needed - no global initialization required!

# --- END AI AGENT INTEGRATION ---
# --- INITIALIZE THE AI AGENT ---
# ai_analyzer = EnhancedQuestionAnalyzer() # REMOVED - see above
# --- END INITIALIZATION ---
app = FastAPI(title="RAG API", description="Retrieval-Augmented Generation API with Conversation Support", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def read_root():
    return {"message": "RAG API with Conversation Support and Enhanced AI Agent is running"}
@app.get("/health")
def health_check():
    """Health check endpoint"""
    db_exists = os.path.exists(CHROMA_PATH)
    db_contents = []
    if db_exists:
        try:
            db_contents = os.listdir(CHROMA_PATH)
        except:
            db_contents = ["Error reading directory"]
    return {
        "status": "healthy",
        "database_exists": db_exists,
        "database_path": CHROMA_PATH,
        "database_contents": db_contents,
        "api_key_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "api_base_configured": bool(os.environ.get("OPENAI_API_BASE")),
        "active_conversations": len(conversations),
        "ai_agent_status": {
            "loaded": True, # Always true now as UniversalRAGAgent is defined
            "nlp_model_available": UniversalRAGAgent().nlp is not None, # Check via instance
            "sentence_model_available": UniversalRAGAgent().sentence_model is not None # Check via instance
        }
    }
def call_openrouter_api(prompt: str, api_key: str, api_base: str) -> str:
    """Make API call to OpenRouter with improved error handling"""
    try:
        if api_base.endswith('/'):
            api_base = api_base.rstrip('/')
        if 'openrouter.ai' in api_base and not api_base.endswith('/api/v1'):
            api_base = "https://openrouter.ai/api/v1"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "RAG Application"
        }
        models_to_try = [
            "microsoft/phi-3-mini-128k-instruct:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "google/gemma-2-9b-it:free",
            "mistralai/mistral-7b-instruct:free",
            "openchat/openchat-7b:free"
        ]
        logger.info(f"Trying {len(models_to_try)} models...")
        for i, model in enumerate(models_to_try):
            try:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1500,
                    "top_p": 0.9
                }
                logger.info(f"Attempting model {i+1}/{len(models_to_try)}: {model}")
                response = requests.post(
                    f"{api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=45
                )
                logger.info(f"Response status: {response.status_code}")
                content_type = response.headers.get('content-type', '')
                if 'text/html' in content_type:
                    logger.warning(f"Got HTML response for {model}")
                    if i == len(models_to_try) - 1:
                        raise HTTPException(status_code=500, detail="All models returned HTML errors")
                    continue
                if response.status_code != 200:
                    logger.warning(f"Model {model} failed with status {response.status_code}: {response.text[:200]}")
                    if i == len(models_to_try) - 1:
                        raise HTTPException(status_code=response.status_code, detail=f"All models failed. Last error: {response.text[:200]}")
                    continue
                try:
                    result = response.json()
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error for {model}: {e}")
                    if i == len(models_to_try) - 1:
                        raise HTTPException(status_code=500, detail="Invalid JSON response from all models")
                    continue
                if 'choices' not in result or not result['choices']:
                    logger.warning(f"No choices in response for {model}")
                    if i == len(models_to_try) - 1:
                        raise HTTPException(status_code=500, detail="No valid response from any model")
                    continue
                response_text = result['choices'][0]['message']['content']
                logger.info(f"Success with model {model}! Response length: {len(response_text)}")
                return response_text.strip()
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error for model {model}: {e}")
                if i == len(models_to_try) - 1:
                    raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error for model {model}: {e}")
                if i == len(models_to_try) - 1:
                    raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
                continue
        raise HTTPException(status_code=500, detail="All models failed unexpectedly")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API call failed completely: {e}")
        raise HTTPException(status_code=500, detail=f"API call failed: {str(e)}")
@app.post("/ask", response_model=QueryResponse)
def ask_question(query: Query):
    try:
        query_text = query.question.strip() if query.question else ""
        logger.info(f"Received query: '{query_text}'")
        if not query_text:
            return QueryResponse(
                response=None,
                error="Question cannot be empty",
                context_found=False,
                session_id=query.session_id or ""
            )
        # Get or create session
        session_id = get_or_create_session(query.session_id)
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        if not api_key:
            logger.error("OPENAI_API_KEY not found")
            return QueryResponse(
                response=None,
                error="OPENAI_API_KEY environment variable is required. Please set your OpenRouter API key.",
                context_found=False,
                session_id=session_id
            )
        if not os.path.exists(CHROMA_PATH):
            logger.error("Database not found")
            return QueryResponse(
                response=None,
                error="Vector database not found. Please run the document ingestion process first.",
                context_found=False,
                session_id=session_id
            )
        # Add user message to conversation history
        add_to_conversation(session_id, "user", query_text)
        questions = parse_multiple_questions(query_text)
        logger.info(f"Parsed {len(questions)} questions: {questions}")
        # Get conversation history for AI analysis
        conversation_history_list = conversations.get(session_id, {}).get('messages', [])
        try:
            # CRITICAL: Use the same embedding model as ingestion script
            embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # Ensure this matches ingestion
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            # Test database connectivity
            test_results = db.similarity_search("test", k=1)
            logger.info(f"Database loaded successfully with {len(test_results)} test results")
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            # --- SUGGESTION 1: More specific error message ---
            return QueryResponse(
                response=None,
                error=f"Failed to load vector database: {str(e)}. This might be due to a schema mismatch. Try deleting the '{CHROMA_PATH}' folder and re-running the ingestion script.",
                context_found=False,
                session_id=session_id
            )
            # --- END SUGGESTION 1 ---
        combined_query = " ".join(questions)
        # --- MODIFIED: Use AI Agent for Analysis and Retrieval ---
        # Perform enhanced retrieval using the AI agent
        # results, analysis = enhanced_retrieval_with_ai_agent(db, combined_query, conversation_history_list, k=5) # REMOVED
        results, search_result = enhanced_retrieval_with_universal_agent(db, combined_query, conversation_history_list, k=5) # ADDED
        # --- END MODIFICATION ---
        logger.info(f"Retrieved {len(results)} results")
        if not results:
            logger.warning("No relevant documents found")
            response_text = "I couldn't find any relevant information in the documents to answer your question. Please try rephrasing your question or check if the documents contain information about this topic."
            # Add assistant response to conversation history
            add_to_conversation(session_id, "assistant", response_text)
            return QueryResponse(
                response=response_text,
                error=None,
                context_found=False,
                session_id=session_id
            )
        # --- MODIFIED: Use AI Enhanced Context Creation ---
        # Create context using the AI analysis
        # context_text, source_info = create_ai_enhanced_context(results, analysis, questions) # REMOVED
        context_text, source_info = create_universal_context(results, search_result, questions) # ADDED
        # --- END MODIFICATION ---
        logger.info(f"Created context with {len(source_info)} sources")
        # Get conversation history (already retrieved above)
        # conversation_history_list = conversations.get(session_id, {}).get('messages', []) # Already have this
        conversation_history_context = get_conversation_context(session_id, max_messages=8)
        if len(questions) > 1:
            formatted_questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        else:
            formatted_questions = questions[0]
        formatted_prompt = ENHANCED_PROMPT_TEMPLATE.format(
            conversation_history=conversation_history_context if conversation_history_context else "No previous conversation.",
            context=context_text,
            questions=formatted_questions
        )
        logger.info(f"Prompt length: {len(formatted_prompt)} characters")
        try:
            response_text = call_openrouter_api(formatted_prompt, api_key, api_base)
            if not response_text:
                response_text = "I received an empty response. Please try again."
            else:
                 # --- MODIFIED SECTION START ---
                # Filter source_info based on relevance score
                # Only add sources section if there are relevant sources (e.g., score > 0.3)
                MIN_RELEVANCE_SCORE_FOR_SOURCE_DISPLAY = 0.3
                relevant_source_info = [source for source in source_info if source['relevance'] >= MIN_RELEVANCE_SCORE_FOR_SOURCE_DISPLAY]
                # Add sources section only if there are relevant sources
                if relevant_source_info:
                    response_text += "\n**SOURCES:**\n"
                    for source in relevant_source_info: # Use the filtered list
                        page_info = f", Page {source['page']}" if source['page'] else ""
                        response_text += f"- {source['file_name']}{page_info} (Relevance: {source['relevance']:.2f})\n"
                # --- MODIFIED SECTION END ---
            # Add assistant response to conversation history
            # Pass the potentially filtered source_info or the original if you want full history
            # Using relevant_source_info for history might be cleaner if you only want to track used sources,
            # but using source_info preserves the full retrieval context for this turn.
            # Let's use source_info for history for now to keep the history consistent with what was retrieved.
            add_to_conversation(session_id, "assistant", response_text, source_info)
            logger.info(f"Successfully generated response of length {len(response_text)}")
            # Return the filtered sources in the API response as well
            return QueryResponse(
                response=response_text,
                error=None,
                context_found=True,
                sources=relevant_source_info, # Return the filtered list
                session_id=session_id
            )
        except HTTPException as he:
            logger.error(f"API call failed: {he.detail}")
            error_response = f"API Error: {he.detail}"
            add_to_conversation(session_id, "assistant", error_response)
            return QueryResponse(
                response=None,
                error=error_response,
                context_found=True,
                sources=source_info,
                session_id=session_id
            )
    except Exception as e:
        logger.error(f"Unexpected error in ask_question: {e}")
        return QueryResponse(
            response=None,
            error=f"An unexpected error occurred: {str(e)}",
            context_found=False,
            session_id=query.session_id or ""
        )
@app.get("/conversation/{session_id}", response_model=ConversationHistory)
def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conv_data = conversations[session_id]
    return ConversationHistory(
        session_id=session_id,
        messages=conv_data['messages'],
        created_at=conv_data['created_at'].isoformat(),
        last_updated=conv_data['last_updated'].isoformat()
    )
@app.delete("/conversation/{session_id}")
def clear_conversation(session_id: str):
    """Clear conversation history for a session"""
    if session_id in conversations:
        del conversations[session_id]
        return {"message": f"Conversation {session_id} cleared"}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")
@app.get("/conversations")
def list_conversations():
    """List all active conversations"""
    return {
        "active_conversations": len(conversations),
        "sessions": [
            {
                "session_id": session_id,
                "message_count": len(data['messages']),
                "created_at": data['created_at'].isoformat(),
                "last_updated": data['last_updated'].isoformat()
            }
            for session_id, data in conversations.items()
        ]
    }
@app.post("/new-conversation")
def start_new_conversation():
    """Start a new conversation session"""
    session_id = get_or_create_session()
    return {"session_id": session_id, "message": "New conversation started"}
@app.get("/test-api")
def test_api():
    """Test endpoint to check API connectivity"""
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}
    try:
        response_text = call_openrouter_api("Hello, this is a test. Please respond with 'Test successful!'", api_key, api_base)
        return {
            "success": True,
            "response": response_text,
            "api_key_prefix": f"{api_key[:8]}...",
            "api_base": api_base
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
@app.get("/debug-db")
def debug_database():
    """Debug endpoint to check database status"""
    if not os.path.exists(CHROMA_PATH):
        return {"error": "Database folder does not exist", "path": CHROMA_PATH}
    try:
        # Check database folder contents
        db_contents = os.listdir(CHROMA_PATH)
        # Try to load the database
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # Ensure this matches ingestion
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        # Test different search queries
        test_queries = ["test", "document", "content", "information"]
        search_results = {}
        for query in test_queries:
            try:
                results = db.similarity_search(query, k=3)
                search_results[query] = {
                    "count": len(results),
                    "previews": [
                        {
                            "content": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                            "metadata": doc.metadata
                        } for doc in results[:2]  # Show first 2 results
                    ]
                }
            except Exception as e:
                search_results[query] = {"error": str(e)}
        return {
            "database_exists": True,
            "database_path": CHROMA_PATH,
            "database_contents": db_contents,
            "search_tests": search_results,
            "status": "Database appears to be working" if any(r.get("count", 0) > 0 for r in search_results.values()) else "Database exists but no search results found"
        }
    except Exception as e:
        return {
            "error": f"Database test failed: {str(e)}",
            "path": CHROMA_PATH,
            "suggestion": "Try running the ingestion script to recreate the database"
        }
@app.get("/sources")
def get_sources_info():
    """Get information about available sources in the database"""
    if not os.path.exists(CHROMA_PATH):
        return {"error": "Database not found", "path": CHROMA_PATH}
    try:
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # Ensure this matches ingestion
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        sample_docs = db.similarity_search("document content", k=50)
        sources = {}
        total_docs = len(sample_docs)
        for doc in sample_docs:
            source = doc.metadata.get('source', 'Unknown')
            file_name = doc.metadata.get('file_name', os.path.basename(source) if source != 'Unknown' else 'Unknown')
            page = doc.metadata.get('page_number', 'N/A')
            if file_name not in sources:
                sources[file_name] = {
                    'full_path': source,
                    'pages': set(),
                    'chunks': 0,
                    'sample_content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
            if page != 'N/A':
                sources[file_name]['pages'].add(str(page))
            sources[file_name]['chunks'] += 1
        for source_info in sources.values():
            source_info['pages'] = sorted(source_info['pages'], key=lambda x: int(x) if x.isdigit() else 0)
        return {
            "total_documents_sampled": total_docs,
            "total_sources": len(sources),
            "sources": sources
        }
    except Exception as e:
        return {"error": f"Failed to analyze sources: {str(e)}"}
# --- ADDITIONAL AI AGENT ENDPOINTS ---
# Note: These endpoints were using the old ai_analyzer. They would need to be updated to use UniversalRAGAgent if needed.
# For now, they are commented out or removed as per the instruction to strictly follow the guide.
# @app.post("/analyze-question")
# def analyze_single_question(query: Query):
#     """Endpoint to analyze a single question using the AI agent"""
#     # This endpoint relied on the old ai_analyzer and needs significant rework to use UniversalRAGAgent
#     # Omitted for now to strictly follow merging instructions.
#     return {"error": "Endpoint not implemented with Universal AI Agent"}

# @app.get("/debug-ai-agent")
# def debug_ai_agent():
#     """Debug endpoint to test AI agent capabilities"""
#     # This endpoint relied on the old ai_analyzer and needs rework
#     # Omitted for now to strictly follow merging instructions.
#     return {"error": "Endpoint not implemented with Universal AI Agent"}

# Chrome Extension Integration Helper
# @app.post("/chrome-extension/analyze-and-search")
# def chrome_extension_endpoint(request: dict):
#     """Special endpoint optimized for Chrome extension integration"""
#     # This endpoint relied on the old ai_analyzer and needs significant rework
#     # Omitted for now to strictly follow merging instructions.
#     return {"error": "Endpoint not implemented with Universal AI Agent"}

# def generate_search_tips(analysis): # Function depended on old analysis
#     # Omitted for now to strictly follow merging instructions.
#     return []

# Batch processing endpoint for multiple queries
# @app.post("/batch-analyze")
# def batch_analyze_questions(request: dict):
#     """Analyze multiple questions at once - useful for Chrome extension batch processing"""
#     # This endpoint relied on the old ai_analyzer and needs significant rework
#     # Omitted for now to strictly follow merging instructions.
#     return {"error": "Endpoint not implemented with Universal AI Agent"}

# Smart search suggestions endpoint
# @app.get("/search-suggestions/{query}")
# def get_search_suggestions(query: str, session_id: Optional[str] = None):
#     """Get intelligent search suggestions based on query analysis"""
#     # This endpoint relied on the old ai_analyzer and needs significant rework
#     # Omitted for now to strictly follow merging instructions.
#     return {"error": "Endpoint not implemented with Universal AI Agent"}
# --- End Additional Endpoints ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
# --- End of app.py ---
