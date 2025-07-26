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
CHROMA_PATH = os.path.join(os.getcwd(), "my_chroma_db")
logger.info(f"Checking CHROMA_PATH: {CHROMA_PATH}")
if os.path.exists(CHROMA_PATH):
    logger.info(f"Database contents: {os.listdir(CHROMA_PATH)}")
else:
    logger.warning("Database folder not found!")

# In-memory conversation storage (in production, use Redis or a database)
conversations: Dict[str, Dict] = {}

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
            if score >= 0.1:
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

# Modified prompt template to include conversation history
ENHANCED_PROMPT_TEMPLATE = """You are a helpful assistant engaged in an ongoing conversation. Answer the current question using the provided context sources and conversation history.
IMPORTANT: When citing information, use the document name format shown in brackets, for example [RCW 10.01.240.pdf] or [RCW 10.01.240.pdf (Page 1)] - do NOT use generic SOURCE numbers.
CONVERSATION HISTORY:
{conversation_history}
CURRENT CONTEXT:
{context}
CURRENT QUESTION: {questions}
Please provide a helpful answer based on the context above and the conversation history. When you reference information, cite it using the document name in brackets as shown in the context (e.g., [document_name.pdf] or [document_name.pdf (Page X)]).
If the user is asking a follow-up question or referring to something mentioned earlier in the conversation, acknowledge that context in your response.
RESPONSE:"""

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

# --- INTEGRATE AI AGENT CODE HERE ---
# Enhanced AI Agent for Better Question Understanding and Search

class QueryType(Enum):
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    MULTI_PART = "multi_part"

class QueryIntent(Enum):
    SEARCH = "search"
    CLARIFICATION = "clarification"
    FOLLOW_UP = "follow_up"
    DEFINITION = "definition"
    EXPLANATION = "explanation"

@dataclass
class QuestionAnalysis:
    original_query: str
    query_type: QueryType
    intent: QueryIntent
    key_entities: List[str]
    keywords: List[str]
    reformulated_queries: List[str]
    context_requirements: List[str]
    confidence_score: float
    search_strategy: str

class EnhancedQuestionAnalyzer:
    def __init__(self):
        # Load spaCy model for NER and linguistic analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy 'en_core_web_sm' model loaded successfully.")
        except OSError:
            logger.warning("Warning: spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        # Load sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SentenceTransformer 'all-MiniLM-L6-v2' model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
            self.sentence_model = None
        # Query patterns for different types
        self.query_patterns = {
            QueryType.FACTUAL: [
                r'\b(what|who|where|when|which)\b',
                r'\bis\b.*\?',
                r'\bare\b.*\?'
            ],
            QueryType.PROCEDURAL: [
                r'\b(how to|how do|steps|process|procedure)\b',
                r'\bexplain.*how\b',
                r'\bwhat.*process\b'
            ],
            QueryType.COMPARATIVE: [
                r'\b(compare|versus|vs|difference|similar|different)\b',
                r'\bbetter|worse|more|less\b',
                r'\brather than|instead of\b'
            ],
            QueryType.TEMPORAL: [
                r'\b(when|before|after|during|timeline|history)\b',
                r'\b(first|then|next|finally|sequence)\b'
            ],
            QueryType.CAUSAL: [
                r'\b(why|because|cause|reason|result|due to)\b',
                r'\bwhat.*cause\b',
                r'\bhow.*affect\b'
            ]
        }

    def analyze_question(self, query: str, conversation_history: List[Dict] = None) -> QuestionAnalysis:
        """Comprehensive question analysis for better retrieval"""
        # Basic preprocessing
        query_lower = query.lower().strip()
        # Determine query type
        query_type = self._classify_query_type(query_lower)
        # Determine intent
        intent = self._determine_intent(query_lower, conversation_history)
        # Extract entities and keywords
        entities = self._extract_entities(query)
        keywords = self._extract_keywords(query_lower)
        # Generate reformulated queries
        reformulated = self._generate_reformulations(query, query_type, entities, keywords)
        # Determine context requirements
        context_reqs = self._analyze_context_requirements(query_lower, intent, conversation_history)
        # Calculate confidence score
        confidence = self._calculate_confidence(query, entities, keywords)
        # Determine search strategy
        strategy = self._determine_search_strategy(query_type, intent, entities)
        logger.debug(f"AI Analysis - Query: '{query}', Type: {query_type.value}, Intent: {intent.value}, Entities: {entities}")
        return QuestionAnalysis(
            original_query=query,
            query_type=query_type,
            intent=intent,
            key_entities=entities,
            keywords=keywords,
            reformulated_queries=reformulated,
            context_requirements=context_reqs,
            confidence_score=confidence,
            search_strategy=strategy
        )

    def _classify_query_type(self, query: str) -> QueryType:
        """Classify the type of query for appropriate handling"""
        scores = {}
        for query_type, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches
            scores[query_type] = score
        # Check for multi-part questions
        if len(re.findall(r'\?', query)) > 1 or len(re.findall(r'\band\b|\bor\b|\balso\b', query)) > 0:
            return QueryType.MULTI_PART
        # Return the type with highest score
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return QueryType.FACTUAL  # Default

    def _determine_intent(self, query: str, history: List[Dict] = None) -> QueryIntent:
        """Determine the user's intent"""
        # Follow-up indicators
        followup_patterns = [
            r'\b(also|additionally|furthermore|moreover)\b',
            r'\b(what about|how about)\b',
            r'\b(and|then)\b.*\?'
        ]
        # Clarification indicators
        clarification_patterns = [
            r'\b(mean|clarify|explain|elaborate)\b',
            r'\bwhat do you mean\b',
            r'\bcan you explain\b'
        ]
        # Definition indicators
        definition_patterns = [
            r'\b(define|definition|what is|what are)\b',
            r'\bmeans?\b',
            r'\brefers to\b'
        ]
        if any(re.search(pattern, query) for pattern in followup_patterns):
            return QueryIntent.FOLLOW_UP
        elif any(re.search(pattern, query) for pattern in clarification_patterns):
            return QueryIntent.CLARIFICATION
        elif any(re.search(pattern, query) for pattern in definition_patterns):
            return QueryIntent.DEFINITION
        elif re.search(r'\b(how|why|explain)\b', query):
            return QueryIntent.EXPLANATION
        return QueryIntent.SEARCH

    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities and important nouns"""
        entities = []
        if self.nlp:
            try:
                doc = self.nlp(query)
                # Named entities
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'LAW', 'NORP']:
                        entities.append(ent.text)
                # Important nouns and noun phrases
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) <= 4:  # Slightly increase phrase length limit
                        entities.append(chunk.text)
            except Exception as e:
                logger.error(f"spaCy processing error: {e}")
        # Fallback: simple noun extraction if spaCy fails or is not available
        if not entities:
            # Simple regex-based entity extraction (improved)
            entities.extend(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query))
        return list(set([e.strip() for e in entities if e.strip()])) # Clean and deduplicate

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords for search"""
        # Remove stop words and common question words
        stop_words = {
            'what', 'how', 'when', 'where', 'why', 'who', 'which', 'is', 'are',
            'was', 'were', 'do', 'does', 'did', 'can', 'could', 'would', 'should',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'about', 'from', 'this', 'that', 'these', 'those',
            'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            'we', 'us', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves'
        }
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return list(set(keywords)) # Deduplicate

    def _generate_reformulations(self, query: str, query_type: QueryType,
                               entities: List[str], keywords: List[str]) -> List[str]:
        """Generate alternative formulations of the query"""
        reformulations = [query]  # Always include original
        # For procedural queries, add "steps" and "how to" variants
        if query_type == QueryType.PROCEDURAL:
            if not re.search(r'\bsteps\b', query, re.IGNORECASE):
                reformulations.append(f"steps to {' '.join(keywords[:3])}")
            if not re.search(r'\bhow to\b', query, re.IGNORECASE):
                reformulations.append(f"how to {' '.join(keywords[:3])}")
        # For factual queries, add "what is" variants
        elif query_type == QueryType.FACTUAL:
            if entities:
                reformulations.append(f"what is {entities[0]}")
                reformulations.append(f"definition of {entities[0]}")
        # Add keyword-only search
        if len(keywords) > 1:
            reformulations.append(' '.join(keywords[:4]))  # Top 4 keywords
        # Add entity-focused search
        if entities:
            reformulations.append(' '.join(entities[:3]))  # Top 3 entities
        return reformulations[:6]  # Limit to 6 reformulations

    def _analyze_context_requirements(self, query: str, intent: QueryIntent,
                                    history: List[Dict] = None) -> List[str]:
        """Analyze what contextual information might be needed"""
        requirements = []
        # Check for pronouns that might need context resolution
        pronouns = re.findall(r'\b(it|this|that|they|them|these|those)\b', query, re.IGNORECASE)
        if pronouns:
            requirements.append("pronoun_resolution")
        # Check for follow-up indicators
        if intent == QueryIntent.FOLLOW_UP:
            requirements.append("previous_topic")
        # Check for comparative language
        if re.search(r'\b(compared to|versus|rather than|instead of)\b', query, re.IGNORECASE):
            requirements.append("comparison_context")
        # Check for temporal references
        if re.search(r'\b(before|after|during|since|until)\b', query, re.IGNORECASE):
            requirements.append("temporal_context")
        return requirements

    def _calculate_confidence(self, query: str, entities: List[str],
                            keywords: List[str]) -> float:
        """Calculate confidence score for the analysis"""
        score = 0.5  # Base score
        # Boost for clear entities
        if entities:
            score += min(0.25, len(entities) * 0.05)
        # Boost for good keywords
        if keywords:
            score += min(0.2, len(keywords) * 0.03)
        # Boost for complete sentences
        if query.endswith('?') or query.endswith('.'):
            score += 0.05
        # Penalty for very short queries
        if len(query.split()) < 3:
            score -= 0.15
        return min(1.0, max(0.1, score))

    def _determine_search_strategy(self, query_type: QueryType, intent: QueryIntent,
                                 entities: List[str]) -> str:
        """Determine the best search strategy"""
        if query_type == QueryType.MULTI_PART:
            return "multi_query_expansion"
        elif query_type == QueryType.PROCEDURAL:
            return "sequential_steps_focus"
        elif query_type == QueryType.COMPARATIVE:
            return "comparative_analysis"
        elif intent == QueryIntent.DEFINITION:
            return "definition_focused"
        elif entities:
            return "entity_centric"
        else:
            return "keyword_semantic_hybrid"

# Integration with your existing FastAPI code
class EnhancedRAGSystem:
    def __init__(self, db, analyzer: EnhancedQuestionAnalyzer):
        self.db = db
        self.analyzer = analyzer

    def enhanced_search(self, query: str, conversation_history: List[Dict] = None, k: int = 5) -> Tuple[List, QuestionAnalysis]:
        """Enhanced search with intelligent query analysis"""
        # Analyze the question
        analysis = self.analyzer.analyze_question(query, conversation_history)
        all_results = []
        # Execute search based on strategy
        if analysis.search_strategy == "multi_query_expansion":
            # Search with multiple reformulations
            for reformulated_query in analysis.reformulated_queries:
                try:
                    results = self.db.similarity_search_with_relevance_scores(reformulated_query, k=max(1, k//2))
                    all_results.extend(results)
                except Exception as e:
                    logger.warning(f"Multi-query search failed for '{reformulated_query}': {e}")
                    continue
        elif analysis.search_strategy == "entity_centric":
            # Focus on entity-based searches
            if analysis.key_entities:
                entity_query = " ".join(analysis.key_entities)
                try:
                    entity_results = self.db.similarity_search_with_relevance_scores(entity_query, k=k)
                    all_results.extend(entity_results)
                except Exception as e:
                    logger.warning(f"Entity search failed for '{entity_query}': {e}")
            # Also search with original query
            try:
                original_results = self.db.similarity_search_with_relevance_scores(query, k=max(1, k//2))
                all_results.extend(original_results)
            except Exception as e:
                logger.warning(f"Original query search failed: {e}")
        else:
            # Default hybrid approach
            try:
                # Original query
                original_results = self.db.similarity_search_with_relevance_scores(query, k=k)
                all_results.extend(original_results)
                # Keyword-based search
                if analysis.keywords:
                    keyword_query = " ".join(analysis.keywords[:4])
                    if keyword_query and keyword_query.lower() != query.lower():
                        keyword_results = self.db.similarity_search_with_relevance_scores(keyword_query, k=max(1, k//2))
                        all_results.extend(keyword_results)
            except Exception as e:
                logger.error(f"Hybrid search error: {e}")
        # Remove duplicates and sort by relevance
        seen_content = set()
        unique_results = []
        for doc, score in all_results:
            # Use first 150 chars for content hash to be more robust
            content_hash = hash(doc.page_content[:150].strip())
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                # Boost score for entity matches
                if analysis.key_entities:
                     entity_boost = sum(1 for entity in analysis.key_entities if entity.lower() in doc.page_content.lower())
                     if entity_boost > 0:
                         score = min(1.0, score + (entity_boost * 0.05)) # Slightly boost score
                unique_results.append((doc, score))
        # Sort by score and return top k
        unique_results.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Enhanced search returned {len(unique_results[:k])} results after deduplication.")
        return unique_results[:k], analysis

# Modified version of your existing functions to integrate the enhanced analyzer
# --- MODIFIED: Use global ai_analyzer instance ---
def enhanced_retrieval_with_ai_agent(db, query_text: str, conversation_history: List[Dict] = None, k: int = 5):
    """Your existing enhanced_retrieval function with AI agent integration"""
    # analyzer = EnhancedQuestionAnalyzer() # REMOVED - use global
    enhanced_system = EnhancedRAGSystem(db, ai_analyzer) # USE GLOBAL ai_analyzer
    try:
        logger.info(f"AI Agent analyzing query: '{query_text}'")
        # Use the enhanced search system
        results, analysis = enhanced_system.enhanced_search(query_text, conversation_history, k)
        logger.info(f"AI Agent Analysis - Type: {analysis.query_type.value}, "
                   f"Intent: {analysis.intent.value}, "
                   f"Strategy: {analysis.search_strategy}, "
                   f"Entities: {analysis.key_entities}, "
                   f"Confidence: {analysis.confidence_score:.2f}")
        return results, analysis
    except Exception as e:
        logger.error(f"Enhanced retrieval with AI agent failed: {e}")
        # Fallback to your original method
        # Ensure fallback returns (results, None) for analysis compatibility
        fallback_results = enhanced_retrieval(db, query_text, k)
        return fallback_results, None

# Enhanced context creation that uses AI analysis
def create_ai_enhanced_context(results, analysis: QuestionAnalysis, questions: list) -> tuple:
    """Enhanced context creation using AI analysis"""
    if not results:
        return "No relevant context found.", []
    context_parts = []
    source_info = []
    # Sort results based on AI analysis
    if analysis and analysis.query_type == QueryType.PROCEDURAL:
        # For procedural queries, prioritize step-by-step content
        results = sorted(results, key=lambda x: (
            x[1],  # Original relevance score
            1 if any(word in x[0].page_content.lower() for word in ['step', 'process', 'procedure', 'how']) else 0
        ), reverse=True)
    elif analysis and analysis.key_entities:
        # For entity-rich queries, prioritize content with entities
        results = sorted(results, key=lambda x: (
            x[1],  # Original relevance score
            sum(1 for entity in analysis.key_entities if entity.lower() in x[0].page_content.lower())
        ), reverse=True)
    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get('source', 'Unknown')
        file_name = doc.metadata.get('file_name', os.path.basename(source) if source != 'Unknown' else 'Unknown')
        page = doc.metadata.get('page_number', '')
        page_info = f" (Page {page})" if page else ""
        display_source = file_name if file_name != 'Unknown' else source
        content = doc.page_content
        # Truncate based on query type
        max_length = 650 if analysis and analysis.query_type == QueryType.PROCEDURAL else 550
        if len(content) > max_length:
            content = content[:max_length] + "..."
        # Enhanced relevance display
        relevance_note = ""
        if analysis:
            if any(entity.lower() in content.lower() for entity in analysis.key_entities):
                relevance_note = " [Entity Match]"
            elif analysis.query_type == QueryType.PROCEDURAL and any(word in content.lower() for word in ['step', 'process', 'procedure']):
                relevance_note = " [Process Info]"
        context_part = f"[{display_source}{page_info}] (Relevance: {score:.2f}{relevance_note}):\n{content}"
        context_parts.append(context_part)
        source_info.append({
            'id': i+1,
            'file_name': display_source,
            'page': page,
            'relevance': score,
            'full_path': source,
            'ai_analysis': {
                'entity_match': any(entity.lower() in content.lower() for entity in (analysis.key_entities if analysis else [])),
                'query_type_relevant': analysis.query_type.value if analysis else None
            }
        })
    context_text = "\n" + "\n".join(context_parts)
    return context_text, source_info
# --- END AI AGENT INTEGRATION ---

# --- INITIALIZE THE AI AGENT ---
ai_analyzer = EnhancedQuestionAnalyzer()
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
            "loaded": ai_analyzer is not None,
            "nlp_model_available": ai_analyzer.nlp is not None if ai_analyzer else False,
            "sentence_model_available": ai_analyzer.sentence_model is not None if ai_analyzer else False
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
            embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            # Test database connectivity
            test_results = db.similarity_search("test", k=1)
            logger.info(f"Database loaded successfully with {len(test_results)} test results")
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            return QueryResponse(
                response=None,
                error=f"Failed to load vector database: {str(e)}. Make sure you've run the ingestion script first.",
                context_found=False,
                session_id=session_id
            )

        combined_query = " ".join(questions)

        # --- MODIFIED: Use AI Agent for Analysis and Retrieval ---
        # Perform enhanced retrieval using the AI agent
        results, analysis = enhanced_retrieval_with_ai_agent(db, combined_query, conversation_history_list, k=5)
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
        context_text, source_info = create_ai_enhanced_context(results, analysis, questions)
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
                        response_text += f"• {source['file_name']}{page_info} (Relevance: {source['relevance']:.2f})\n"
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
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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
@app.post("/analyze-question")
def analyze_single_question(query: Query):
    """Endpoint to analyze a single question using the AI agent"""
    try:
        query_text = query.question.strip() if query.question else ""
        if not query_text:
            return {"error": "Question cannot be empty"}

        # Get conversation history if session provided
        conversation_history_list = []
        if query.session_id and query.session_id in conversations:
            conversation_history_list = conversations[query.session_id]['messages']

        # Analyze the question using the global ai_analyzer
        analysis = ai_analyzer.analyze_question(query_text, conversation_history_list)

        # Return the analysis results in a structured format
        return {
            "original_query": analysis.original_query,
            "query_type": analysis.query_type.value,
            "intent": analysis.intent.value,
            "key_entities": analysis.key_entities,
            "keywords": analysis.keywords,
            "reformulated_queries": analysis.reformulated_queries,
            "context_requirements": analysis.context_requirements,
            "confidence_score": analysis.confidence_score,
            "search_strategy": analysis.search_strategy,
            "recommendations": {
                "suggested_reformulations": analysis.reformulated_queries[:3],
                "key_terms_to_focus": analysis.key_entities + analysis.keywords[:3],
                "search_approach": analysis.search_strategy
            }
        }
    except Exception as e:
        logger.error(f"Question analysis failed: {e}")
        return {"error": f"Analysis failed: {str(e)}"}

@app.get("/debug-ai-agent")
def debug_ai_agent():
    """Debug endpoint to test AI agent capabilities"""
    try:
        # Test questions
        test_questions = [
            "What is the definition of due process?",
            "How do I file a complaint?",
            "Compare the penalties for misdemeanor vs felony charges",
            "Why was this law created?",
            "What steps are involved in the appeal process?"
        ]
        results = {}
        for question in test_questions:
            try:
                analysis = ai_analyzer.analyze_question(question)
                results[question] = {
                    "query_type": analysis.query_type.value,
                    "intent": analysis.intent.value,
                    "entities": analysis.key_entities,
                    "keywords": analysis.keywords[:5],
                    "confidence": analysis.confidence_score,
                    "strategy": analysis.search_strategy,
                    "reformulations": analysis.reformulated_queries[:3]
                }
            except Exception as e:
                results[question] = {"error": str(e)}
        return {
            "ai_agent_status": "functional",
            "test_results": results,
            "analyzer_loaded": ai_analyzer is not None,
            "nlp_model_available": ai_analyzer.nlp is not None if ai_analyzer else False,
            "sentence_model_available": ai_analyzer.sentence_model is not None if ai_analyzer else False
        }
    except Exception as e:
        return {"error": f"AI agent debug failed: {str(e)}"}

# Chrome Extension Integration Helper
@app.post("/chrome-extension/analyze-and-search")
def chrome_extension_endpoint(request: dict):
    """Special endpoint optimized for Chrome extension integration"""
    try:
        query_text = request.get('query', '').strip()
        session_id = request.get('session_id')
        page_context = request.get('page_context', '')  # Current webpage context
        user_selection = request.get('selected_text', '')  # Selected text from page
        if not query_text:
            return {"error": "Query is required"}
        # Enhance query with page context if available
        enhanced_query = query_text
        context_info = []
        if user_selection:
            enhanced_query = f"Regarding '{user_selection}': {query_text}"
            context_info.append(f"User selected: {user_selection}")
        if page_context:
            context_info.append(f"Page context: {page_context[:200]}...")
        # Get or create session
        session_id = get_or_create_session(session_id)
        # Analyze the enhanced query
        conversation_history = conversations.get(session_id, {}).get('messages', [])
        analysis = ai_analyzer.analyze_question(enhanced_query, conversation_history)
        # Perform search if database is available
        search_results = []
        if os.path.exists(CHROMA_PATH):
            try:
                embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
                results, _ = enhanced_retrieval_with_ai_agent(
                    db, enhanced_query, conversation_history, k=3
                )
                for doc, score in results:
                    search_results.append({
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "source": doc.metadata.get('file_name', 'Unknown'),
                        "page": doc.metadata.get('page_number', ''),
                        "relevance": score
                    })
            except Exception as e:
                logger.error(f"Search failed in chrome extension endpoint: {e}")
        return {
            "query_analysis": {
                "original_query": query_text,
                "enhanced_query": enhanced_query,
                "query_type": analysis.query_type.value,
                "intent": analysis.intent.value,
                "entities": analysis.key_entities,
                "keywords": analysis.keywords,
                "confidence": analysis.confidence_score,
                "suggested_reformulations": analysis.reformulated_queries[:3]
            },
            "search_results": search_results,
            "context_info": context_info,
            "session_id": session_id,
            "recommendations": {
                "search_tips": generate_search_tips(analysis),
                "related_queries": analysis.reformulated_queries[:2]
            }
        }
    except Exception as e:
        logger.error(f"Chrome extension endpoint failed: {e}")
        return {"error": str(e)}

def generate_search_tips(analysis: QuestionAnalysis) -> list:
    """Generate helpful search tips based on AI analysis"""
    tips = []
    if analysis.query_type.value == "procedural":
        tips.append("Try searching for step-by-step guides or procedures")
        tips.append("Look for documents containing 'process', 'steps', or 'how to'")
    elif analysis.query_type.value == "comparative":
        tips.append("Search for documents that discuss differences or comparisons")
        tips.append("Try using 'versus', 'compared to', or 'difference between'")
    elif analysis.query_type.value == "factual":
        tips.append("Look for definitions or explanatory content")
        tips.append("Try searching with 'what is' or 'definition of'")
    if analysis.key_entities:
        tips.append(f"Focus on these key terms: {', '.join(analysis.key_entities[:3])}")
    if analysis.confidence_score < 0.5:
        tips.append("Try being more specific or adding more details to your question")
    return tips[:4]  # Limit to 4 tips

# Batch processing endpoint for multiple queries
@app.post("/batch-analyze")
def batch_analyze_questions(request: dict):
    """Analyze multiple questions at once - useful for Chrome extension batch processing"""
    try:
        queries = request.get('queries', [])
        session_id = request.get('session_id')
        if not queries or not isinstance(queries, list):
            return {"error": "Queries list is required"}
        if len(queries) > 10:  # Limit batch size
            return {"error": "Maximum 10 queries per batch"}
        session_id = get_or_create_session(session_id)
        conversation_history = conversations.get(session_id, {}).get('messages', [])
        results = []
        for i, query in enumerate(queries):
            try:
                if not query.strip():
                    results.append({"error": "Empty query", "index": i})
                    continue
                analysis = ai_analyzer.analyze_question(query, conversation_history)
                results.append({
                    "index": i,
                    "original_query": query,
                    "analysis": {
                        "query_type": analysis.query_type.value,
                        "intent": analysis.intent.value,
                        "entities": analysis.key_entities,
                        "keywords": analysis.keywords[:5],
                        "confidence": analysis.confidence_score,
                        "strategy": analysis.search_strategy,
                        "reformulations": analysis.reformulated_queries[:2]
                    }
                })
            except Exception as e:
                results.append({"error": str(e), "index": i})
        return {
            "session_id": session_id,
            "total_processed": len(queries),
            "results": results
        }
    except Exception as e:
        return {"error": str(e)}

# Smart search suggestions endpoint
@app.get("/search-suggestions/{query}")
def get_search_suggestions(query: str, session_id: Optional[str] = None):
    """Get intelligent search suggestions based on query analysis"""
    try:
        if not query.strip():
            return {"error": "Query parameter is required"}
        # Get conversation context
        conversation_history = []
        if session_id and session_id in conversations:
            conversation_history = conversations[session_id]['messages']
        # Analyze the query
        analysis = ai_analyzer.analyze_question(query, conversation_history)
        # Generate suggestions based on analysis
        suggestions = []
        # Add reformulated queries
        suggestions.extend(analysis.reformulated_queries[:3])
        # Add entity-focused suggestions
        if analysis.key_entities:
            for entity in analysis.key_entities[:2]:
                suggestions.append(f"Tell me more about {entity}")
                if analysis.query_type.value == "factual":
                    suggestions.append(f"What is the definition of {entity}?")
        # Add query-type specific suggestions
        if analysis.query_type.value == "procedural":
            suggestions.append(f"What are the steps to {' '.join(analysis.keywords[:3])}?")
            suggestions.append(f"How do I complete the {' '.join(analysis.keywords[:2])} process?")
        elif analysis.query_type.value == "comparative":
            if len(analysis.key_entities) >= 2:
                suggestions.append(f"What's the difference between {analysis.key_entities[0]} and {analysis.key_entities[1]}?")
        # Remove duplicates and limit
        unique_suggestions = list(dict.fromkeys(suggestions))[:8]
        return {
            "original_query": query,
            "suggestions": unique_suggestions,
            "analysis_summary": {
                "type": analysis.query_type.value,
                "intent": analysis.intent.value,
                "confidence": analysis.confidence_score
            },
            "search_tips": generate_search_tips(analysis)
        }
    except Exception as e:
        logger.error(f"Search suggestions failed: {e}")
        return {"error": str(e)}
# --- End Additional Endpoints ---

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

# --- End of app.py ---
