"""RAG (Retrieval-Augmented Generation) operations service"""
import os
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import DEFAULT_CHROMA_PATH, DEFAULT_SEARCH_K, ENHANCED_SEARCH_K, CONFIDENCE_WEIGHTS
from ..core.dependencies import get_nlp
from ..core.exceptions import RetrievalError
from .container_manager import get_container_manager
from ..utils.text_processing import remove_duplicate_documents

logger = logging.getLogger(__name__)

def load_database():
    """Load the default database"""
    try:
        if not os.path.exists(DEFAULT_CHROMA_PATH):
            logger.warning(f"Default database path does not exist: {DEFAULT_CHROMA_PATH}")
            return None
        
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(
            collection_name="default",
            embedding_function=embedding_function,
            persist_directory=DEFAULT_CHROMA_PATH
        )
        logger.debug("Default database loaded successfully")
        return db
    except Exception as e:
        logger.error(f"Failed to load default database: {e}")
        raise RetrievalError(f"Failed to load default database: {str(e)}")

def enhanced_retrieval_v2(db, query_text: str, conversation_history_context: str, k: int = ENHANCED_SEARCH_K, document_filter: Dict = None) -> Tuple[List, str]:
    """Enhanced retrieval with multiple strategies"""
    logger.info(f"[ENHANCED_RETRIEVAL] Original query: '{query_text}'")
    
    try:
        direct_results = db.similarity_search_with_score(query_text, k=k, filter=document_filter)
        logger.info(f"[ENHANCED_RETRIEVAL] Direct search returned {len(direct_results)} results")
        
        expanded_query = f"{query_text} {conversation_history_context}"
        expanded_results = db.similarity_search_with_score(expanded_query, k=k, filter=document_filter)
        logger.info(f"[ENHANCED_RETRIEVAL] Expanded search returned {len(expanded_results)} results")
        
        sub_queries = []
        nlp = get_nlp()
        if nlp:
            doc = nlp(query_text)
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PERSON", "LAW", "DATE"]:
                    sub_queries.append(f"What is {ent.text}?")
        
        if not sub_queries:
            question_words = ["what", "who", "when", "where", "why", "how"]
            for word in question_words:
                if word in query_text.lower():
                    sub_queries.append(f"{word.capitalize()} {query_text.lower().replace(word, '').strip()}?")
        
        sub_query_results = []
        for sq in sub_queries[:3]:
            sq_results = db.similarity_search_with_score(sq, k=3, filter=document_filter)
            sub_query_results.extend(sq_results)
        
        logger.info(f"[ENHANCED_RETRIEVAL] Sub-query search returned {len(sub_query_results)} results")
        
        all_results = direct_results + expanded_results + sub_query_results
        unique_results = remove_duplicate_documents(all_results)
        top_results = unique_results[:k]
        
        logger.info(f"[ENHANCED_RETRIEVAL] Final results after deduplication: {len(top_results)}")
        return top_results, "enhanced_retrieval_v2"
        
    except Exception as e:
        logger.error(f"[ENHANCED_RETRIEVAL] Error in enhanced retrieval: {e}")
        basic_results = db.similarity_search_with_score(query_text, k=k, filter=document_filter)
        return basic_results, "basic_fallback"

def combined_search(query: str, user_id: Optional[str], search_scope: str, conversation_context: str, 
                   use_enhanced: bool = True, k: int = DEFAULT_SEARCH_K, document_id: str = None) -> Tuple[List, List[str], str]:
    """Combined search across all sources"""
    all_results = []
    sources_searched = []
    retrieval_method = "basic"
    
    if search_scope in ["all", "default_only"]:
        try:
            default_db = load_database()
            if default_db:
                if use_enhanced:
                    default_results, method = enhanced_retrieval_v2(default_db, query, conversation_context, k=k)
                    retrieval_method = method
                else:
                    default_results = default_db.similarity_search_with_score(query, k=k)
                    retrieval_method = "basic_search"
                
                for doc, score in default_results:
                    doc.metadata['source_type'] = 'default_database'
                    all_results.append((doc, score))
                sources_searched.append("default_database")
        except Exception as e:
            logger.error(f"Error searching default database: {e}")
    
    if user_id and search_scope in ["all", "user_only"]:
        try:
            container_manager = get_container_manager()
            if use_enhanced:
                user_results = container_manager.enhanced_search_user_container(user_id, query, conversation_context, k=k, document_id=document_id)
            else:
                user_results = container_manager.search_user_container(user_id, query, k=k, document_id=document_id)
            
            for doc, score in user_results:
                doc.metadata['source_type'] = 'user_container'
                all_results.append((doc, score))
            if user_results:
                sources_searched.append("user_container")
        except Exception as e:
            logger.error(f"Error searching user container: {e}")
    
    if use_enhanced:
        all_results = remove_duplicate_documents(all_results)
    else:
        all_results.sort(key=lambda x: x[1], reverse=True)
    
    return all_results[:k], sources_searched, retrieval_method

def calculate_confidence_score(results_with_scores: List[Tuple], response_length: int) -> float:
    """Calculate confidence score for results"""
    try:
        if not results_with_scores:
            return 0.2
        
        scores = [score for _, score in results_with_scores]
        avg_relevance = np.mean(scores)
        doc_factor = min(1.0, len(results_with_scores) / 5.0)
        
        if len(scores) > 1:
            score_std = np.std(scores)
            consistency_factor = max(0.5, 1.0 - score_std)
        else:
            consistency_factor = 0.7
            
        completeness_factor = min(1.0, response_length / 500.0)
        
        confidence = (
            avg_relevance * CONFIDENCE_WEIGHTS["relevance"] +
            doc_factor * CONFIDENCE_WEIGHTS["document_count"] +
            consistency_factor * CONFIDENCE_WEIGHTS["consistency"] +
            completeness_factor * CONFIDENCE_WEIGHTS["completeness"]
        )
        
        confidence = max(0.0, min(1.0, confidence))
        return confidence
    
    except Exception as e:
        logger.error(f"Error calculating confidence score: {e}")
        return 0.5
