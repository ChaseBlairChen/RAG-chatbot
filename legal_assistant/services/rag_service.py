"""RAG (Retrieval-Augmented Generation) operations service"""
import os
import logging
import numpy as np
import re
from typing import List, Tuple, Dict, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import (
    DEFAULT_CHROMA_PATH, DEFAULT_SEARCH_K, ENHANCED_SEARCH_K, 
    CONFIDENCE_WEIGHTS, FeatureFlags, MIN_RELEVANCE_SCORE, SEARCH_CONFIG
)
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

def enhanced_retrieval_v3(db, query_text: str, conversation_history_context: str, 
                         k: int = ENHANCED_SEARCH_K, document_filter: Dict = None) -> Tuple[List, str]:
    """Enhanced retrieval with query understanding and result filtering"""
    logger.info(f"[ENHANCED_RETRIEVAL_V3] Query: '{query_text}'")
    
    try:
        # Step 1: Query analysis and expansion
        expanded_queries = expand_query_intelligently(query_text)
        logger.info(f"Expanded to {len(expanded_queries)} queries")
        
        all_results = []
        seen_content_hashes = set()
        
        # Step 2: Search with each query variant
        for query_variant in expanded_queries[:3]:  # Limit to top 3 variants
            results = db.similarity_search_with_score(
                query_variant, 
                k=k * 2,  # Get more results for filtering
                filter=document_filter
            )
            
            # Deduplicate and filter by relevance
            for doc, score in results:
                # Create content hash for deduplication
                content_hash = hash(doc.page_content[:200])
                
                # Apply stricter relevance filtering
                if score >= MIN_RELEVANCE_SCORE and content_hash not in seen_content_hashes:
                    # Boost score for exact matches
                    if SEARCH_CONFIG["boost_exact_matches"]:
                        score = boost_score_for_exact_matches(query_text, doc.page_content, score)
                    
                    all_results.append((doc, score))
                    seen_content_hashes.add(content_hash)
        
        # Step 3: Sort by relevance and apply cutoff
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        # Step 4: Apply strict relevance cutoff
        filtered_results = [
            (doc, score) for doc, score in all_results 
            if score >= MIN_RELEVANCE_SCORE
        ]
        
        # Step 5: Rerank if enabled
        if SEARCH_CONFIG["rerank_enabled"] and len(filtered_results) > 0:
            filtered_results = rerank_results(query_text, filtered_results[:SEARCH_CONFIG["max_results_to_rerank"]])
        
        # Return top k results
        final_results = filtered_results[:k]
        
        logger.info(f"[ENHANCED_RETRIEVAL_V3] Returning {len(final_results)} results after filtering")
        return final_results, "enhanced_retrieval_v3"
        
    except Exception as e:
        logger.error(f"[ENHANCED_RETRIEVAL_V3] Error: {e}")
        # Fallback to basic search
        results = db.similarity_search_with_score(query_text, k=k, filter=document_filter)
        return [(doc, score) for doc, score in results if score >= MIN_RELEVANCE_SCORE], "basic_fallback"

def expand_query_intelligently(query: str) -> List[str]:
    """Intelligently expand query for better recall"""
    expanded = [query]  # Original query first
    
    # Add question variations
    if not query.endswith('?'):
        expanded.append(query + '?')
    
    # Handle bill/statute queries specially
    bill_match = re.search(r'(HB|SB|SSB|ESSB|SHB|ESHB)\s*(\d+)', query, re.IGNORECASE)
    if bill_match:
        bill_type = bill_match.group(1).upper()
        bill_num = bill_match.group(2)
        
        # Add variations
        expanded.extend([
            f"{bill_type} {bill_num}",
            f"{bill_type}{bill_num}",
            f"House Bill {bill_num}" if bill_type.startswith('H') else f"Senate Bill {bill_num}",
            f"bill {bill_num} {bill_type}"
        ])
    
    # Extract key entities for expansion
    key_terms = extract_key_terms(query)
    if key_terms:
        # Create focused query with just key terms
        expanded.append(' '.join(key_terms))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_expanded = []
    for q in expanded:
        if q not in seen:
            seen.add(q)
            unique_expanded.append(q)
    
    return unique_expanded

def extract_key_terms(query: str) -> List[str]:
    """Extract key terms from query"""
    # Remove common words
    stop_words = {'what', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                  'to', 'for', 'of', 'with', 'by', 'from', 'about', 'into', 'through',
                  'during', 'before', 'after', 'above', 'below', 'between', 'under'}
    
    words = query.lower().split()
    key_terms = [w for w in words if w not in stop_words and len(w) > 2]
    
    return key_terms

def boost_score_for_exact_matches(query: str, content: str, base_score: float) -> float:
    """Boost score for exact matches"""
    query_lower = query.lower()
    content_lower = content.lower()
    
    # Check for exact phrase match
    if query_lower in content_lower:
        return min(1.0, base_score * SEARCH_CONFIG["boost_factor"])
    
    # Check for all query terms present
    query_terms = extract_key_terms(query)
    if all(term in content_lower for term in query_terms):
        return min(1.0, base_score * 1.2)
    
    return base_score

def rerank_results(query: str, results: List[Tuple]) -> List[Tuple]:
    """Rerank results using cross-encoder if available"""
    try:
        from sentence_transformers import CrossEncoder
        
        # Initialize reranker (cache this in production)
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Prepare pairs for reranking
        pairs = [[query, doc.page_content] for doc, _ in results]
        
        # Get reranking scores
        rerank_scores = reranker.predict(pairs)
        
        # Combine scores with original scores
        reranked = []
        for i, (doc, orig_score) in enumerate(results):
            # Normalize rerank score to 0-1
            rerank_score = (rerank_scores[i] + 10) / 20
            
            # Weighted combination
            final_score = 0.4 * orig_score + 0.6 * rerank_score
            reranked.append((doc, final_score))
        
        # Sort by final score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("Results reranked successfully")
        return reranked
        
    except ImportError:
        logger.warning("Cross-encoder not available, skipping reranking")
        return results
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return results

def hybrid_retrieval_default(db, query_text: str, k: int = ENHANCED_SEARCH_K, document_filter: Dict = None) -> Tuple[List, str]:
    """Hybrid retrieval for default database if hybrid search is available"""
    logger.info(f"[HYBRID_RETRIEVAL] Attempting hybrid search for default database")
    
    try:
        # Check if hybrid search is available
        if not FeatureFlags.HYBRID_SEARCH_AVAILABLE:
            logger.info("[HYBRID_RETRIEVAL] Hybrid search not available, falling back to enhanced retrieval")
            return enhanced_retrieval_v3(db, query_text, "", k, document_filter)
        
        # Import hybrid searcher
        from .hybrid_search import get_hybrid_searcher
        searcher = get_hybrid_searcher()
        
        # Perform hybrid search on default database
        results = searcher.hybrid_search(
            query=query_text,
            vector_store=db,
            k=k,
            keyword_weight=0.3,
            semantic_weight=0.7,
            rerank=True,
            filter_dict=document_filter
        )
        
        logger.info(f"[HYBRID_RETRIEVAL] Hybrid search returned {len(results)} results for default database")
        return results, "hybrid_search_default"
        
    except Exception as e:
        logger.error(f"[HYBRID_RETRIEVAL] Hybrid search failed for default database: {e}")
        # Fall back to enhanced retrieval
        return enhanced_retrieval_v3(db, query_text, "", k, document_filter)

def combined_search(query: str, user_id: Optional[str], search_scope: str, conversation_context: str, 
                   use_enhanced: bool = True, k: int = DEFAULT_SEARCH_K, document_id: str = None) -> Tuple[List, List[str], str]:
    """Combined search across all sources with hybrid search support"""
    all_results = []
    sources_searched = []
    retrieval_method = "basic"
    
    # Search default database
    if search_scope in ["all", "default_only"]:
        try:
            default_db = load_database()
            if default_db:
                # Try hybrid search first if available and enhanced search is requested
                if use_enhanced and FeatureFlags.HYBRID_SEARCH_AVAILABLE:
                    logger.info("[COMBINED_SEARCH] Using hybrid search for default database")
                    default_results, method = hybrid_retrieval_default(default_db, query, k=k)
                    retrieval_method = method
                elif use_enhanced:
                    logger.info("[COMBINED_SEARCH] Using enhanced retrieval v3 for default database")
                    default_results, method = enhanced_retrieval_v3(default_db, query, conversation_context, k=k)
                    retrieval_method = method
                else:
                    logger.info("[COMBINED_SEARCH] Using basic search for default database")
                    default_results = default_db.similarity_search_with_score(query, k=k)
                    retrieval_method = "basic_search"
                
                # Add source type metadata
                for doc, score in default_results:
                    doc.metadata['source_type'] = 'default_database'
                    all_results.append((doc, score))
                sources_searched.append("default_database")
                logger.info(f"[COMBINED_SEARCH] Added {len(default_results)} results from default database")
        except Exception as e:
            logger.error(f"Error searching default database: {e}")
    
    # Search user container
    if user_id and search_scope in ["all", "user_only"]:
        try:
            container_manager = get_container_manager()
            
            # Try hybrid search first if available
            if FeatureFlags.HYBRID_SEARCH_AVAILABLE and hasattr(container_manager, 'hybrid_search_user_container'):
                logger.info("[COMBINED_SEARCH] Using hybrid search for user container")
                user_results = container_manager.hybrid_search_user_container(
                    user_id, query, k=k, document_id=document_id
                )
                # Update retrieval method only if we haven't used hybrid for default DB
                if retrieval_method not in ["hybrid_search_default", "hybrid_search"]:
                    retrieval_method = "hybrid_search_user"
            elif use_enhanced:
                logger.info("[COMBINED_SEARCH] Using enhanced search for user container")
                user_results = container_manager.enhanced_search_user_container(
                    user_id, query, conversation_context, k=k, document_id=document_id
                )
                # Update retrieval method only if we haven't used a better method
                if retrieval_method not in ["hybrid_search_default", "hybrid_search", "enhanced_retrieval_v3"]:
                    retrieval_method = "enhanced_search_user"
            else:
                logger.info("[COMBINED_SEARCH] Using basic search for user container")
                user_results = container_manager.search_user_container(
                    user_id, query, k=k, document_id=document_id
                )
                if retrieval_method == "basic":
                    retrieval_method = "basic_search_user"
            
            # Add source type metadata
            for doc, score in user_results:
                doc.metadata['source_type'] = 'user_container'
                all_results.append((doc, score))
            if user_results:
                sources_searched.append("user_container")
                logger.info(f"[COMBINED_SEARCH] Added {len(user_results)} results from user container")
        except Exception as e:
            logger.error(f"Error searching user container: {e}")
    
    # Post-process results
    if use_enhanced or FeatureFlags.HYBRID_SEARCH_AVAILABLE:
        # Remove duplicates for enhanced/hybrid searches
        all_results = remove_duplicate_documents(all_results)
        logger.info(f"[COMBINED_SEARCH] Removed duplicates, {len(all_results)} unique results remaining")
    else:
        # Sort by score for basic search
        all_results.sort(key=lambda x: x[1], reverse=True)
    
    # Apply relevance threshold
    all_results = [(doc, score) for doc, score in all_results if score >= MIN_RELEVANCE_SCORE]
    
    # Limit to k results
    final_results = all_results[:k]
    
    logger.info(f"[COMBINED_SEARCH] Final search completed:")
    logger.info(f"  - Method: {retrieval_method}")
    logger.info(f"  - Sources: {sources_searched}")
    logger.info(f"  - Results: {len(final_results)}")
    logger.info(f"  - Hybrid available: {FeatureFlags.HYBRID_SEARCH_AVAILABLE}")
    
    return final_results, sources_searched, retrieval_method

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
