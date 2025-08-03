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

def normalize_score(score: float) -> float:
    """Normalize score to 0-1 range"""
    # Handle different score ranges that might come from different sources
    if score > 1.0:
        # Assume it's in 0-100 range or similar, normalize to 0-1
        if score <= 100.0:
            return score / 100.0
        else:
            # For very large scores, use log normalization
            return min(1.0, np.log(score + 1) / np.log(101))
    elif score < 0.0:
        # Handle negative scores (distance-based metrics)
        return max(0.0, 1.0 + score)
    else:
        # Already in 0-1 range
        return score

def normalize_results(results: List[Tuple]) -> List[Tuple]:
    """Normalize all scores in results to 0-1 range"""
    normalized_results = []
    for doc, score in results:
        normalized_score = normalize_score(score)
        normalized_results.append((doc, normalized_score))
    return normalized_results

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
        direct_results = normalize_results(direct_results)
        logger.info(f"[ENHANCED_RETRIEVAL] Direct search returned {len(direct_results)} results")
        
        expanded_query = f"{query_text} {conversation_history_context}"
        expanded_results = db.similarity_search_with_score(expanded_query, k=k, filter=document_filter)
        expanded_results = normalize_results(expanded_results)
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
            sq_results = normalize_results(sq_results)
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
        basic_results = normalize_results(basic_results)
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
            
            # Normalize scores immediately
            results = normalize_results(results)
            
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
        results = normalize_results(results)
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
    """Boost score for exact matches - ensures output stays in 0-1 range"""
    query_lower = query.lower()
    content_lower = content.lower()
    
    # Ensure base_score is normalized
    base_score = normalize_score(base_score)
    
    # Check for exact phrase match
    if query_lower in content_lower:
        boosted_score = base_score * SEARCH_CONFIG["boost_factor"]
        return min(1.0, boosted_score)
    
    # Check for all query terms present
    query_terms = extract_key_terms(query)
    if query_terms and all(term in content_lower for term in query_terms):
        boosted_score = base_score * 1.2
        return min(1.0, boosted_score)
    
    return base_score

def rerank_results(query: str, results: List[Tuple]) -> List[Tuple]:
    """Rerank results using cross-encoder if available - returns normalized scores"""
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
            # Ensure original score is normalized
            orig_score = normalize_score(orig_score)
            
            # Normalize rerank score to 0-1 range
            # MS-MARCO cross-encoder typically outputs scores around -10 to +10
            rerank_score = np.clip((rerank_scores[i] + 10) / 20, 0.0, 1.0)
            
            # Weighted combination - ensures result stays in 0-1
            final_score = 0.4 * orig_score + 0.6 * rerank_score
            final_score = np.clip(final_score, 0.0, 1.0)
            
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
        
        # Normalize scores from hybrid search
        results = normalize_results(results)
        
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
                    default_results = db.similarity_search_with_score(query, k=k)
                    default_results = normalize_results(default_results)
                    retrieval_method = "basic_search"
                
                # Add source type metadata
                for doc, score in default_results:
                    # Ensure score is normalized
                    score = normalize_score(score)
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
                # Normalize scores from user container hybrid search
                user_results = normalize_results(user_results)
                # Update retrieval method only if we haven't used hybrid for default DB
                if retrieval_method not in ["hybrid_search_default", "hybrid_search"]:
                    retrieval_method = "hybrid_search_user"
            elif use_enhanced:
                logger.info("[COMBINED_SEARCH] Using enhanced search for user container")
                user_results = container_manager.enhanced_search_user_container(
                    user_id, query, conversation_context, k=k, document_id=document_id
                )
                # Normalize scores from user container enhanced search
                user_results = normalize_results(user_results)
                # Update retrieval method only if we haven't used a better method
                if retrieval_method not in ["hybrid_search_default", "hybrid_search", "enhanced_retrieval_v3"]:
                    retrieval_method = "enhanced_search_user"
            else:
                logger.info("[COMBINED_SEARCH] Using basic search for user container")
                user_results = container_manager.search_user_container(
                    user_id, query, k=k, document_id=document_id
                )
                # Normalize scores from user container basic search
                user_results = normalize_results(user_results)
                if retrieval_method == "basic":
                    retrieval_method = "basic_search_user"
            
            # Add source type metadata
            for doc, score in user_results:
                # Ensure score is normalized
                score = normalize_score(score)
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
        # Sort by score for basic search (scores should already be normalized)
        all_results.sort(key=lambda x: x[1], reverse=True)
    
    # Apply relevance threshold (ensure MIN_RELEVANCE_SCORE is in 0-1 range)
    normalized_min_score = normalize_score(MIN_RELEVANCE_SCORE)
    all_results = [(doc, score) for doc, score in all_results if score >= normalized_min_score]
    
    # Limit to k results
    final_results = all_results[:k]
    
    logger.info(f"[COMBINED_SEARCH] Final search completed:")
    logger.info(f"  - Method: {retrieval_method}")
    logger.info(f"  - Sources: {sources_searched}")
    logger.info(f"  - Results: {len(final_results)}")
    logger.info(f"  - Hybrid available: {FeatureFlags.HYBRID_SEARCH_AVAILABLE}")
    logger.info(f"  - Score range: {[score for _, score in final_results[:3]]}")  # Log first 3 scores for debugging
    
    return final_results, sources_searched, retrieval_method

def calculate_confidence_score(results_with_scores: List[Tuple], response_length: int) -> float:
    """Calculate confidence score for results - expects normalized scores in 0-1 range"""
    try:
        if not results_with_scores:
            return 0.2
        
        # Ensure all scores are normalized
        normalized_results = normalize_results(results_with_scores)
        scores = [score for _, score in normalized_results]
        
        avg_relevance = np.mean(scores)
        doc_factor = min(1.0, len(scores) / 5.0)
        
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
        
        # Ensure confidence is in 0-1 range
        confidence = max(0.0, min(1.0, confidence))
        
        logger.debug(f"Confidence calculation: avg_relevance={avg_relevance:.3f}, "
                    f"doc_factor={doc_factor:.3f}, consistency={consistency_factor:.3f}, "
                    f"completeness={completeness_factor:.3f}, final={confidence:.3f}")
        
        return confidence
    
    except Exception as e:
        logger.error(f"Error calculating confidence score: {e}")
        return 0.5
