"""
Enhanced RAG (Retrieval-Augmented Generation) Operations Service

This service manages all aspects of document retrieval with sophisticated
retrieval strategies, confidence scoring, and backward compatibility.
"""
import os
import logging
import numpy as np
import time
from typing import List, Tuple, Dict, Optional, Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

from ..config import (
    DEFAULT_CHROMA_PATH, DEFAULT_SEARCH_K, ENHANCED_SEARCH_K,
    MIN_RELEVANCE_SCORE, SEARCH_CONFIG, CONFIDENCE_WEIGHTS, FeatureFlags
)
from ..core.dependencies import get_nlp
from ..core.exceptions import RetrievalError
from .container_manager import get_container_manager

logger = logging.getLogger(__name__)

class RAGService:
    """
    Enhanced RAG service with sophisticated retrieval strategies and confidence scoring.
    """
    
    def __init__(self):
        """Initialize the RAG service with all dependencies"""
        self.logger = logging.getLogger(f"{__name__}.RAGService")
        
        # Load core dependencies
        self._default_db = self._load_default_database()
        self._container_manager = get_container_manager()
        self._nlp = get_nlp()
        
        # Performance metrics
        self.search_stats = {
            'total_searches': 0,
            'avg_retrieval_time': 0.0,
            'successful_searches': 0,
            'failed_searches': 0
        }
        
        self.logger.info("✅ RAGService initialized successfully")
        self.logger.info(f"   Default DB: {'Available' if self._default_db else 'Not Available'}")
        self.logger.info(f"   NLP Model: {'Available' if self._nlp else 'Not Available'}")
    
    def _load_default_database(self) -> Optional[Chroma]:
        """Load the default shared database"""
        try:
            if not os.path.exists(DEFAULT_CHROMA_PATH):
                self.logger.warning(f"Default database path does not exist: {DEFAULT_CHROMA_PATH}")
                return None
            
            embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = Chroma(
                collection_name="default",
                embedding_function=embedding_function,
                persist_directory=DEFAULT_CHROMA_PATH
            )
            self.logger.info("✅ Default database loaded successfully")
            return db
        except Exception as e:
            self.logger.error(f"❌ Failed to load default database: {e}")
            return None
    
    def combined_search(self, query: str, user_id: Optional[str], search_scope: str, 
                       conversation_context: str, use_enhanced: bool = True, 
                       k: int = DEFAULT_SEARCH_K, document_id: str = None) -> Tuple[List, List[str], str]:
        """
        Enhanced combined search across all sources with intelligent strategy selection
        """
        start_time = time.time()
        all_results = []
        sources_searched = []
        retrieval_method = "enhanced_combined"
        
        self.logger.info(f"[COMBINED_SEARCH] Query: '{query[:100]}...', User: {user_id}, Scope: {search_scope}")
        
        try:
            # Search default database
            if search_scope in ["all", "default_only"] and self._default_db:
                default_results = self._search_database(
                    self._default_db, query, conversation_context, 
                    use_enhanced=use_enhanced, k=k
                )
                
                # Add source type metadata
                for doc, score in default_results:
                    doc.metadata['source_type'] = 'default_database'
                    all_results.append((doc, score))
                
                sources_searched.append("default_database")
                self.logger.info(f"[COMBINED_SEARCH] Default DB: {len(default_results)} results")
            
            # Search user container
            if user_id and search_scope in ["all", "user_only"]:
                user_results = self._search_user_container(
                    user_id, query, conversation_context,
                    use_enhanced=use_enhanced, k=k, document_id=document_id
                )
                
                # Add source type metadata
                for doc, score in user_results:
                    doc.metadata['source_type'] = 'user_container'
                    all_results.append((doc, score))
                
                if user_results:
                    sources_searched.append("user_container")
                    self.logger.info(f"[COMBINED_SEARCH] User container: {len(user_results)} results")
            
            # Post-process results
            if use_enhanced:
                # Remove duplicates
                all_results = self._remove_duplicates(all_results)
                retrieval_method = "enhanced_combined"
            else:
                # Basic sorting by score
                all_results.sort(key=lambda x: x[1], reverse=True)
                retrieval_method = "basic_combined"
            
            # Apply relevance threshold
            filtered_results = [(doc, score) for doc, score in all_results if score >= MIN_RELEVANCE_SCORE]
            final_results = filtered_results[:k]
            
            # Update statistics
            search_time = time.time() - start_time
            self._update_search_stats(search_time, True)
            
            self.logger.info(f"[COMBINED_SEARCH] Completed in {search_time:.3f}s: {len(final_results)} results")
            
            return final_results, sources_searched, retrieval_method
            
        except Exception as e:
            search_time = time.time() - start_time
            self._update_search_stats(search_time, False)
            self.logger.error(f"❌ Combined search failed: {e}")
            return [], sources_searched, "error"
    
    def _search_database(self, db: Chroma, query: str, context: str, 
                        use_enhanced: bool = True, k: int = DEFAULT_SEARCH_K) -> List[Tuple]:
        """Search a specific database with enhanced strategies"""
        
        if not use_enhanced:
            # Basic semantic search
            results = db.similarity_search_with_score(query, k=k)
            return self._normalize_results(results)
        
        # Enhanced search with multiple strategies
        return self._enhanced_retrieval(db, query, context, k)
    
    def _search_user_container(self, user_id: str, query: str, context: str,
                              use_enhanced: bool = True, k: int = DEFAULT_SEARCH_K,
                              document_id: str = None) -> List[Tuple]:
        """Search user's document container"""
        
        try:
            if use_enhanced:
                # Use container manager's enhanced search
                results = self._container_manager.enhanced_search_user_container(
                    user_id, query, context, k=k, document_id=document_id
                )
            else:
                # Use basic container search
                results = self._container_manager.search_user_container(
                    user_id, query, k=k, document_id=document_id
                )
            
            return self._normalize_results(results)
            
        except Exception as e:
            self.logger.error(f"❌ User container search failed: {e}")
            return []
    
    def _enhanced_retrieval(self, db: Chroma, query: str, context: str, k: int) -> List[Tuple]:
        """
        Enhanced retrieval with multiple strategies and query expansion
        """
        all_results = []
        seen_content = set()
        
        # Strategy 1: Direct semantic search
        direct_results = self._normalize_results(db.similarity_search_with_score(query, k=k))
        for doc, score in direct_results:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_content:
                all_results.append((doc, score))
                seen_content.add(content_hash)
        
        # Strategy 2: Expanded query search
        if context:
            expanded_query = f"{query} {context}"
            expanded_results = self._normalize_results(db.similarity_search_with_score(expanded_query, k=k))
            for doc, score in expanded_results:
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_content:
                    # Slightly lower score for expanded results
                    all_results.append((doc, score * 0.9))
                    seen_content.add(content_hash)
        
        # Strategy 3: Query expansion with key terms
        expanded_queries = self._expand_query_intelligently(query)
        for expanded_query in expanded_queries[1:3]:  # Skip original query, limit to 2 expansions
            try:
                exp_results = self._normalize_results(db.similarity_search_with_score(expanded_query, k=k//2))
                for doc, score in exp_results:
                    content_hash = hash(doc.page_content[:200])
                    if content_hash not in seen_content:
                        # Lower score for expansion results
                        all_results.append((doc, score * 0.8))
                        seen_content.add(content_hash)
            except Exception as e:
                self.logger.warning(f"Expanded query search failed: {e}")
        
        # Sort and filter results
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply relevance filtering
        filtered_results = [(doc, score) for doc, score in all_results if score >= MIN_RELEVANCE_SCORE]
        
        return filtered_results[:k]
    
    def _remove_duplicates(self, results: List[Tuple]) -> List[Tuple]:
        """Enhanced duplicate removal with content similarity"""
        if not results:
            return []
        
        unique_results = []
        seen_hashes = set()
        
        for doc, score in results:
            # Create content hash for deduplication
            content_sample = doc.page_content[:200].strip()
            content_hash = hash(content_sample)
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_results.append((doc, score))
        
        # Sort by score
        unique_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Removed {len(results) - len(unique_results)} duplicates")
        return unique_results
    
    def calculate_confidence_score(self, results_with_scores: List[Tuple], 
                                 response_length: int, additional_factors: Dict = None) -> float:
        """
        Enhanced confidence scoring with multiple factors
        """
        try:
            if not results_with_scores:
                return 0.1
            
            additional_factors = additional_factors or {}
            
            # Ensure all scores are normalized
            normalized_results = self._normalize_results(results_with_scores)
            scores = [score for _, score in normalized_results]
            
            # Factor 1: Average relevance
            avg_relevance = np.mean(scores)
            
            # Factor 2: Document count factor
            doc_factor = min(1.0, len(scores) / 5.0)
            
            # Factor 3: Score consistency
            if len(scores) > 1:
                score_std = np.std(scores)
                consistency_factor = max(0.3, 1.0 - score_std)
            else:
                consistency_factor = 0.7
            
            # Factor 4: Response completeness
            completeness_factor = min(1.0, response_length / 500.0)
            
            # Factor 5: Source diversity
            source_types = set()
            for doc, _ in normalized_results:
                source_type = doc.metadata.get('source_type', 'unknown')
                source_types.add(source_type)
            diversity_factor = min(1.0, len(source_types) / 2.0)
            
            # Factor 6: Quality of top results
            top_scores = scores[:3]
            top_quality_factor = np.mean(top_scores) if top_scores else 0.0
            
            # Weighted combination
            base_confidence = (
                avg_relevance * CONFIDENCE_WEIGHTS.get("relevance", 0.5) +
                doc_factor * CONFIDENCE_WEIGHTS.get("document_count", 0.2) +
                consistency_factor * CONFIDENCE_WEIGHTS.get("consistency", 0.2) +
                completeness_factor * CONFIDENCE_WEIGHTS.get("completeness", 0.1)
            )
            
            # Apply additional factors
            enhanced_confidence = (
                base_confidence * 0.7 +
                diversity_factor * 0.1 +
                top_quality_factor * 0.2
            )
            
            # Apply external factors if provided
            if additional_factors:
                external_factor = additional_factors.get('external_sources_found', 0) * 0.1
                enhanced_confidence = min(1.0, enhanced_confidence + external_factor)
            
            # Ensure bounds
            final_confidence = max(0.0, min(1.0, enhanced_confidence))
            
            self.logger.debug(f"Confidence calculation: avg_rel={avg_relevance:.3f}, "
                            f"doc_factor={doc_factor:.3f}, consistency={consistency_factor:.3f}, "
                            f"completeness={completeness_factor:.3f}, final={final_confidence:.3f}")
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {e}")
            return 0.5  # Safe default
    
    # --- Utility Methods ---
    
    @staticmethod
    def _normalize_score(score: float, metric: str = "cosine") -> float:
        """Normalize score to 0-1 range based on metric type"""
        if metric == "cosine":
            # Cosine similarity: -1 to 1 range
            return max(0.0, min(1.0, (score + 1) / 2))
        elif metric == "l2":
            # L2 distance: convert to similarity
            return max(0.0, min(1.0, 1.0 / (1.0 + score)))
        elif metric == "euclidean":
            # Euclidean distance: convert to similarity
            return max(0.0, min(1.0, 1.0 / (1.0 + score)))
        else:
            # Assume already normalized
            return max(0.0, min(1.0, score))
    
    def _normalize_results(self, results: List[Tuple], metric: str = "cosine") -> List[Tuple]:
        """Apply score normalization to results"""
        return [(doc, self._normalize_score(score, metric)) for doc, score in results]
    
    def _expand_query_intelligently(self, query: str) -> List[str]:
        """Generate multiple query variants for better search recall"""
        expanded_queries = {query}  # Use set for automatic deduplication
        
        # Add question variation
        if not query.endswith('?'):
            expanded_queries.add(query + '?')
        
        # Handle bill/statute queries specially
        import re
        bill_match = re.search(r'(HB|SB|SSB|ESSB|SHB|ESHB)\s*(\d+)', query, re.IGNORECASE)
        if bill_match:
            bill_type = bill_match.group(1).upper()
            bill_num = bill_match.group(2)
            
            # Add variations
            expanded_queries.update([
                f"{bill_type} {bill_num}",
                f"{bill_type}{bill_num}",
                f"House Bill {bill_num}" if bill_type.startswith('H') else f"Senate Bill {bill_num}",
                f"bill {bill_num}"
            ])
        
        # Extract key entities for expansion using NLP
        if self._nlp:
            try:
                doc = self._nlp(query)
                for ent in doc.ents:
                    if ent.label_ in ["ORG", "PERSON", "LAW", "DATE"] and len(ent.text) > 2:
                        expanded_queries.add(ent.text)
                        expanded_queries.add(f"What is {ent.text}?")
            except Exception as e:
                self.logger.warning(f"NLP expansion failed: {e}")
        
        # Extract key terms for focused query
        key_terms = self._extract_key_terms(query)
        if len(key_terms) > 1:
            expanded_queries.add(' '.join(key_terms))
        
        return list(expanded_queries)
    
    @staticmethod
    def _extract_key_terms(query: str) -> List[str]:
        """Extract key terms by removing stop words"""
        import re
        stop_words = {
            'what', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'about', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between', 'under',
            'can', 'will', 'would', 'should', 'could', 'may', 'might', 'must',
            'do', 'does', 'did', 'have', 'has', 'had', 'be', 'am', 'is', 'are',
            'was', 'were', 'been', 'being'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]
        
        return key_terms
    
    def _update_search_stats(self, search_time: float, success: bool):
        """Update search performance statistics"""
        self.search_stats['total_searches'] += 1
        
        if success:
            self.search_stats['successful_searches'] += 1
        else:
            self.search_stats['failed_searches'] += 1
        
        # Update average retrieval time
        total = self.search_stats['total_searches']
        current_avg = self.search_stats['avg_retrieval_time']
        self.search_stats['avg_retrieval_time'] = ((current_avg * (total - 1)) + search_time) / total
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get current search performance statistics"""
        success_rate = (self.search_stats['successful_searches'] / 
                       max(1, self.search_stats['total_searches']))
        
        return {
            **self.search_stats,
            'success_rate': success_rate,
            'features_available': {
                'default_database': self._default_db is not None,
                'nlp_model': self._nlp is not None,
                'hybrid_search': FeatureFlags.HYBRID_SEARCH_AVAILABLE
            }
        }

# --- Global Instance and Backward Compatible Functions ---

_rag_service = None

def get_rag_service() -> RAGService:
    """Get or create global RAG service instance"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service

# --- BACKWARD COMPATIBLE FUNCTIONS ---
# These maintain the exact interface your existing code expects

def load_database():
    """BACKWARD COMPATIBLE: Load the default database"""
    rag_service = get_rag_service()
    return rag_service._default_db

def enhanced_retrieval_v2(db, query_text: str, conversation_history_context: str, 
                         k: int = ENHANCED_SEARCH_K, document_filter: Dict = None) -> Tuple[List, str]:
    """BACKWARD COMPATIBLE: Enhanced retrieval with multiple strategies"""
    
    rag_service = get_rag_service()
    
    try:
        results = rag_service._enhanced_retrieval(db, query_text, conversation_history_context, k)
        
        # Apply document filter if provided
        if document_filter:
            filtered_results = []
            for doc, score in results:
                match = all(doc.metadata.get(k) == v for k, v in document_filter.items())
                if match:
                    filtered_results.append((doc, score))
            results = filtered_results
        
        return results, "enhanced_retrieval_v2"
        
    except Exception as e:
        logger.error(f"Enhanced retrieval failed: {e}")
        # Fallback to basic search
        try:
            basic_results = db.similarity_search_with_score(query_text, k=k, filter=document_filter)
            return rag_service._normalize_results(basic_results), "basic_fallback"
        except Exception as fallback_error:
            logger.error(f"Fallback search also failed: {fallback_error}")
            return [], "search_failed"

def combined_search(query: str, user_id: Optional[str], search_scope: str, 
                   conversation_context: str, use_enhanced: bool = True, 
                   k: int = DEFAULT_SEARCH_K, document_id: str = None) -> Tuple[List, List[str], str]:
    """BACKWARD COMPATIBLE: Combined search across all sources"""
    
    rag_service = get_rag_service()
    return rag_service.combined_search(
        query, user_id, search_scope, conversation_context,
        use_enhanced, k, document_id
    )

def calculate_confidence_score(results_with_scores: List[Tuple], response_length: int) -> float:
    """BACKWARD COMPATIBLE: Calculate confidence score for results"""
    
    rag_service = get_rag_service()
    return rag_service.calculate_confidence_score(results_with_scores, response_length)

def remove_duplicate_documents(results_with_scores: List[Tuple]) -> List[Tuple]:
    """BACKWARD COMPATIBLE: Remove duplicate documents from search results"""
    
    rag_service = get_rag_service()
    return rag_service._remove_duplicates(results_with_scores)

def hybrid_retrieval_default(db, query_text: str, k: int = ENHANCED_SEARCH_K, 
                           document_filter: Dict = None) -> Tuple[List, str]:
    """BACKWARD COMPATIBLE: Hybrid retrieval for default database"""
    
    # For now, fall back to enhanced retrieval
    return enhanced_retrieval_v2(db, query_text, "", k, document_filter)

# --- Performance Monitoring ---

def get_rag_performance_metrics() -> Dict[str, Any]:
    """Get comprehensive RAG performance metrics"""
    
    rag_service = get_rag_service()
    
    return {
        'service_stats': rag_service.get_search_stats(),
        'configuration': {
            'default_search_k': DEFAULT_SEARCH_K,
            'enhanced_search_k': ENHANCED_SEARCH_K,
            'min_relevance_score': MIN_RELEVANCE_SCORE,
            'search_config': SEARCH_CONFIG,
            'confidence_weights': CONFIDENCE_WEIGHTS
        },
        'feature_availability': {
            'default_database_available': rag_service._default_db is not None,
            'nlp_available': rag_service._nlp is not None,
            'hybrid_search_available': FeatureFlags.HYBRID_SEARCH_AVAILABLE
        }
    }
