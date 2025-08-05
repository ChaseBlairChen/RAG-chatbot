# legal_assistant/services/rag_service.py - COMPLETE ENHANCED VERSION
"""
Enhanced RAG (Retrieval-Augmented Generation) Operations Service

This service manages all aspects of document retrieval with class-based architecture,
sophisticated retrieval strategies, confidence scoring, and backward compatibility.
"""
import os
import logging
import numpy as np
import re
import math
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
    Enhanced RAG service with class-based architecture and sophisticated retrieval strategies.
    
    Features:
    - Multiple retrieval strategies (semantic, hybrid, query expansion)
    - Confidence scoring and quality assessment
    - Reranking with cross-encoder models
    - Backward compatibility with existing functions
    """
    
    def __init__(self):
        """Initialize the RAG service with all dependencies"""
        self.logger = logging.getLogger(f"{__name__}.RAGService")
        
        # Load core dependencies
        self._default_db = self._load_default_database()
        self._reranker = self._load_reranker() if SEARCH_CONFIG.get("rerank_enabled", False) else None
        self._container_manager = get_container_manager()
        self._nlp = get_nlp()
        
        # Initialize text processor for enhanced extraction
        self._init_text_processor()
        
        # Performance metrics
        self.search_stats = {
            'total_searches': 0,
            'avg_retrieval_time': 0.0,
            'rerank_usage': 0,
            'hybrid_usage': 0
        }
        
        self.logger.info("✅ RAGService initialized successfully")
        self.logger.info(f"   Default DB: {'Available' if self._default_db else 'Not Available'}")
        self.logger.info(f"   Reranker: {'Available' if self._reranker else 'Not Available'}")
        self.logger.info(f"   NLP Model: {'Available' if self._nlp else 'Not Available'}")
    
    def _init_text_processor(self):
        """Initialize text processor for enhanced extraction"""
        try:
            from ..utils.text_processing import get_text_processor
            self._text_processor = get_text_processor()
            self.logger.info("✅ Text processor available for enhanced extraction")
        except ImportError:
            self._text_processor = None
            self.logger.warning("⚠️ Enhanced text processor not available")
    
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
    
    def _load_reranker(self) -> Optional[Any]:
        """Load cross-encoder for reranking if available"""
        try:
            from sentence_transformers import CrossEncoder
            reranker_model = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
            reranker = CrossEncoder(reranker_model)
            self.logger.info(f"✅ Reranker model '{reranker_model}' loaded")
            return reranker
        except ImportError:
            self.logger.warning("⚠️ CrossEncoder not available - install sentence-transformers[cross-encoder]")
            return None
        except Exception as e:
            self.logger.error(f"❌ Failed to load reranker: {e}")
            return None
    
    # --- Main Search Methods ---
    
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
                # Remove duplicates and rerank
                all_results = self._remove_duplicates(all_results)
                
                if self._reranker and len(all_results) > 1:
                    all_results = self._rerank_results(query, all_results)
                    retrieval_method = "enhanced_combined_reranked"
                    self.search_stats['rerank_usage'] += 1
            else:
                # Basic sorting by score
                all_results.sort(key=lambda x: x[1], reverse=True)
                retrieval_method = "basic_combined"
            
            # Apply relevance threshold
            filtered_results = [(doc, score) for doc, score in all_results if score >= MIN_RELEVANCE_SCORE]
            final_results = filtered_results[:k]
            
            # Update statistics
            search_time = time.time() - start_time
            self._update_search_stats(search_time, retrieval_method)
            
            self.logger.info(f"[COMBINED_SEARCH] Completed in {search_time:.3f}s: {len(final_results)} results")
            
            return final_results, sources_searched, retrieval_method
            
        except Exception as e:
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
        
        # Strategy 4: Sub-query search using NLP entities
        if self._nlp:
            try:
                doc = self._nlp(query)
                for ent in doc.ents[:3]:  # Limit to 3 entities
                    if ent.label_ in ["ORG", "PERSON", "LAW", "DATE"] and len(ent.text) > 2:
                        entity_query = f"What is {ent.text}?"
                        ent_results = self._normalize_results(db.similarity_search_with_score(entity_query, k=3))
                        for doc, score in ent_results:
                            content_hash = hash(doc.page_content[:200])
                            if content_hash not in seen_content:
                                # Lower score for entity results
                                all_results.append((doc, score * 0.7))
                                seen_content.add(content_hash)
            except Exception as e:
                self.logger.warning(f"Entity-based search failed: {e}")
        
        # Sort and filter results
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply relevance filtering
        filtered_results = [(doc, score) for doc, score in all_results if score >= MIN_RELEVANCE_SCORE]
        
        return filtered_results[:k]
    
    def _remove_duplicates(self, results: List[Tuple]) -> List[Tuple]:
        """Enhanced duplicate removal with content similarity"""
        if not results:
            return []
        
        if self._text_processor:
            return self._text_processor.remove_duplicates(results)
        else:
            # Fallback duplicate removal
            unique_results = []
            seen_hashes = set()
            
            for doc, score in results:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    unique_results.append((doc, score))
            
            return unique_results
    
    def _rerank_results(self, query: str, results: List[Tuple]) -> List[Tuple]:
        """Rerank results using cross-encoder if available"""
        if not self._reranker or len(results) <= 1:
            return results
        
        try:
            pairs = [[query, doc.page_content] for doc, _ in results]
            rerank_scores = self._reranker.predict(pairs)
            
            # Normalize rerank scores to 0-1
            if len(rerank_scores) > 1:
                min_score = rerank_scores.min()
                max_score = rerank_scores.max()
                if max_score > min_score:
                    rerank_scores = (rerank_scores - min_score) / (max_score - min_score)
                else:
                    rerank_scores = np.ones_like(rerank_scores) * 0.5
            else:
                rerank_scores = np.array([0.5])
            
            # Combine original and rerank scores
            reranked = []
            for i, (doc, orig_score) in enumerate(results):
                # Weighted combination
                final_score = (
                    SEARCH_CONFIG.get("semantic_weight", 0.4) * orig_score +
                    SEARCH_CONFIG.get("rerank_weight", 0.6) * rerank_scores[i]
                )
                final_score = np.clip(final_score, 0.0, 1.0)
                reranked.append((doc, final_score))
            
            # Sort by final score
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            self.logger.debug(f"Reranked {len(results)} results")
            return reranked
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            return results
    
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
            
            # Factor 5: Source diversity (new)
            source_types = set()
            for doc, _ in normalized_results:
                source_type = doc.metadata.get('source_type', 'unknown')
                source_types.add(source_type)
            diversity_factor = min(1.0, len(source_types) / 2.0)
            
            # Factor 6: Quality of top results (new)
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
    
    def hybrid_search_if_available(self, user_id: str, query: str, k: int = DEFAULT_SEARCH_K,
                                  document_id: str = None) -> Tuple[List[Tuple], str]:
        """Perform hybrid search if available, otherwise fall back to enhanced search"""
        
        if not FeatureFlags.HYBRID_SEARCH_AVAILABLE:
            self.logger.info("Hybrid search not available, using enhanced semantic search")
            return self._search_user_container(user_id, query, "", k=k, document_id=document_id), "enhanced_semantic"
        
        try:
            # Try hybrid search
            user_db = self._container_manager.get_user_database_safe(user_id)
            if not user_db:
                return [], "no_user_database"
            
            from .hybrid_search import get_hybrid_searcher
            searcher = get_hybrid_searcher()
            
            filter_dict = {"file_id": document_id} if document_id else None
            
            results = searcher.hybrid_search(
                query=query,
                vector_store=user_db,
                k=k,
                keyword_weight=SEARCH_CONFIG.get("keyword_weight", 0.3),
                semantic_weight=SEARCH_CONFIG.get("semantic_weight", 0.7),
                rerank=SEARCH_CONFIG.get("rerank_enabled", False),
                filter_dict=filter_dict
            )
            
            normalized_results = self._normalize_results(results)
            self.search_stats['hybrid_usage'] += 1
            
            self.logger.info(f"Hybrid search returned {len(normalized_results)} results")
            return normalized_results, "hybrid_search"
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed, falling back to enhanced search: {e}")
            return self._search_user_container(user_id, query, "", k=k, document_id=document_id), "enhanced_fallback"
    
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
    
    def _update_search_stats(self, search_time: float, method: str):
        """Update search performance statistics"""
        self.search_stats['total_searches'] += 1
        
        # Update average retrieval time
        total = self.search_stats['total_searches']
        current_avg = self.search_stats['avg_retrieval_time']
        self.search_stats['avg_retrieval_time'] = ((current_avg * (total - 1)) + search_time) / total
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get current search performance statistics"""
        return {
            **self.search_stats,
            'rerank_usage_rate': (self.search_stats['rerank_usage'] / 
                                max(1, self.search_stats['total_searches'])),
            'hybrid_usage_rate': (self.search_stats['hybrid_usage'] / 
                                max(1, self.search_stats['total_searches'])),
            'features_available': {
                'default_database': self._default_db is not None,
                'reranker': self._reranker is not None,
                'nlp_model': self._nlp is not None,
                'text_processor': self._text_processor is not None,
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
        basic_results = db.similarity_search_with_score(query_text, k=k, filter=document_filter)
        return rag_service._normalize_results(basic_results), "basic_fallback"

def enhanced_retrieval_v3(db, query_text: str, conversation_history_context: str, 
                         k: int = ENHANCED_SEARCH_K, document_filter: Dict = None) -> Tuple[List, str]:
    """BACKWARD COMPATIBLE: Most advanced retrieval with query understanding"""
    
    rag_service = get_rag_service()
    
    try:
        # Use the enhanced retrieval method
        results = rag_service._enhanced_retrieval(db, query_text, conversation_history_context, k)
        
        # Apply document filter
        if document_filter:
            filtered_results = []
            for doc, score in results:
                match = all(doc.metadata.get(k) == v for k, v in document_filter.items())
                if match:
                    filtered_results.append((doc, score))
            results = filtered_results
        
        # Apply reranking if available
        if rag_service._reranker and len(results) > 1:
            results = rag_service._rerank_results(query_text, results)
            return results, "enhanced_retrieval_v3_reranked"
        
        return results, "enhanced_retrieval_v3"
        
    except Exception as e:
        logger.error(f"Enhanced retrieval v3 failed: {e}")
        return enhanced_retrieval_v2(db, query_text, conversation_history_context, k, document_filter)

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
    
    rag_service = get_rag_service()
    
    try:
        if not FeatureFlags.HYBRID_SEARCH_AVAILABLE:
            return enhanced_retrieval_v3(db, query_text, "", k, document_filter)
        
        # This would require implementing hybrid search for arbitrary databases
        # For now, fall back to enhanced retrieval
        return enhanced_retrieval_v3(db, query_text, "", k, document_filter)
        
    except Exception as e:
        logger.error(f"Hybrid retrieval failed: {e}")
        return enhanced_retrieval_v2(db, query_text, "", k, document_filter)

# --- Advanced Features ---

def search_with_custom_strategy(query: str, user_id: str, strategy_config: Dict) -> Tuple[List, str]:
    """NEW: Search with custom strategy configuration"""
    
    rag_service = get_rag_service()
    
    # Extract strategy parameters
    k = strategy_config.get('k', DEFAULT_SEARCH_K)
    use_reranking = strategy_config.get('use_reranking', True)
    semantic_weight = strategy_config.get('semantic_weight', 0.7)
    keyword_weight = strategy_config.get('keyword_weight', 0.3)
    
    try:
        # Perform search with custom configuration
        results = rag_service._search_user_container(
            user_id, query, "", use_enhanced=True, k=k*2  # Get more for reranking
        )
        
        # Apply custom reranking if requested
        if use_reranking and rag_service._reranker and len(results) > 1:
            results = rag_service._rerank_results(query, results)
        
        # Apply custom filtering
        min_score = strategy_config.get('min_relevance_score', MIN_RELEVANCE_SCORE)
        filtered_results = [(doc, score) for doc, score in results if score >= min_score]
        
        return filtered_results[:k], f"custom_strategy_k{k}"
        
    except Exception as e:
        logger.error(f"Custom strategy search failed: {e}")
        return [], "custom_strategy_error"

def get_retrieval_diagnostics(query: str, user_id: str) -> Dict[str, Any]:
    """NEW: Get detailed diagnostics for a query"""
    
    rag_service = get_rag_service()
    
    diagnostics = {
        'query': query,
        'user_id': user_id,
        'timestamp': time.time(),
        'query_analysis': {},
        'retrieval_analysis': {},
        'recommendations': []
    }
    
    # Analyze query
    diagnostics['query_analysis'] = {
        'length': len(query),
        'word_count': len(query.split()),
        'has_legal_terms': bool(re.search(r'\b(?:USC|CFR|RCW|WAC|HB|SB)\s+\d+', query)),
        'has_questions': '?' in query,
        'key_terms': rag_service._extract_key_terms(query),
        'expanded_queries': rag_service._expand_query_intelligently(query)
    }
    
    # Test retrieval
    try:
        start_time = time.time()
        results, sources, method = rag_service.combined_search(
            query, user_id, "all", "", use_enhanced=True, k=10
        )
        retrieval_time = time.time() - start_time
        
        diagnostics['retrieval_analysis'] = {
            'results_found': len(results),
            'sources_searched': sources,
            'retrieval_method': method,
            'retrieval_time': retrieval_time,
            'score_range': [min(scores := [s for _, s in results]), max(scores)] if results else [0, 0],
            'avg_score': np.mean([s for _, s in results]) if results else 0,
            'above_threshold': len([s for _, s in results if s >= MIN_RELEVANCE_SCORE])
        }
        
        # Generate recommendations
        if len(results) == 0:
            diagnostics['recommendations'].extend([
                "Try broader search terms",
                "Check if documents are uploaded and processed",
                "Consider searching external databases"
            ])
        elif len(results) < 3:
            diagnostics['recommendations'].extend([
                "Try adding synonyms or related terms",
                "Use more specific legal terminology",
                "Check document chunking quality"
            ])
        else:
            diagnostics['recommendations'].append("Good retrieval results - query is well-formed")
    
    except Exception as e:
        diagnostics['retrieval_analysis'] = {'error': str(e)}
        diagnostics['recommendations'].append("Retrieval system error - check logs")
    
    return diagnostics

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
            'reranker_available': rag_service._reranker is not None,
            'nlp_available': rag_service._nlp is not None,
            'hybrid_search_available': FeatureFlags.HYBRID_SEARCH_AVAILABLE
        }
    }

"""
USAGE EXAMPLES:

# Basic usage (backward compatible)
results, sources, method = combined_search(query, user_id, "all", context)
confidence = calculate_confidence_score(results, len(response))

# Enhanced usage (new features)
rag_service = get_rag_service()
results, method = rag_service.hybrid_search_if_available(user_id, query)
diagnostics = get_retrieval_diagnostics(query, user_id)
performance = get_rag_performance_metrics()

# Custom strategy usage
strategy = {'k': 20, 'use_reranking': True, 'min_relevance_score': 0.5}
results, method = search_with_custom_strategy(query, user_id, strategy)

MIGRATION BENEFITS:
✅ Class-based architecture for better maintainability
✅ Enhanced retrieval strategies with multiple approaches
✅ Sophisticated confidence scoring with multiple factors
✅ Performance monitoring and diagnostics
✅ Backward compatibility with all existing code
✅ Advanced features available for optimization
✅ Better error handling and graceful degradation
"""
