"""
RAG (Retrieval-Augmented Generation) Operations Service

This service manages all aspects of document retrieval for a RAG system.
It supports multiple databases (default and user-specific), sophisticated
retrieval strategies (semantic, hybrid, query expansion), and a confidence
scoring mechanism to evaluate the quality of retrieved results.
"""
import os
import logging
import numpy as np
import re
import math
from typing import List, Tuple, Dict, Optional, Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from sentence_transformers import CrossEncoder

from ..config import (
    DEFAULT_CHROMA_PATH, DEFAULT_SEARCH_K,
    MIN_RELEVANCE_SCORE, SEARCH_CONFIG, CONFIDENCE_WEIGHTS
)
from ..core.dependencies import get_nlp
from ..core.exceptions import RetrievalError
from .container_manager import get_container_manager
from ..utils.text_processing import remove_duplicate_documents

logger = logging.getLogger(__name__)

class RAGService:
    """
    Core service for managing Retrieval-Augmented Generation operations.

    This class encapsulates logic for loading databases, executing various
    retrieval strategies, reranking results, and calculating a confidence score.
    """
    
    def __init__(self):
        """
        Initializes the RAGService. Loads the default database and a
        cross-encoder for reranking if configured.
        """
        self._default_db = self._load_default_database()
        self._reranker = self._load_reranker()
        self._container_manager = get_container_manager()
        self._nlp = get_nlp()
        logger.info("RAGService initialized.")
        
    def _load_default_database(self) -> Optional[Chroma]:
        """
        Loads the default shared database from the configured path.
        
        Returns:
            A Chroma database instance or None if loading fails.
        """
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
            logger.debug("Default database loaded successfully.")
            return db
        except Exception as e:
            logger.error(f"Failed to load default database: {e}")
            return None

    def _load_reranker(self) -> Optional[CrossEncoder]:
        """
        Loads a cross-encoder model for reranking if enabled in config.
        
        Returns:
            A CrossEncoder instance or None if unavailable.
        """
        if not SEARCH_CONFIG.get("rerank_enabled", False):
            logger.info("Reranking is disabled by configuration.")
            return None
        
        try:
            reranker_model = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
            reranker = CrossEncoder(reranker_model)
            logger.info(f"✅ Reranker model '{reranker_model}' loaded successfully.")
            return reranker
        except ImportError:
            logger.warning("Cross-encoder library not installed. Reranking will be skipped.")
            return None
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            return None

    @staticmethod
    def _normalize_score(score: float, metric: str = "cosine") -> float:
        """
        Normalizes a retrieval score to a 0-1 range based on the metric type.
        
        Args:
            score: The raw score from the retrieval engine.
            metric: The distance metric used (e.g., 'cosine', 'l2').
            
        Returns:
            The normalized score.
        """
        if metric == "cosine":
            # Cosine similarity is already in a -1 to 1 range, so we shift and scale it
            return (score + 1) / 2
        elif metric == "l2":
            # L2 distance is typically non-negative. We can use an inverse scaling.
            return 1.0 / (1.0 + score)
        else:
            # For other metrics, we assume a 0-1 range and handle edge cases.
            return max(0.0, min(1.0, score))

    def _normalize_results(self, results: List[Tuple], metric: str = "cosine") -> List[Tuple]:
        """
        Applies score normalization to a list of results.
        """
        return [(doc, self._normalize_score(score, metric)) for doc, score in results]

    def _expand_query_intelligently(self, query: str) -> List[str]:
        """
        Generates multiple query variants for better search recall.
        
        Args:
            query: The original user query.
            
        Returns:
            A list of unique query strings.
        """
        expanded_queries = {query} # Use a set to handle duplicates automatically

        if self._nlp:
            doc = self._nlp(query)
            # Add named entities as separate search queries
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PERSON", "LAW", "DATE"]:
                    expanded_queries.add(ent.text)

        # Handle bill/statute queries specially
        bill_match = re.search(r'(HB|SB|SSB|ESSB|SHB|ESHB)\s*(\d+)', query, re.IGNORECASE)
        if bill_match:
            bill_type = bill_match.group(1).upper()
            bill_num = bill_match.group(2)
            expanded_queries.add(f"{bill_type} {bill_num}")
            expanded_queries.add(f"{bill_type}{bill_num}")
            
        # Add a keyword-only query
        key_terms = self._extract_key_terms(query)
        if key_terms:
            expanded_queries.add(' '.join(key_terms))
            
        return list(expanded_queries)

    @staticmethod
    def _extract_key_terms(query: str) -> List[str]:
        """Simple extraction of key terms by removing common stop words."""
        stop_words = {'what', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                      'to', 'for', 'of', 'with', 'by', 'from', 'about', 'into', 'through'}
        words = re.findall(r'\b\w+\b', query.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]

    def _rerank_results(self, query: str, results: List[Tuple]) -> List[Tuple]:
        """
        Reranks a list of retrieved documents using a cross-encoder model.
        
        Args:
            query: The original user query.
            results: A list of (Document, score) tuples.
            
        Returns:
            The reranked list of (Document, score) tuples.
        """
        if not self._reranker or not results:
            return results

        try:
            pairs = [[query, doc.page_content] for doc, _ in results]
            rerank_scores = self._reranker.predict(pairs)
            
            reranked = []
            for i, (doc, orig_score) in enumerate(results):
                # Normalize rerank score to a 0-1 range
                normalized_rerank_score = (rerank_scores[i] - rerank_scores.min()) / (rerank_scores.max() - rerank_scores.min())
                
                final_score = (
                    (SEARCH_CONFIG.get("semantic_weight", 0.4) * orig_score) +
                    (SEARCH_CONFIG.get("rerank_weight", 0.6) * normalized_rerank_score)
                )
                final_score = np.clip(final_score, 0.0, 1.0)
                reranked.append((doc, final_score))

            reranked.sort(key=lambda x: x[1], reverse=True)
            logger.debug("Results reranked successfully.")
            return reranked
        except Exception as e:
            logger.error(f"Reranking failed, returning original results: {e}")
            return results

    def _execute_search_strategy(self, db: Chroma, query: str, context: str, k: int, filter_dict: Optional[Dict]) -> List[Tuple]:
        """
        Executes an enhanced search strategy (query expansion, sub-queries, etc.).
        """
        all_results = []
        seen_content = set()
        
        # Strategy 1: Direct semantic search
        direct_results = self._normalize_results(db.similarity_search_with_score(query, k=k, filter=filter_dict))
        for doc, score in direct_results:
            if doc.page_content not in seen_content:
                all_results.append((doc, score))
                seen_content.add
