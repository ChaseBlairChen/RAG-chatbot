"""Hybrid search combining keyword and semantic search"""
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

class HybridSearcher:
    """Combines keyword search (BM25) with semantic search for better retrieval"""
    
    def __init__(self):
        self.reranker = None
        self._initialize_reranker()
    
    def _initialize_reranker(self):
        """Initialize the cross-encoder for reranking"""
        try:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("✅ Cross-encoder reranker initialized")
        except Exception as e:
            logger.warning(f"Could not initialize reranker: {e}")
            self.reranker = None
    
    def hybrid_search(self, 
                     query: str, 
                     vector_store,
                     k: int = 10,
                     keyword_weight: float = 0.3,
                     semantic_weight: float = 0.7,
                     rerank: bool = True,
                     filter_dict: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """
        Perform hybrid search combining keyword and semantic search
        
        Args:
            query: Search query
            vector_store: ChromaDB vector store
            k: Number of results to return
            keyword_weight: Weight for keyword search (0-1)
            semantic_weight: Weight for semantic search (0-1)
            rerank: Whether to use cross-encoder reranking
            filter_dict: Optional filter for search
            
        Returns:
            List of (document, score) tuples
        """
        try:
            # Step 1: Get all documents for keyword search
            all_docs = vector_store.get()
            
            if not all_docs or 'documents' not in all_docs:
                logger.warning("No documents found in vector store")
                return []
            
            # Apply filter if provided
            if filter_dict:
                filtered_indices = []
                for i, metadata in enumerate(all_docs.get('metadatas', [])):
                    match = all([metadata.get(k) == v for k, v in filter_dict.items()])
                    if match:
                        filtered_indices.append(i)
                
                if not filtered_indices:
                    logger.warning("No documents match the filter criteria")
                    return []
                
                # Filter documents
                documents = [all_docs['documents'][i] for i in filtered_indices]
                metadatas = [all_docs['metadatas'][i] for i in filtered_indices]
                ids = [all_docs['ids'][i] for i in filtered_indices]
            else:
                documents = all_docs['documents']
                metadatas = all_docs.get('metadatas', [{}] * len(documents))
                ids = all_docs.get('ids', list(range(len(documents))))
            
            # Create Document objects
            doc_objects = []
            for i, (content, metadata, doc_id) in enumerate(zip(documents, metadatas, ids)):
                doc = Document(page_content=content, metadata=metadata or {})
                doc.metadata['_id'] = doc_id
                doc_objects.append(doc)
            
            # Step 2: Keyword search with BM25
            keyword_scores = self._bm25_search(query, doc_objects)
            
            # Step 3: Semantic search
            semantic_results = vector_store.similarity_search_with_score(
                query, 
                k=min(k * 3, len(documents)),  # Get more for better reranking
                filter=filter_dict
            )
            
            # Create mapping of document content to semantic scores
            semantic_scores_map = {}
            for doc, score in semantic_results:
                # Use first 100 chars as key to handle duplicates
                key = doc.page_content[:100]
                semantic_scores_map[key] = 1 - score  # Convert distance to similarity
            
            # Step 4: Combine scores
            combined_results = []
            
            for i, doc in enumerate(doc_objects):
                key = doc.page_content[:100]
                
                # Get keyword score
                keyword_score = keyword_scores[i]
                
                # Get semantic score
                semantic_score = semantic_scores_map.get(key, 0.0)
                
                # Combine scores
                combined_score = (keyword_weight * keyword_score + 
                                semantic_weight * semantic_score)
                
                combined_results.append((doc, combined_score))
            
            # Sort by combined score
            combined_results.sort(key=lambda x: x[1], reverse=True)
            
            # Step 5: Rerank top results if reranker available
            if rerank and self.reranker and len(combined_results) > 0:
                top_results = combined_results[:min(k * 2, len(combined_results))]
                reranked_results = self._rerank_results(query, top_results)
                return reranked_results[:k]
            else:
                return combined_results[:k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fall back to pure semantic search
            try:
                return vector_store.similarity_search_with_score(query, k=k, filter=filter_dict)
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
                return []
    
    def _bm25_search(self, query: str, documents: List[Document]) -> np.ndarray:
        """Perform BM25 keyword search"""
        try:
            # Tokenize documents
            tokenized_docs = []
            for doc in documents:
                # Simple tokenization - could be improved
                tokens = doc.page_content.lower().split()
                tokenized_docs.append(tokens)
            
            # Initialize BM25
            bm25 = BM25Okapi(tokenized_docs)
            
            # Tokenize query
            tokenized_query = query.lower().split()
            
            # Get scores
            scores = bm25.get_scores(tokenized_query)
            
            # Normalize scores to 0-1
            if scores.max() > 0:
                scores = scores / scores.max()
            
            return scores
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            # Return zero scores
            return np.zeros(len(documents))
    
    def _rerank_results(self, query: str, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Rerank results using cross-encoder"""
        try:
            if not results:
                return results
            
            # Prepare pairs for reranking
            pairs = [[query, doc.page_content] for doc, _ in results]
            
            # Get reranking scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Combine with original scores (weighted average)
            reranked_results = []
            for i, (doc, original_score) in enumerate(results):
                # Normalize rerank score to 0-1
                rerank_score = (rerank_scores[i] + 10) / 20  # Assuming scores are -10 to 10
                
                # Weighted combination
                final_score = 0.3 * original_score + 0.7 * rerank_score
                reranked_results.append((doc, final_score))
            
            # Sort by final score
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results

# Global instance
_hybrid_searcher = None

def get_hybrid_searcher() -> HybridSearcher:
    """Get or create hybrid searcher instance"""
    global _hybrid_searcher
    if _hybrid_searcher is None:
        _hybrid_searcher = HybridSearcher()
    return _hybrid_searcher
