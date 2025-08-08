"""
Enhanced Hybrid Search with Advanced Semantic Understanding and Multi-Modal Search
"""
import logging
import asyncio
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import re

from ..config import MIN_RELEVANCE_SCORE, DEFAULT_SEARCH_K, ENHANCED_SEARCH_K
from ..storage.managers import get_vector_store
from ..utils.text_processing import preprocess_text, extract_key_terms

logger = logging.getLogger(__name__)

@dataclass
class SearchQuery:
    """Enhanced search query with context"""
    original_query: str
    processed_query: str
    query_type: str
    search_intent: str
    key_terms: List[str]
    legal_entities: List[str]
    jurisdiction: Optional[str] = None
    time_period: Optional[str] = None
    document_types: List[str] = None

@dataclass
class SearchResult:
    """Enhanced search result with metadata"""
    content: str
    document_id: str
    relevance_score: float
    search_method: str
    metadata: Dict[str, Any]
    context_snippets: List[str]
    legal_citations: List[str]
    confidence: float

class EnhancedHybridSearch:
    """Enhanced hybrid search with advanced semantic understanding"""
    
    def __init__(self):
        self.vector_store = get_vector_store()
        self.search_methods = {
            "semantic": self._semantic_search,
            "keyword": self._keyword_search,
            "citation": self._citation_search,
            "temporal": self._temporal_search,
            "jurisdictional": self._jurisdictional_search
        }
    
    async def search(self, query: str, search_scope: str = "all", 
                    search_methods: List[str] = None, k: int = None) -> List[SearchResult]:
        """Enhanced search with multiple methods"""
        
        # Process and analyze query
        search_query = self._process_query(query)
        
        # Determine search methods
        if search_methods is None:
            search_methods = self._determine_search_methods(search_query)
        
        # Set search parameters
        k = k or ENHANCED_SEARCH_K
        
        # Perform searches
        results = []
        for method in search_methods:
            if method in self.search_methods:
                method_results = await self.search_methods[method](search_query, k)
                results.extend(method_results)
        
        # Merge and rank results
        merged_results = self._merge_and_rank_results(results, search_query)
        
        # Filter by relevance
        filtered_results = [r for r in merged_results if r.relevance_score >= MIN_RELEVANCE_SCORE]
        
        return filtered_results[:k]
    
    def _process_query(self, query: str) -> SearchQuery:
        """Process and analyze search query"""
        processed_query = preprocess_text(query)
        key_terms = extract_key_terms(processed_query)
        legal_entities = self._extract_legal_entities(query)
        query_type = self._classify_query_type(query)
        search_intent = self._determine_search_intent(query)
        jurisdiction = self._extract_jurisdiction(query)
        time_period = self._extract_time_period(query)
        document_types = self._extract_document_types(query)
        
        return SearchQuery(
            original_query=query,
            processed_query=processed_query,
            query_type=query_type,
            search_intent=search_intent,
            key_terms=key_terms,
            legal_entities=legal_entities,
            jurisdiction=jurisdiction,
            time_period=time_period,
            document_types=document_types or []
        )
    
    def _extract_legal_entities(self, query: str) -> List[str]:
        """Extract legal entities from query"""
        entities = []
        
        # Legal citations
        citation_patterns = [
            r'\b\d+\s+U\.S\.C\.\s+\d+\b',  # USC citations
            r'\b\d+\s+C\.F\.R\.\s+\d+\b',  # CFR citations
            r'\b\d+\s+F\.\d+\s+\d+\b',     # Federal cases
            r'\b\d+\s+S\.\s+Ct\.\s+\d+\b', # Supreme Court
            r'\b\d+\s+F\.\s*Supp\.\s+\d+\b' # District Court
        ]
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend(matches)
        
        # Legal terms
        legal_terms = [
            r'\b(contract|agreement|statute|regulation|case law)\b',
            r'\b(breach|damages|liability|negligence|fraud)\b',
            r'\b(jurisdiction|venue|standing|mootness)\b',
            r'\b(appeal|motion|petition|hearing|trial)\b'
        ]
        
        for pattern in legal_terms:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend(matches)
        
        return list(set(entities))
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['statute', 'law', 'regulation', 'code']):
            return 'statutory'
        elif any(term in query_lower for term in ['case', 'court', 'decision', 'ruling']):
            return 'case_law'
        elif any(term in query_lower for term in ['procedure', 'process', 'how to']):
            return 'procedural'
        elif any(term in query_lower for term in ['contract', 'agreement', 'terms']):
            return 'contractual'
        elif any(term in query_lower for term in ['compliance', 'regulatory', 'requirements']):
            return 'compliance'
        else:
            return 'general'
    
    def _determine_search_intent(self, query: str) -> str:
        """Determine search intent"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['what is', 'define', 'meaning']):
            return 'definition'
        elif any(term in query_lower for term in ['how to', 'procedure', 'steps']):
            return 'procedural'
        elif any(term in query_lower for term in ['compare', 'difference', 'similar']):
            return 'comparative'
        elif any(term in query_lower for term in ['analyze', 'review', 'assess']):
            return 'analytical'
        elif any(term in query_lower for term in ['find', 'search', 'locate']):
            return 'retrieval'
        else:
            return 'information'
    
    def _extract_jurisdiction(self, query: str) -> Optional[str]:
        """Extract jurisdiction from query"""
        jurisdiction_patterns = [
            r'\b(federal|federal court|us court)\b',
            r'\b(state|state court|state law)\b',
            r'\b(california|ca|texas|tx|new york|ny)\b',
            r'\b(supreme court|appellate court|district court)\b'
        ]
        
        for pattern in jurisdiction_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group().lower()
        
        return None
    
    def _extract_time_period(self, query: str) -> Optional[str]:
        """Extract time period from query"""
        time_patterns = [
            r'\b(\d{4})\b',  # Year
            r'\b(recent|current|latest|new)\b',
            r'\b(historical|old|previous|former)\b',
            r'\b(last \d+ years?)\b',
            r'\b(since \d{4})\b'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group().lower()
        
        return None
    
    def _extract_document_types(self, query: str) -> List[str]:
        """Extract document types from query"""
        doc_types = []
        
        type_patterns = {
            'contract': r'\b(contract|agreement|terms|conditions)\b',
            'statute': r'\b(statute|law|act|code|regulation)\b',
            'case': r'\b(case|decision|ruling|opinion|judgment)\b',
            'form': r'\b(form|application|petition|filing)\b',
            'brief': r'\b(brief|motion|pleading|memorandum)\b'
        }
        
        for doc_type, pattern in type_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                doc_types.append(doc_type)
        
        return doc_types
    
    def _determine_search_methods(self, search_query: SearchQuery) -> List[str]:
        """Determine appropriate search methods based on query"""
        methods = ['semantic']  # Always include semantic search
        
        # Add keyword search for specific terms
        if search_query.legal_entities or search_query.key_terms:
            methods.append('keyword')
        
        # Add citation search for legal citations
        if any('citation' in entity.lower() for entity in search_query.legal_entities):
            methods.append('citation')
        
        # Add temporal search for time-based queries
        if search_query.time_period:
            methods.append('temporal')
        
        # Add jurisdictional search for jurisdiction-specific queries
        if search_query.jurisdiction:
            methods.append('jurisdictional')
        
        return methods
    
    async def _semantic_search(self, search_query: SearchQuery, k: int) -> List[SearchResult]:
        """Enhanced semantic search"""
        try:
            # Use vector store for semantic search
            results = self.vector_store.similarity_search_with_score(
                search_query.processed_query, k=k
            )
            
            search_results = []
            for content, score in results:
                result = SearchResult(
                    content=content.page_content,
                    document_id=content.metadata.get('document_id', ''),
                    relevance_score=float(score),
                    search_method='semantic',
                    metadata=content.metadata,
                    context_snippets=self._extract_context_snippets(content.page_content, search_query),
                    legal_citations=self._extract_citations(content.page_content),
                    confidence=min(1.0, 1.0 - float(score))
                )
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def _keyword_search(self, search_query: SearchQuery, k: int) -> List[SearchResult]:
        """Enhanced keyword search"""
        try:
            # Combine key terms and legal entities
            search_terms = search_query.key_terms + search_query.legal_entities
            
            # Perform keyword search
            results = self.vector_store.similarity_search_with_score(
                ' '.join(search_terms), k=k
            )
            
            search_results = []
            for content, score in results:
                # Boost score for keyword matches
                keyword_matches = sum(1 for term in search_terms 
                                    if term.lower() in content.page_content.lower())
                boosted_score = float(score) * (1.0 - 0.1 * keyword_matches)
                
                result = SearchResult(
                    content=content.page_content,
                    document_id=content.metadata.get('document_id', ''),
                    relevance_score=boosted_score,
                    search_method='keyword',
                    metadata=content.metadata,
                    context_snippets=self._extract_context_snippets(content.page_content, search_query),
                    legal_citations=self._extract_citations(content.page_content),
                    confidence=min(1.0, 1.0 - boosted_score)
                )
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    async def _citation_search(self, search_query: SearchQuery, k: int) -> List[SearchResult]:
        """Search for specific legal citations"""
        try:
            # Extract citations from query
            citations = [entity for entity in search_query.legal_entities 
                        if any(pattern in entity.lower() for pattern in ['usc', 'cfr', 'f.', 's. ct.'])]
            
            if not citations:
                return []
            
            # Search for each citation
            all_results = []
            for citation in citations:
                results = self.vector_store.similarity_search_with_score(citation, k=k//2)
                all_results.extend(results)
            
            # Deduplicate and rank
            seen_ids = set()
            unique_results = []
            for content, score in all_results:
                doc_id = content.metadata.get('document_id', '')
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_results.append((content, score))
            
            search_results = []
            for content, score in unique_results:
                result = SearchResult(
                    content=content.page_content,
                    document_id=content.metadata.get('document_id', ''),
                    relevance_score=float(score),
                    search_method='citation',
                    metadata=content.metadata,
                    context_snippets=self._extract_context_snippets(content.page_content, search_query),
                    legal_citations=self._extract_citations(content.page_content),
                    confidence=min(1.0, 1.0 - float(score))
                )
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Citation search failed: {e}")
            return []
    
    async def _temporal_search(self, search_query: SearchQuery, k: int) -> List[SearchResult]:
        """Search based on time period"""
        # This would implement time-based filtering
        # For now, return empty list
        return []
    
    async def _jurisdictional_search(self, search_query: SearchQuery, k: int) -> List[SearchResult]:
        """Search based on jurisdiction"""
        # This would implement jurisdiction-based filtering
        # For now, return empty list
        return []
    
    def _merge_and_rank_results(self, results: List[SearchResult], search_query: SearchQuery) -> List[SearchResult]:
        """Merge and rank search results"""
        # Group by document ID
        doc_results = {}
        for result in results:
            doc_id = result.document_id
            if doc_id not in doc_results:
                doc_results[doc_id] = []
            doc_results[doc_id].append(result)
        
        # Merge results for same document
        merged_results = []
        for doc_id, doc_result_list in doc_results.items():
            if len(doc_result_list) == 1:
                merged_results.append(doc_result_list[0])
            else:
                # Merge multiple results for same document
                merged_result = self._merge_document_results(doc_result_list, search_query)
                merged_results.append(merged_result)
        
        # Sort by relevance score
        merged_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return merged_results
    
    def _merge_document_results(self, results: List[SearchResult], search_query: SearchQuery) -> SearchResult:
        """Merge multiple results for the same document"""
        # Use the best result as base
        best_result = max(results, key=lambda x: x.relevance_score)
        
        # Combine context snippets
        all_snippets = []
        for result in results:
            all_snippets.extend(result.context_snippets)
        
        # Combine legal citations
        all_citations = []
        for result in results:
            all_citations.extend(result.legal_citations)
        
        # Calculate average confidence
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        # Boost relevance score for multiple method matches
        method_boost = 1.0 + 0.1 * len(set(r.search_method for r in results))
        boosted_score = min(1.0, best_result.relevance_score * method_boost)
        
        return SearchResult(
            content=best_result.content,
            document_id=best_result.document_id,
            relevance_score=boosted_score,
            search_method='hybrid',
            metadata=best_result.metadata,
            context_snippets=list(set(all_snippets))[:5],  # Limit to 5 unique snippets
            legal_citations=list(set(all_citations)),  # Remove duplicates
            confidence=avg_confidence
        )
    
    def _extract_context_snippets(self, content: str, search_query: SearchQuery) -> List[str]:
        """Extract relevant context snippets"""
        snippets = []
        
        # Split content into sentences
        sentences = re.split(r'[.!?]+', content)
        
        # Find sentences containing key terms
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(term.lower() in sentence_lower for term in search_query.key_terms):
                snippets.append(sentence.strip())
        
        return snippets[:3]  # Limit to 3 snippets
    
    def _extract_citations(self, content: str) -> List[str]:
        """Extract legal citations from content"""
        citations = []
        
        citation_patterns = [
            r'\b\d+\s+U\.S\.C\.\s+\d+\b',
            r'\b\d+\s+C\.F\.R\.\s+\d+\b',
            r'\b\d+\s+F\.\d+\s+\d+\b',
            r'\b\d+\s+S\.\s+Ct\.\s+\d+\b',
            r'\b\d+\s+F\.\s*Supp\.\s+\d+\b'
        ]
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, content)
            citations.extend(matches)
        
        return list(set(citations))  # Remove duplicates

# Global instance
enhanced_hybrid_search = EnhancedHybridSearch()
