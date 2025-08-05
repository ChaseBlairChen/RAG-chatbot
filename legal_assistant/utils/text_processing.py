"""
Refactored text processing utilities encapsulated in a class for improved structure and maintainability.
"""
import re
import logging
import hashlib
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util

# Assume this dependency loader is still valid
from ..core.dependencies import get_sentence_model, get_sentence_model_name

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    A class to encapsulate text processing functionalities like chunking and information extraction.

    Usage:
        processor = TextProcessor()
        chunks = processor.semantic_chunking(my_long_text)
        info = processor.extract_bill_information(context, "HB 1234")
    """
    def __init__(self, sentence_model: SentenceTransformer = None):
        """Initializes the processor and loads the sentence model once."""
        self.model = sentence_model or get_sentence_model()
        self.model_name = get_sentence_model_name()
        if self.model:
            logger.info(f"TextProcessor initialized with model: {self.model_name}")
        else:
            logger.warning("TextProcessor initialized without a sentence model. Semantic features will be disabled.")

        # --- Data-driven regex patterns for extraction ---
        self._bill_extraction_patterns = [
            {'name': 'sponsors', 'pattern': r"Sponsors?\s*:\s*([^\n]+)"},
            {'name': 'final_status', 'pattern': r"Final\s+Status\s*:\s*([^\n]+)"},
            {'name': 'description', 'pattern': rf"{{bill_number}}[^\n]*\n([^\n]+(?:\n[^\n]+)*?)(?=\n\s*[A-Z]{{2,}}|\n\s*[A-Z]{{1,3}}\s+\d+|\Z)"}
        ]

        self._universal_extraction_patterns = [
            {'name': 'key_entities', 'pattern': r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", 'limit': 10},
            {'name': 'bill_numbers', 'pattern': r"(?:HB|SB|SSB|ESSB|SHB|ESHB)\s*\d+", 'limit': 10},
            {'name': 'dollar_amounts', 'pattern': r"\$[\d,]+(?:\.\d{2})?", 'limit': 10},
            {'name': 'dates', 'pattern': r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", 'limit': 10},
            {'name': 'full_dates', 'pattern': r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}", 'limit': 10}
        ]

    def parse_questions(self, query_text: str) -> List[str]:
        """Parses a string that may contain multiple questions."""
        # Use regex split to handle multiple delimiters gracefully.
        # This pattern splits on one or more occurrences of '?' or ';'
        questions = re.split(r'[?;]+', query_text)
        # Filter out empty strings and strip whitespace
        return [q.strip() for q in questions if q.strip()]

    def semantic_chunking(
        self,
        text: str,
        max_chunk_size: int = 1500,
        overlap: int = 200,
        similarity_percentile: int = 25
    ) -> List[str]:
        """
        Performs semantic chunking by splitting text at points of low semantic similarity.
        """
        if not self.model:
            logger.warning("No sentence model loaded. Falling back to basic chunking.")
            return self.basic_chunking(text, max_chunk_size, overlap)
        
        if len(text) <= max_chunk_size:
            return [text]

        # 1. Split text into sentences (a better unit than paragraphs for this)
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return [text] if text else []
            
        # 2. Get embeddings for each sentence
        embeddings = self.model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
        
        # 3. Calculate cosine similarity between adjacent sentences
        similarities = util.cos_sim(embeddings[:-1], embeddings[1:])
        
        # 4. Identify split points where similarity drops below a threshold
        # We use a percentile to adapt to the document's own coherence level.
        threshold = np.percentile(similarities.diag(), similarity_percentile)
        split_indices = [i for i, sim in enumerate(similarities.diag()) if sim < threshold]
        
        # 5. Group sentences into chunks based on split points
        chunks = []
        start_idx = 0
        for end_idx in split_indices:
            chunk_sentences = sentences[start_idx:end_idx + 1]
            chunks.append(" ".join(chunk_sentences))
            start_idx = end_idx + 1
            
        # Add the final group of sentences as the last chunk
        if start_idx < len(sentences):
            chunks.append(" ".join(sentences[start_idx:]))

        logger.info(f"Semantic chunking created {len(chunks)} chunks.")
        # Optionally, run a basic chunker over the semantic chunks if any are too large
        return self._ensure_max_chunk_size(chunks, max_chunk_size, overlap)

    def basic_chunking(self, text: str, max_chunk_size: int = 1500, overlap: int = 200) -> List[str]:
        """A simple fallback chunker based on fixed size and overlap."""
        if len(text) <= max_chunk_size:
            return [text]
            
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            length_function=len,
        )
        return text_splitter.split_text(text)
        
    def remove_duplicates(self, results_with_scores: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
        """Removes duplicate documents from search results using a reliable hash."""
        if not results_with_scores:
            return []
            
        unique_results = []
        seen_hashes = set()
        
        for doc, score in results_with_scores:
            # Assumes 'doc' has a 'page_content' attribute
            content = doc.page_content
            # Use a secure hash of the *entire* content for reliability
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_results.append((doc, score))
                
        # Sort by score again as the order might have changed
        unique_results.sort(key=lambda x: x[1], reverse=True)
        return unique_results

    def extract_bill_information(self, context_text: str, bill_number: str) -> Dict[str, str]:
        """Extracts specific information about a bill using a data-driven regex approach."""
        # Format the description pattern with the specific bill number
        patterns = [
            {'name': p['name'], 'pattern': p['pattern'].format(bill_number=re.escape(bill_number))}
            if '{bill_number}' in p['pattern'] else p
            for p in self._bill_extraction_patterns
        ]
        return self._extract_with_patterns(context_text, patterns)

    def extract_universal_information(self, context_text: str) -> Dict[str, Any]:
        """Extracts generic entities from any text using a data-driven regex approach."""
        return self._extract_with_patterns(context_text, self._universal_extraction_patterns, find_all=True)

    # --- Private Helper Methods ---

    def _split_into_sentences(self, text: str) -> List[str]:
        """A simple regex-based sentence splitter."""
        # Positive lookbehind for sentence-ending punctuation. Handles cases like U.S.A.
        text = re.sub(r'([.?!])\s+', r'\1|', text)
        sentences = [s.strip() for s in text.split('|') if s.strip()]
        return sentences

    def _ensure_max_chunk_size(self, chunks: List[str], max_chunk_size: int, overlap: int) -> List[str]:
        """Ensures no chunk exceeds the max size by splitting oversized ones."""
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_chunk_size:
                final_chunks.extend(self.basic_chunking(chunk, max_chunk_size, overlap))
            else:
                final_chunks.append(chunk)
        return final_chunks
        
    def _extract_with_patterns(self, text: str, patterns: List[Dict], find_all: bool = False) -> Dict[str, Any]:
        """Generic extraction utility that uses a list of pattern dictionaries."""
        extracted_info = {}
        for item in patterns:
            name = item['name']
            pattern = item['pattern']
            limit = item.get('limit', 1)
            
            try:
                if find_all:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if name not in extracted_info:
                        extracted_info[name] = []
                    extracted_info[name].extend(matches[:limit])
                else:
                    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                    if match:
                        # Extract the first capture group, or the full match if no groups
                        extracted_info[name] = (match.group(1) if match.groups() else match.group(0)).strip()
            except re.error as e:
                logger.error(f"Regex error for pattern '{name}': {e}")
        return extracted_info
