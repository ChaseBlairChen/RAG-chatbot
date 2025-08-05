# legal_assistant/utils/text_processing.py - ENHANCED VERSION WITH BACKWARD COMPATIBILITY
"""
Enhanced text processing utilities with class-based architecture and backward compatibility.
Maintains all existing function interfaces while adding powerful new features.
"""
import re
import logging
import hashlib
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer, util

from ..core.dependencies import get_sentence_model, get_sentence_model_name

logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Enhanced text processing class with semantic chunking and intelligent extraction.
    
    Usage:
        processor = TextProcessor()
        chunks = processor.semantic_chunking(text)
        info = processor.extract_bill_information(context, "HB 1234")
    """
    
    def __init__(self, sentence_model: SentenceTransformer = None):
        """Initialize processor and load sentence model once for efficiency"""
        self.model = sentence_model or get_sentence_model()
        self.model_name = get_sentence_model_name()
        
        if self.model:
            logger.info(f"✅ TextProcessor initialized with model: {self.model_name}")
        else:
            logger.warning("⚠️ TextProcessor initialized without sentence model - semantic features disabled")
        
        # --- Enhanced data-driven regex patterns ---
        self._bill_extraction_patterns = [
            {
                'name': 'sponsors', 
                'pattern': r"Sponsors?\s*:\s*([^\n]+)",
                'description': 'Bill sponsors'
            },
            {
                'name': 'final_status', 
                'pattern': r"Final\s+Status\s*:\s*([^\n]+)",
                'description': 'Final bill status'
            },
            {
                'name': 'session_law', 
                'pattern': r"(?:C\s+\d+\s+L\s+\d+)",
                'description': 'Session law citation'
            },
            {
                'name': 'effective_date', 
                'pattern': r"Effective\s+Date\s*:\s*([^\n]+)",
                'description': 'Bill effective date'
            },
            {
                'name': 'description', 
                'pattern': r"{bill_number}[^\n]*\n([^\n]+(?:\n[^\n]+)*?)(?=\n\s*[A-Z]{{2,}}|\n\s*[A-Z]{{1,3}}\s+\d+|\Z)",
                'description': 'Bill description text'
            }
        ]
        
        self._universal_extraction_patterns = [
            {
                'name': 'key_entities', 
                'pattern': r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", 
                'limit': 15,
                'description': 'Named entities and proper nouns'
            },
            {
                'name': 'bill_numbers', 
                'pattern': r"(?:HB|SB|SSB|ESSB|SHB|ESHB)\s*\d+", 
                'limit': 20,
                'description': 'Legislative bill numbers'
            },
            {
                'name': 'statute_citations', 
                'pattern': r"(?:RCW|USC|CFR|WAC)\s+\d+(?:\.\d+)*", 
                'limit': 10,
                'description': 'Legal statute citations'
            },
            {
                'name': 'dollar_amounts', 
                'pattern': r"\$[\d,]+(?:\.\d{2})?", 
                'limit': 10,
                'description': 'Monetary amounts'
            },
            {
                'name': 'dates_numeric', 
                'pattern': r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", 
                'limit': 10,
                'description': 'Numeric date formats'
            },
            {
                'name': 'dates_written', 
                'pattern': r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}", 
                'limit': 10,
                'description': 'Written date formats'
            },
            {
                'name': 'percentages', 
                'pattern': r"\b\d+(?:\.\d+)?%", 
                'limit': 10,
                'description': 'Percentage values'
            },
            {
                'name': 'phone_numbers', 
                'pattern': r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", 
                'limit': 5,
                'description': 'Phone numbers'
            }
        ]
        
        # Legal document patterns for better chunking
        self._legal_section_patterns = [
            r'\n\s*(?:ARTICLE|Article)\s+([IVX\d]+)[^\n]*',
            r'\n\s*(?:SECTION|Section)\s+(\d+(?:\.\d+)?)[^\n]*',
            r'\n\s*(\d+)\.\s+[A-Z][^\n]*',  # Numbered sections
            r'\n\s*\(([a-z])\)\s*',  # Subsections (a), (b), etc.
            r'\n\s*WHEREAS',
            r'\n\s*NOW, THEREFORE',
        ]
    
    def semantic_chunking(
        self, 
        text: str, 
        max_chunk_size: int = 1500, 
        overlap: int = 200,
        similarity_percentile: int = 25
    ) -> List[str]:
        """
        Enhanced semantic chunking with legal document awareness
        """
        if not self.model:
            logger.warning("Semantic model unavailable, using basic chunking")
            return self.basic_chunking(text, max_chunk_size, overlap)
        
        if len(text) <= max_chunk_size:
            return [text]
        
        try:
            # First check if this is a legal document
            if self._is_legal_document(text):
                return self._semantic_chunking_legal(text, max_chunk_size, overlap)
            else:
                return self._semantic_chunking_general(text, max_chunk_size, overlap, similarity_percentile)
                
        except Exception as e:
            logger.error(f"Semantic chunking failed: {e}, falling back to basic chunking")
            return self.basic_chunking(text, max_chunk_size, overlap)
    
    def _semantic_chunking_legal(self, text: str, max_chunk_size: int, overlap: int) -> List[str]:
        """Semantic chunking optimized for legal documents"""
        
        # Look for legal section breaks
        sections = []
        current_pos = 0
        
        # Find legal section boundaries
        for pattern in self._legal_section_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                if match.start() > current_pos:
                    section_text = text[current_pos:match.start()].strip()
                    if section_text:
                        sections.append(section_text)
                current_pos = match.start()
        
        # Add remaining text
        if current_pos < len(text):
            remaining_text = text[current_pos:].strip()
            if remaining_text:
                sections.append(remaining_text)
        
        # If no legal patterns found, fall back to paragraph splitting
        if not sections:
            sections = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not sections:
            return [text]
        
        # Now apply semantic grouping to sections
        if len(sections) <= 1:
            return sections
        
        try:
            # Get embeddings for sections
            section_embeddings = self.model.encode(sections, convert_to_tensor=True, show_progress_bar=False)
            
            # Calculate similarities
            similarities = util.cos_sim(section_embeddings[:-1], section_embeddings[1:])
            
            # Use adaptive threshold
            threshold = np.percentile(similarities.diag(), 30)  # Slightly higher for legal docs
            
            # Group sections semantically
            chunks = []
            current_chunk_sections = []
            current_size = 0
            
            for i, section in enumerate(sections):
                section_size = len(section)
                
                # Check if we should start a new chunk
                should_split = False
                
                if current_size + section_size > max_chunk_size and current_chunk_sections:
                    should_split = True
                elif i > 0 and i-1 < len(similarities.diag()):
                    # Check semantic similarity
                    if similarities.diag()[i-1] < threshold and current_chunk_sections:
                        should_split = True
                
                if should_split:
                    # Save current chunk
                    chunk_text = '\n\n'.join(current_chunk_sections)
                    chunks.append(chunk_text)
                    
                    # Start new chunk with overlap if similar content
                    if i > 0 and similarities.diag()[i-1] > 0.7:  # High similarity
                        overlap_sections = current_chunk_sections[-1:] if current_chunk_sections else []
                        current_chunk_sections = overlap_sections + [section]
                        current_size = sum(len(s) for s in current_chunk_sections)
                    else:
                        current_chunk_sections = [section]
                        current_size = section_size
                else:
                    current_chunk_sections.append(section)
                    current_size += section_size
            
            # Add remaining sections
            if current_chunk_sections:
                chunk_text = '\n\n'.join(current_chunk_sections)
                chunks.append(chunk_text)
            
            logger.info(f"Legal semantic chunking created {len(chunks)} chunks from {len(sections)} sections")
            return chunks
            
        except Exception as e:
            logger.error(f"Legal semantic chunking failed: {e}")
            return sections  # Return sections as-is
    
    def _semantic_chunking_general(self, text: str, max_chunk_size: int, overlap: int, similarity_percentile: int) -> List[str]:
        """General semantic chunking for non-legal documents"""
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return [text]
        
        # Get embeddings for sentences
        embeddings = self.model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
        
        # Calculate cosine similarity between adjacent sentences
        similarities = util.cos_sim(embeddings[:-1], embeddings[1:])
        
        # Identify split points where similarity drops below threshold
        threshold = np.percentile(similarities.diag(), similarity_percentile)
        split_indices = [i for i, sim in enumerate(similarities.diag()) if sim < threshold]
        
        # Group sentences into chunks based on split points and size constraints
        chunks = []
        start_idx = 0
        
        for end_idx in split_indices + [len(sentences)]:
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)
            
            # If chunk is too large, split it further
            if len(chunk_text) > max_chunk_size:
                sub_chunks = self._split_large_chunk(chunk_sentences, max_chunk_size, overlap)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)
            
            start_idx = end_idx
        
        logger.info(f"General semantic chunking created {len(chunks)} chunks from {len(sentences)} sentences")
        return chunks
    
    def _split_large_chunk(self, sentences: List[str], max_chunk_size: int, overlap: int) -> List[str]:
        """Split large semantic chunk while preserving sentence boundaries"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))
                
                # Calculate overlap sentences
                overlap_sentences = []
                overlap_size = 0
                
                for sent in reversed(current_chunk):
                    if overlap_size + len(sent) <= overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_size += len(sent)
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_size = overlap_size + sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add remaining sentences
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _is_legal_document(self, text: str) -> bool:
        """Detect if document is a legal document based on patterns"""
        legal_indicators = [
            r'\b(?:WHEREAS|NOW THEREFORE|AGREEMENT|CONTRACT|PARTY|PARTIES)\b',
            r'\b(?:Section|Article|Clause)\s+\d+',
            r'\b(?:shall|hereby|herein|thereof|whereof)\b',
            r'\b(?:HB|SB|SSB|ESSB|SHB|ESHB)\s+\d+',
            r'\b(?:RCW|USC|CFR|WAC)\s+\d+',
            r'\bFinal\s+Status\s*:',
            r'\bSponsors?\s*:'
        ]
        
        legal_score = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in legal_indicators)
        return legal_score > 5  # Threshold for legal document detection
    
    def basic_chunking(self, text: str, max_chunk_size: int = 1500, overlap: int = 200) -> List[str]:
        """Enhanced basic chunking with sentence boundary preservation"""
        if len(text) <= max_chunk_size:
            return [text]
        
        # Try sentence-based chunking first
        sentences = self._split_into_sentences(text)
        
        if len(sentences) > 1:
            return self._chunk_by_sentences(sentences, max_chunk_size, overlap)
        else:
            # Fall back to character-based chunking
            return self._chunk_by_characters(text, max_chunk_size, overlap)
    
    def _chunk_by_sentences(self, sentences: List[str], max_chunk_size: int, overlap: int) -> List[str]:
        """Chunk text by sentences with overlap"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))
                
                # Calculate overlap
                overlap_sentences = []
                overlap_size = 0
                
                for sent in reversed(current_chunk):
                    if overlap_size + len(sent) <= overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_size += len(sent)
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_size = overlap_size + sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add remaining sentences
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _chunk_by_characters(self, text: str, max_chunk_size: int, overlap: int) -> List[str]:
        """Character-based chunking with smart boundaries"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to break at sentence boundary
            chunk = text[start:end]
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            last_space = chunk.rfind(' ')
            
            # Find the best breaking point
            break_points = [last_period, last_newline, last_space]
            break_point = max([bp for bp in break_points if bp > start + max_chunk_size // 2], default=-1)
            
            if break_point > 0:
                end = start + break_point + 1
            
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks
    
    def parse_questions(self, query_text: str) -> List[str]:
        """Enhanced question parsing with better delimiter handling"""
        if not query_text.strip():
            return []
        
        # Handle multiple question delimiters
        if ';' in query_text:
            questions = [q.strip() for q in query_text.split(';') if q.strip()]
        elif query_text.count('?') > 1:
            # Split on '?' but keep the '?' with each question
            parts = query_text.split('?')
            questions = []
            for i, part in enumerate(parts[:-1]):  # Exclude last empty part
                part = part.strip()
                if part:
                    questions.append(part + '?')
        else:
            # Single question
            final_question = query_text.strip()
            if not final_question.endswith('?') and '?' not in final_question:
                final_question += '?'
            questions = [final_question]
        
        # Filter out very short questions
        questions = [q for q in questions if len(q.strip()) > 3]
        
        return questions
    
    def remove_duplicates(self, results_with_scores: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
        """Enhanced duplicate removal with content hashing"""
        if not results_with_scores:
            return []
        
        unique_results = []
        seen_hashes = set()
        
        for doc, score in results_with_scores:
            # Create content hash for deduplication
            if hasattr(doc, 'page_content'):
                content = doc.page_content
            else:
                content = str(doc)
            
            # Use first 200 chars for hash to handle near-duplicates
            content_sample = content[:200].strip()
            content_hash = hashlib.sha256(content_sample.encode('utf-8')).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_results.append((doc, score))
        
        # Sort by score
        unique_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Removed {len(results_with_scores) - len(unique_results)} duplicates")
        return unique_results
    
    def extract_bill_information(self, context_text: str, bill_number: str) -> Dict[str, str]:
        """Enhanced bill information extraction with better patterns"""
        
        # Escape bill number for regex safety
        escaped_bill = re.escape(bill_number)
        
        # Create patterns with the specific bill number
        patterns = []
        for pattern_info in self._bill_extraction_patterns:
            pattern = pattern_info['pattern']
            if '{bill_number}' in pattern:
                pattern = pattern.format(bill_number=escaped_bill)
            
            patterns.append({
                'name': pattern_info['name'],
                'pattern': pattern,
                'description': pattern_info['description']
            })
        
        # Extract information
        extracted_info = self._extract_with_patterns(context_text, patterns)
        
        # Enhanced bill-specific extraction
        bill_context = self._find_bill_context(context_text, bill_number)
        if bill_context:
            # Extract additional info from bill context
            additional_info = self._extract_bill_context_info(bill_context)
            extracted_info.update(additional_info)
        
        logger.info(f"Extracted bill info for {bill_number}: {list(extracted_info.keys())}")
        return extracted_info
    
    def _find_bill_context(self, context_text: str, bill_number: str) -> Optional[str]:
        """Find the context around a specific bill"""
        
        # Enhanced patterns to find bill information with more context
        bill_patterns = [
            rf"{re.escape(bill_number)}[^\n]*(?:\n(?:[^\n]*(?:sponsors?|final\s+status|enables|authorizes|establishes)[^\n]*\n?)*)",
            rf"{re.escape(bill_number)}.*?(?=\n\s*[A-Z]{{2,}}|\n\s*[A-Z]{{1,3}}\s+\d+|\Z)",
            rf"{re.escape(bill_number)}[^\n]*\n(?:[^\n]+\n?){{0,10}}"
        ]
        
        for pattern in bill_patterns:
            bill_match = re.search(pattern, context_text, re.DOTALL | re.IGNORECASE)
            if bill_match:
                bill_context = bill_match.group(0)
                logger.debug(f"Found bill context for {bill_number}: {len(bill_context)} chars")
                return bill_context
        
        return None
    
    def _extract_bill_context_info(self, bill_context: str) -> Dict[str, str]:
        """Extract additional information from bill context"""
        additional_info = {}
        
        # Look for committee information
        committee_match = re.search(r"Committee\s*:\s*([^\n]+)", bill_context, re.IGNORECASE)
        if committee_match:
            additional_info["committee"] = committee_match.group(1).strip()
        
        # Look for vote information
        vote_match = re.search(r"(?:passed|failed).*?(\d+[-\s]*\d+)", bill_context, re.IGNORECASE)
        if vote_match:
            additional_info["vote_record"] = vote_match.group(0).strip()
        
        # Look for amendment information
        amendment_match = re.search(r"amendment[s]?\s*[#:]?\s*([^\n]+)", bill_context, re.IGNORECASE)
        if amendment_match:
            additional_info["amendments"] = amendment_match.group(1).strip()
        
        return additional_info
    
    def extract_universal_information(self, context_text: str, question: str = "") -> Dict[str, Any]:
        """Enhanced universal information extraction with question awareness"""
        
        # Basic extraction using patterns
        extracted_info = self._extract_with_patterns(
            context_text, 
            self._universal_extraction_patterns, 
            find_all=True
        )
        
        # Question-aware extraction
        if question:
            question_specific_info = self._extract_question_specific_info(context_text, question)
            extracted_info.update(question_specific_info)
        
        # Post-process and clean up
        cleaned_info = self._clean_extracted_info(extracted_info)
        
        logger.debug(f"Universal extraction found: {list(cleaned_info.keys())}")
        return cleaned_info
    
    def _extract_question_specific_info(self, context_text: str, question: str) -> Dict[str, Any]:
        """Extract information specific to the question being asked"""
        specific_info = {}
        question_lower = question.lower()
        
        # If question asks about time/duration
        if any(word in question_lower for word in ['time', 'duration', 'long', 'minutes', 'hours']):
            time_patterns = [
                r'(\d+)\s*(?:minutes?|mins?)\b',
                r'(\d+)\s*(?:hours?|hrs?)\b',
                r'(\d+)\s*(?:days?)\b',
                r'(\d+)\s*(?:weeks?)\b',
                r'(\d+)\s*(?:months?)\b'
            ]
            
            time_values = []
            for pattern in time_patterns:
                matches = re.findall(pattern, context_text, re.IGNORECASE)
                time_values.extend(matches)
            
            if time_values:
                specific_info['time_durations'] = time_values[:5]
        
        # If question asks about requirements
        if any(word in question_lower for word in ['require', 'must', 'shall', 'need']):
            requirement_patterns = [
                r'(?:must|shall|required?)\s+([^.]+)',
                r'(?:requirement|standard)\s*:\s*([^\n]+)',
                r'(?:minimum|maximum)\s+of\s+([^.]+)'
            ]
            
            requirements = []
            for pattern in requirement_patterns:
                matches = re.findall(pattern, context_text, re.IGNORECASE)
                requirements.extend(matches)
            
            if requirements:
                specific_info['requirements'] = [req.strip() for req in requirements[:5]]
        
        return specific_info
    
    def _clean_extracted_info(self, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and deduplicate extracted information"""
        cleaned = {}
        
        for key, value in extracted_info.items():
            if isinstance(value, list):
                # Remove duplicates and empty values
                unique_values = []
                seen = set()
                
                for item in value:
                    item_clean = str(item).strip()
                    if item_clean and item_clean.lower() not in seen:
                        unique_values.append(item_clean)
                        seen.add(item_clean.lower())
                
                if unique_values:
                    cleaned[key] = unique_values
            elif value and str(value).strip():
                cleaned[key] = str(value).strip()
        
        return cleaned
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Enhanced sentence splitting with legal document awareness"""
        
        # Handle legal citations properly (don't split on periods in citations)
        # Protect common legal abbreviations
        protected_abbrevs = ['U.S.', 'U.S.C.', 'C.F.R.', 'F.2d', 'F.3d', 'S.Ct.', 'Inc.', 'Corp.', 'LLC.', 'Ltd.']
        
        # Replace protected abbreviations temporarily
        temp_text = text
        replacements = {}
        
        for i, abbrev in enumerate(protected_abbrevs):
            placeholder = f"__ABBREV_{i}__"
            temp_text = temp_text.replace(abbrev, placeholder)
            replacements[placeholder] = abbrev
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', temp_text)
        
        # Restore protected abbreviations
        restored_sentences = []
        for sentence in sentences:
            for placeholder, original in replacements.items():
                sentence = sentence.replace(placeholder, original)
            
            if sentence.strip():
                restored_sentences.append(sentence.strip())
        
        return restored_sentences
    
    def _extract_with_patterns(self, text: str, patterns: List[Dict], find_all: bool = False) -> Dict[str, Any]:
        """Enhanced pattern extraction with error handling and validation"""
        extracted_info = {}
        
        for pattern_info in patterns:
            name = pattern_info['name']
            pattern = pattern_info['pattern']
            limit = pattern_info.get('limit', 1)
            
            try:
                if find_all:
                    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                    if matches:
                        # Apply limit and filter empty matches
                        valid_matches = [m for m in matches if str(m).strip()][:limit]
                        if valid_matches:
                            extracted_info[name] = valid_matches
                else:
                    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                    if match:
                        # Extract the first capture group, or the full match if no groups
                        result = (match.group(1) if match.groups() else match.group(0)).strip()
                        if result:
                            extracted_info[name] = result
                            
            except re.error as regex_error:
                logger.error(f"Regex error for pattern '{name}': {regex_error}")
            except Exception as e:
                logger.error(f"Extraction error for pattern '{name}': {e}")
        
        return extracted_info

# --- Global Instance for Backward Compatibility ---

_text_processor = None

def get_text_processor() -> TextProcessor:
    """Get or create global text processor instance"""
    global _text_processor
    if _text_processor is None:
        _text_processor = TextProcessor()
    return _text_processor

# --- BACKWARD COMPATIBLE FUNCTIONS ---
# These maintain the exact same interface your existing code expects

def parse_multiple_questions(query_text: str) -> List[str]:
    """BACKWARD COMPATIBLE: Parse multiple questions from a single query"""
    processor = get_text_processor()
    return processor.parse_questions(query_text)

def semantic_chunking_with_bert(text: str, max_chunk_size: int = 1500, overlap: int = 300) -> List[str]:
    """BACKWARD COMPATIBLE: Advanced semantic chunking with powerful BERT models"""
    processor = get_text_processor()
    return processor.semantic_chunking(text, max_chunk_size, overlap)

def basic_text_chunking(text: str, max_chunk_size: int = 1500, overlap: int = 300) -> List[str]:
    """BACKWARD COMPATIBLE: Basic text chunking fallback"""
    processor = get_text_processor()
    return processor.basic_chunking(text, max_chunk_size, overlap)

def remove_duplicate_documents(results_with_scores: List[Tuple]) -> List[Tuple]:
    """BACKWARD COMPATIBLE: Remove duplicate documents from search results"""
    processor = get_text_processor()
    return processor.remove_duplicates(results_with_scores)

def extract_bill_information(context_text: str, bill_number: str) -> Dict[str, str]:
    """BACKWARD COMPATIBLE: Pre-extract bill information using regex patterns"""
    processor = get_text_processor()
    return processor.extract_bill_information(context_text, bill_number)

def extract_universal_information(context_text: str, question: str) -> Dict[str, Any]:
    """BACKWARD COMPATIBLE: Universal information extraction that works for any document type"""
    processor = get_text_processor()
    return processor.extract_universal_information(context_text, question)

# --- ENHANCED FUNCTIONS (New functionality) ---

def extract_statute_information(context_text: str, statute_citation: str) -> Dict[str, Any]:
    """NEW: Extract specific information from statutory text"""
    processor = get_text_processor()
    
    statute_patterns = [
        {
            'name': 'requirements',
            'pattern': r'(?:shall|must|required?)\s+([^.]+)',
            'limit': 10,
            'description': 'Legal requirements'
        },
        {
            'name': 'prohibitions',
            'pattern': r'(?:shall not|prohibited|forbidden)\s+([^.]+)',
            'limit': 5,
            'description': 'Legal prohibitions'
        },
        {
            'name': 'penalties',
            'pattern': r'(?:penalty|fine|punishment)\s*:\s*([^\n]+)',
            'limit': 5,
            'description': 'Penalty provisions'
        },
        {
            'name': 'definitions',
            'pattern': rf'"{re.escape(statute_citation.split()[-1])}"?\s*means\s+([^.]+)',
            'limit': 3,
            'description': 'Legal definitions'
        }
    ]
    
    return processor._extract_with_patterns(context_text, statute_patterns, find_all=True)

def analyze_document_structure(text: str) -> Dict[str, Any]:
    """NEW: Analyze the structure of a document"""
    processor = get_text_processor()
    
    structure_info = {
        'document_type': 'legal' if processor._is_legal_document(text) else 'general',
        'estimated_pages': len(text) // 2500,
        'word_count': len(text.split()),
        'sentence_count': len(processor._split_into_sentences(text)),
        'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
        'has_legal_citations': bool(re.search(r'\b(?:USC|CFR|RCW|WAC)\s+\d+', text)),
        'has_bill_numbers': bool(re.search(r'\b(?:HB|SB|SSB|ESSB|SHB|ESHB)\s+\d+', text)),
        'content_density': len(text.strip()) / len(text) if text else 0
    }
    
    return structure_info

def intelligent_chunking_by_type(text: str, doc_type: str = "auto", 
                               max_chunk_size: int = 1500, overlap: int = 300) -> List[str]:
    """NEW: Intelligent chunking based on detected or specified document type"""
    processor = get_text_processor()
    
    # Auto-detect document type if not specified
    if doc_type == "auto":
        doc_type = "legal" if processor._is_legal_document(text) else "general"
    
    logger.info(f"Using {doc_type} chunking strategy for document")
    
    if doc_type == "legal":
        return processor._semantic_chunking_legal(text, max_chunk_size, overlap)
    else:
        return processor.semantic_chunking(text, max_chunk_size, overlap)

# --- USAGE EXAMPLES ---

"""
USAGE EXAMPLES:

# Basic usage (backward compatible)
questions = parse_multiple_questions("What is HB 1234? Who sponsored it?")
chunks = semantic_chunking_with_bert(long_text)
info = extract_bill_information(context, "HB 1234")

# Enhanced usage (new features)
processor = TextProcessor()
structure = analyze_document_structure(text)
chunks = intelligent_chunking_by_type(text, doc_type="legal")
statute_info = extract_statute_information(context, "RCW 1.23.456")

# Performance usage
processor = TextProcessor()  # Initialize once
for document in documents:
    chunks = processor.semantic_chunking(document.text)  # Reuse model
"""
