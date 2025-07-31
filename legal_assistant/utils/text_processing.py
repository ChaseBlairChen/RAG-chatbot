"""Text processing utilities"""
import re
import logging
import numpy as np
from typing import List, Tuple, Dict, Any
from ..core.dependencies import get_sentence_model, get_sentence_model_name

logger = logging.getLogger(__name__)

def parse_multiple_questions(query_text: str) -> List[str]:
    """Parse multiple questions from a single query"""
    questions = []
    
    if ';' in query_text:
        parts = query_text.split(';')
        for part in parts:
            part = part.strip()
            if part:
                questions.append(part)
    elif '?' in query_text and query_text.count('?') > 1:
        parts = query_text.split('?')
        for part in parts:
            part = part.strip()
            if part:
                questions.append(part + '?')
    else:
        final_question = query_text
        if not final_question.endswith('?') and '?' not in final_question:
            final_question += '?'
        questions = [final_question]
    
    return questions

def semantic_chunking_with_bert(text: str, max_chunk_size: int = 1500, overlap: int = 300) -> List[str]:
    """Advanced semantic chunking with powerful BERT models for legal documents"""
    try:
        sentence_model = get_sentence_model()
        sentence_model_name = get_sentence_model_name()
        
        if sentence_model is None:
            logger.warning("No sentence model available, using basic chunking")
            return basic_text_chunking(text, max_chunk_size, overlap)
        
        logger.info(f"Using semantic chunking with model: {sentence_model_name}")
        
        # For legal documents, split on legal sections and paragraphs
        # Look for common legal document patterns
        legal_patterns = [
            r'\n\s*SECTION\s+\d+',
            r'\n\s*\d+\.\s+',  # Numbered sections
            r'\n\s*\([a-z]\)',  # Subsections (a), (b), etc.
            r'\n\s*WHEREAS',
            r'\n\s*NOW, THEREFORE',
            r'\n\s*Article\s+[IVX\d]+',
        ]
        
        # Split text into meaningful sections first
        sections = []
        current_pos = 0
        
        # Find legal section breaks
        for pattern in legal_patterns:
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
            sections = [text]
        
        # If document is small enough, return as single chunk
        if len(text) <= max_chunk_size:
            return [text]
        
        # Calculate embeddings for sections (batch processing for efficiency)
        try:
            section_embeddings = sentence_model.encode(sections, batch_size=32, show_progress_bar=False)
        except Exception as e:
            logger.warning(f"Embedding calculation failed: {e}, using basic chunking")
            return basic_text_chunking(text, max_chunk_size, overlap)
        
        # Advanced semantic grouping using cosine similarity
        chunks = []
        current_chunk = []
        current_chunk_size = 0
        
        for i, section in enumerate(sections):
            section_size = len(section)
            
            # If adding this section would exceed chunk size
            if current_chunk_size + section_size > max_chunk_size and current_chunk:
                
                # For legal documents, try to find natural breaking points
                if len(current_chunk) > 1:
                    # Calculate semantic similarity to decide on best split point
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(chunk_text)
                    
                    # Intelligent overlap: keep semantically similar content
                    if i > 0:
                        # Use similarity to determine overlap
                        prev_embedding = section_embeddings[i-1:i]
                        curr_embedding = section_embeddings[i:i+1]
                        
                        try:
                            similarity = np.dot(prev_embedding[0], curr_embedding[0])
                            if similarity > 0.7:  # High similarity - include more overlap
                                overlap_sections = current_chunk[-2:] if len(current_chunk) > 1 else current_chunk[-1:]
                            else:
                                overlap_sections = current_chunk[-1:] if current_chunk else []
                            
                            current_chunk = overlap_sections + [section]
                            current_chunk_size = sum(len(s) for s in current_chunk)
                        except:
                            # Fallback to simple overlap
                            current_chunk = [current_chunk[-1], section] if current_chunk else [section]
                            current_chunk_size = sum(len(s) for s in current_chunk)
                    else:
                        current_chunk = [section]
                        current_chunk_size = section_size
                else:
                    # Single large section - need to split it
                    if section_size > max_chunk_size:
                        # Split large section into smaller parts
                        large_section_chunks = basic_text_chunking(section, max_chunk_size, overlap)
                        chunks.extend(large_section_chunks[:-1])  # Add all but last
                        current_chunk = [large_section_chunks[-1]]  # Keep last for next iteration
                        current_chunk_size = len(large_section_chunks[-1])
                    else:
                        chunks.append(section)
                        current_chunk = []
                        current_chunk_size = 0
            else:
                current_chunk.append(section)
                current_chunk_size += section_size
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(chunk_text)
        
        # Ensure we have at least one chunk
        if not chunks:
            chunks = [text[:max_chunk_size]]
        
        logger.info(f"Semantic chunking created {len(chunks)} chunks from {len(sections)} sections")
        return chunks
        
    except Exception as e:
        logger.error(f"Advanced semantic chunking failed: {e}, falling back to basic chunking")
        return basic_text_chunking(text, max_chunk_size, overlap)

def basic_text_chunking(text: str, max_chunk_size: int = 1500, overlap: int = 300) -> List[str]:
    """Basic text chunking fallback"""
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chunk_size
        
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to break at a sentence boundary
        chunk = text[start:end]
        last_period = chunk.rfind('.')
        last_newline = chunk.rfind('\n')
        
        # Find the best breaking point
        break_point = max(last_period, last_newline)
        if break_point > start + max_chunk_size // 2:  # Only if break point is reasonable
            end = start + break_point + 1
        
        chunks.append(text[start:end])
        start = end - overlap  # Add overlap
    
    return chunks

def remove_duplicate_documents(results_with_scores: List[Tuple]) -> List[Tuple]:
    """Remove duplicate documents from search results"""
    if not results_with_scores:
        return []
    
    unique_results = []
    seen_content = set()
    
    for doc, score in results_with_scores:
        content_hash = hash(doc.page_content[:100])
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_results.append((doc, score))
    
    unique_results.sort(key=lambda x: x[1], reverse=True)
    return unique_results

def extract_bill_information(context_text: str, bill_number: str) -> Dict[str, str]:
    """Pre-extract bill information using regex patterns"""
    extracted_info = {}
    
    # Enhanced pattern to find bill information with more context
    bill_patterns = [
        rf"{bill_number}[^\n]*(?:\n(?:[^\n]*(?:sponsors?|final\s+status|enables|authorizes|establishes)[^\n]*\n?)*)",
        rf"{bill_number}.*?(?=\n\s*[A-Z]{{2,}}|\n\s*[A-Z]{{1,3}}\s+\d+|\Z)",
        rf"{bill_number}[^\n]*\n(?:[^\n]+\n?){{0,5}}"
    ]
    
    for pattern in bill_patterns:
        bill_match = re.search(pattern, context_text, re.DOTALL | re.IGNORECASE)
        if bill_match:
            bill_text = bill_match.group(0)
            logger.info(f"Found bill text for {bill_number}: {bill_text[:200]}...")
            
            # Extract sponsors with multiple patterns
            sponsor_patterns = [
                rf"Sponsors?\s*:\s*([^\n]+)",
                rf"Sponsor\s*:\s*([^\n]+)",
                rf"(?:Rep\.|Sen\.)\s+([^,\n]+(?:,\s*[^,\n]+)*)"
            ]
            
            for sponsor_pattern in sponsor_patterns:
                sponsor_match = re.search(sponsor_pattern, bill_text, re.IGNORECASE)
                if sponsor_match:
                    extracted_info["sponsors"] = sponsor_match.group(1).strip()
                    break
            
            # Extract final status with multiple patterns
            status_patterns = [
                rf"Final Status\s*:\s*([^\n]+)",
                rf"Status\s*:\s*([^\n]+)",
                rf"(?:C\s+\d+\s+L\s+\d+)"
            ]
            
            for status_pattern in status_patterns:
                status_match = re.search(status_pattern, bill_text, re.IGNORECASE)
                if status_match:
                    extracted_info["final_status"] = status_match.group(1).strip()
                    break
            
            # Extract description - everything after bill number until next bill or section
            desc_patterns = [
                rf"{bill_number}[^\n]*\n([^\n]+(?:\n[^\n]+)*?)(?=\n\s*[A-Z]{{2,}}|\n\s*[A-Z]{{1,3}}\s+\d+|\Z)",
                rf"{bill_number}[^\n]*\n([^\n]+)"
            ]
            
            for desc_pattern in desc_patterns:
                desc_match = re.search(desc_pattern, bill_text, re.IGNORECASE)
                if desc_match:
                    description = desc_match.group(1).strip()
                    # Clean up description
                    description = re.sub(r'\s+', ' ', description)
                    extracted_info["description"] = description
                    break
            
            logger.info(f"Extracted info for {bill_number}: {extracted_info}")
            return extracted_info
    
    logger.warning(f"No bill information found for {bill_number}")
    return extracted_info

def extract_universal_information(context_text: str, question: str) -> Dict[str, Any]:
    """Universal information extraction that works for any document type"""
    extracted_info = {
        "key_entities": [],
        "numbers_and_dates": [],
        "relationships": []
    }
    
    try:
        # Extract names (people, organizations, bills, cases, etc.)
        name_patterns = [
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",  # Names
            r"(?:HB|SB|SSB|ESSB|SHB|ESHB)\s*\d+",  # Bill numbers
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, context_text)
            extracted_info["key_entities"].extend(matches[:10])  # Limit to prevent overflow
        
        # Extract numbers, dates, amounts
        number_patterns = [
            r"\$[\d,]+(?:\.\d{2})?",  # Dollar amounts
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",  # Dates
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",  # Written dates
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, context_text, re.IGNORECASE)
            extracted_info["numbers_and_dates"].extend(matches[:10])
        
        # Extract relationships
        relationship_patterns = [
            r"(?:sponsors?|authored?\s+by):\s*([^.\n]+)",
            r"(?:final\s+status|status):\s*([^.\n]+)",
        ]
        
        for pattern in relationship_patterns:
            matches = re.findall(pattern, context_text, re.IGNORECASE)
            extracted_info["relationships"].extend(matches[:5])
    
    except Exception as e:
        logger.warning(f"Error in universal extraction: {e}")
    
    return extracted_info
