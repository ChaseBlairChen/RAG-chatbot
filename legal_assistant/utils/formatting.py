"""Formatting utilities"""
import os
from typing import List, Tuple
from ..config import MIN_RELEVANCE_SCORE

def format_context_for_llm(results_with_scores: List[Tuple], max_length: int = 3000) -> Tuple[str, List]:
    """Format search results for LLM context"""
    context_parts = []
    source_info = []
    
    total_length = 0
    for i, (doc, score) in enumerate(results_with_scores):
        if total_length >= max_length:
            break
            
        content = doc.page_content.strip()
        metadata = doc.metadata
        
        source_path = metadata.get('source', 'unknown_source')
        page = metadata.get('page', None)
        source_type = metadata.get('source_type', 'unknown')
        
        display_source = os.path.basename(source_path)
        page_info = f" (Page {page})" if page is not None else ""
        source_prefix = f"[{source_type.upper()}]" if source_type != 'unknown' else ""
        
        if len(content) > 800:
            content = content[:800] + "... [truncated]"
            
        context_part = f"{source_prefix} [{display_source}{page_info}] (Relevance: {score:.2f}): {content}"
        context_parts.append(context_part)
        
        source_info.append({
            'id': i+1,
            'file_name': display_source,
            'page': page,
            'relevance': score,
            'full_path': source_path,
            'source_type': source_type
        })
        
        total_length += len(context_part)
    
    context_text = "\n\n".join(context_parts)
    return context_text, source_info
