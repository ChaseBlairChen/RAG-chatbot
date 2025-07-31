"""Query processing logic"""
import re
import logging
import traceback
from typing import Optional

from ..models import QueryResponse, ComprehensiveAnalysisRequest, AnalysisType
from ..config import FeatureFlags, OPENROUTER_API_KEY, MIN_RELEVANCE_SCORE
from ..services import (
    ComprehensiveAnalysisProcessor,
    combined_search,
    calculate_confidence_score,
    call_openrouter_api
)
from ..storage.managers import add_to_conversation, get_conversation_context
from ..utils import (
    parse_multiple_questions,
    extract_bill_information,
    extract_universal_information,
    format_context_for_llm
)

logger = logging.getLogger(__name__)

def process_query(question: str, session_id: str, user_id: Optional[str], search_scope: str, 
                 response_style: str = "balanced", use_enhanced_rag: bool = True, 
                 document_id: str = None) -> QueryResponse:
    """Main query processing function"""
    try:
        logger.info(f"Processing query - Question: '{question}', User: {user_id}, Scope: {search_scope}, Enhanced: {use_enhanced_rag}, Document: {document_id}")
        
        if any(phrase in question.lower() for phrase in ["comprehensive analysis", "complete analysis", "full analysis"]):
            logger.info("Detected comprehensive analysis request")
            
            try:
                comp_request = ComprehensiveAnalysisRequest(
                    document_id=document_id,
                    analysis_types=[AnalysisType.COMPREHENSIVE],
                    user_id=user_id or "default_user",
                    session_id=session_id,
                    response_style=response_style
                )
                
                processor = ComprehensiveAnalysisProcessor()
                comp_result = processor.process_comprehensive_analysis(comp_request)
                
                formatted_response = f"""# Comprehensive Legal Document Analysis

## Document Summary
{comp_result.document_summary or 'No summary available'}

## Key Clauses Analysis
{comp_result.key_clauses or 'No clauses analysis available'}

## Risk Assessment
{comp_result.risk_assessment or 'No risk assessment available'}

## Timeline & Deadlines
{comp_result.timeline_deadlines or 'No timeline information available'}

## Party Obligations
{comp_result.party_obligations or 'No obligations analysis available'}

## Missing Clauses Analysis
{comp_result.missing_clauses or 'No missing clauses analysis available'}

---
**Analysis Confidence:** {comp_result.overall_confidence:.1%}
**Processing Time:** {comp_result.processing_time:.2f} seconds

**Sources:** {len(comp_result.sources_by_section.get('summary', []))} document sections analyzed
"""
                
                add_to_conversation(session_id, "user", question)
                add_to_conversation(session_id, "assistant", formatted_response)
                
                return QueryResponse(
                    response=formatted_response,
                    error=None,
                    context_found=True,
                    sources=comp_result.sources_by_section.get('summary', []),
                    session_id=session_id,
                    confidence_score=comp_result.overall_confidence,
                    expand_available=False,
                    sources_searched=["comprehensive_analysis"],
                    retrieval_method=comp_result.retrieval_method
                )
                
            except Exception as e:
                logger.error(f"Comprehensive analysis failed: {e}")
        
        questions = parse_multiple_questions(question) if use_enhanced_rag else [question]
        combined_query = " ".join(questions)
        
        conversation_context = get_conversation_context(session_id)
        
        retrieved_results, sources_searched, retrieval_method = combined_search(
            combined_query, 
            user_id, 
            search_scope, 
            conversation_context,
            use_enhanced=use_enhanced_rag,
            document_id=document_id
        )
        
        if not retrieved_results:
            return QueryResponse(
                response="I couldn't find any relevant information to answer your question in the searched sources.",
                error=None,
                context_found=False,
                sources=[],
                session_id=session_id,
                confidence_score=0.1,
                sources_searched=sources_searched,
                retrieval_method=retrieval_method
            )
        
        # Format context for LLM
        context_text, source_info = format_context_for_llm(retrieved_results)
        
        # Enhanced information extraction
        bill_match = re.search(r"(HB|SB|SSB|ESSB|SHB|ESHB)\s*(\d+)", question, re.IGNORECASE)
        extracted_info = {}

        if bill_match:
            # Bill-specific extraction
            bill_number = f"{bill_match.group(1)} {bill_match.group(2)}"
            logger.info(f"Searching for bill: {bill_number}")
            
            # Search specifically for chunks containing this bill
            bill_specific_results = []
            for doc, score in retrieved_results:
                if 'contains_bills' in doc.metadata and bill_number in doc.metadata['contains_bills']:
                    bill_specific_results.append((doc, score))
                    logger.info(f"Found {bill_number} in chunk {doc.metadata.get('chunk_index', 'unknown')} with score {score}")
            
            # If we found bill-specific chunks, prioritize them
            if bill_specific_results:
                logger.info(f"Using {len(bill_specific_results)} bill-specific chunks for {bill_number}")
                # Use the bill-specific chunks with boosted relevance
                boosted_results = [(doc, min(score + 0.3, 1.0)) for doc, score in bill_specific_results]
                retrieved_results = boosted_results + [r for r in retrieved_results if r not in bill_specific_results]
                retrieved_results = retrieved_results[:len(retrieved_results)]
            
            extracted_info = extract_bill_information(context_text, bill_number)
        else:
            # Universal extraction for any document type
            extracted_info = extract_universal_information(context_text, question)

        # Add extracted information to context to make it more visible to AI
        if extracted_info:
            enhancement = "\n\nKEY INFORMATION FOUND:\n"
            for key, value in extracted_info.items():
                if value:  # Only add if there's actual content
                    if isinstance(value, list):
                        enhancement += f"- {key.replace('_', ' ').title()}: {', '.join(value[:5])}\n"
                    else:
                        enhancement += f"- {key.replace('_', ' ').title()}: {value}\n"
            
            if enhancement.strip() != "KEY INFORMATION FOUND:":
                context_text += enhancement
        
        style_instructions = {
            "concise": "Please provide a concise answer (1-2 sentences) based on the context.",
            "balanced": "Please provide a balanced answer (2-3 paragraphs) based on the context.",
            "detailed": "Please provide a detailed answer with explanations based on the context."
        }
        
        instruction = style_instructions.get(response_style, style_instructions["balanced"])
        
        prompt = f"""You are a legal research assistant. Provide thorough, accurate responses based on the provided documents.

STRICT SOURCE REQUIREMENTS:
- Answer ONLY based on the retrieved documents provided in the context
- Do NOT use general legal knowledge, training data, assumptions, or inferences beyond what's explicitly stated
- If information is not in the provided documents, state: "This information is not available in the provided documents"

SOURCES SEARCHED: {', '.join(sources_searched)}
RETRIEVAL METHOD: {retrieval_method}
{f"DOCUMENT FILTER: Specific document {document_id}" if document_id else "DOCUMENT SCOPE: All available documents"}

HALLUCINATION CHECK - Before responding, verify:
1. Is each claim supported by the retrieved documents?
2. Am I adding information not present in the sources?
3. If uncertain, default to "information not available"

INSTRUCTIONS FOR THOROUGH ANALYSIS:
1. **READ CAREFULLY**: Scan the entire context for information that answers the user's question
2. **EXTRACT DIRECTLY**: When information is clearly stated, provide it exactly as written
3. **BE SPECIFIC**: Include names, numbers, dates, and details when present
4. **QUOTE WHEN HELPFUL**: Use direct quotes for key facts or important language
5. **CITE SOURCES**: Reference the document name for each piece of information
6. **BE COMPLETE**: Provide all relevant information found before saying anything is missing
7. **BE HONEST**: Only say information is unavailable when truly absent from the context

LEGAL ANALYSIS MODES:
1. **BASIC LEGAL RESEARCH** - For factual questions about legislation/statutes/regulations
   - Extract statutory/regulatory information, sponsors, dates, provisions
   
2. **COMPREHENSIVE LEGAL ANALYSIS** - For thorough analysis requiring multiple sources
   - Analyze legal implications, compliance obligations, practical impacts
   - Note ambiguities requiring clarification
   
3. **CASE LAW ANALYSIS** - When precedent needed but unavailable, state:
   "This analysis would benefit from relevant case law not available in the current documents."

HANDLING CONFLICTS:
- If documents contain conflicting information, present both views with citations
- Note the conflict explicitly: "Document A states X, while Document B states Y"

WHEN INFORMATION IS MISSING:
"Based on the provided documents, I cannot provide a complete answer. To provide thorough analysis, I would need documents containing: [specific missing elements]"

RESPONSE STYLE: {instruction}

CONVERSATION HISTORY:
{conversation_context}

DOCUMENT CONTEXT (ANALYZE THOROUGHLY):
{context_text}

USER QUESTION:
{questions}

RESPONSE APPROACH:
- **FIRST**: Identify what specific information the user is asking for
- **SECOND**: Search the context thoroughly for that information  
- **THIRD**: Present any information found clearly and completely
- **FOURTH**: Note what information is not available (if any)
- **ALWAYS**: Cite the source document for each fact provided

RESPONSE:"""
        
        if FeatureFlags.AI_ENABLED and OPENROUTER_API_KEY:
            response_text = call_openrouter_api(prompt, OPENROUTER_API_KEY)
        else:
            response_text = f"Based on the retrieved documents:\n\n{context_text}\n\nPlease review this information to answer your question."
        
        relevant_sources = [s for s in source_info if s['relevance'] >= MIN_RELEVANCE_SCORE]
        
        if relevant_sources:
            response_text += "\n\n**SOURCES:**"
            for source in relevant_sources:
                source_type = source['source_type'].replace('_', ' ').title()
                page_info = f", Page {source['page']}" if source['page'] is not None else ""
                response_text += f"\n- [{source_type}] {source['file_name']}{page_info} (Relevance: {source['relevance']:.2f})"
        
        confidence_score = calculate_confidence_score(retrieved_results, len(response_text))
        
        add_to_conversation(session_id, "user", question)
        add_to_conversation(session_id, "assistant", response_text, source_info)
        
        return QueryResponse(
            response=response_text,
            error=None,
            context_found=True,
            sources=source_info,
            session_id=session_id,
            confidence_score=float(confidence_score),
            sources_searched=sources_searched,
            expand_available=len(questions) > 1 if use_enhanced_rag else False,
            retrieval_method=retrieval_method
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        traceback.print_exc()
        return QueryResponse(
            response=None,
            error=str(e),
            context_found=False,
            sources=[],
            session_id=session_id,
            confidence_score=0.0,
            sources_searched=[],
            retrieval_method="error"
        )
