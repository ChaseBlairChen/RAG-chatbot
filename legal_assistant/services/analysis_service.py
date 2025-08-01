"""Comprehensive analysis service"""
import time
import logging
from typing import Dict, List, Tuple, Optional

from ..models import ComprehensiveAnalysisRequest, StructuredAnalysisResponse, AnalysisType
from ..config import COMPREHENSIVE_SEARCH_K, OPENROUTER_API_KEY, OPENAI_API_BASE
from .container_manager import get_container_manager
from .ai_service import call_openrouter_api
from ..utils.formatting import format_context_for_llm

logger = logging.getLogger(__name__)

class ComprehensiveAnalysisProcessor:
    def __init__(self):
        self.analysis_prompts = {
            "document_summary": "Analyze this document and provide a comprehensive summary including document type, purpose, main parties, key terms, important dates, and financial obligations.",
            "key_clauses": "Extract and analyze key legal clauses including termination, indemnification, liability, governing law, confidentiality, payment terms, and dispute resolution. For each clause, provide specific text references and implications.",
            "risk_assessment": "Identify and assess legal risks including unilateral rights, broad indemnification, unlimited liability, vague obligations, and unfavorable terms. Rate each risk (High/Medium/Low) and suggest mitigation strategies.",
            "timeline_deadlines": "Extract all time-related information including start/end dates, payment deadlines, notice periods, renewal terms, performance deadlines, and warranty periods. Present chronologically.",
            "party_obligations": "List all obligations for each party including what must be done, deadlines, conditions, performance standards, and consequences of non-compliance. Organize by party.",
            "missing_clauses": "Identify commonly expected clauses that may be missing such as force majeure, limitation of liability, dispute resolution, severability, assignment restrictions, and notice provisions. Explain the importance and risks of each missing clause."
        }
    
    def process_comprehensive_analysis(self, request: ComprehensiveAnalysisRequest) -> StructuredAnalysisResponse:
        start_time = time.time()
        
        try:
            search_results, sources_searched, retrieval_method = self._enhanced_document_specific_search(
                request.user_id, 
                request.document_id, 
                "comprehensive legal document analysis",
                k=COMPREHENSIVE_SEARCH_K
            )
            
            if not search_results:
                return StructuredAnalysisResponse(
                    warnings=["No relevant documents found for analysis"],
                    processing_time=time.time() - start_time,
                    retrieval_method="no_documents_found"
                )
            
            context_text, source_info = format_context_for_llm(search_results, max_length=8000)
            
            response = StructuredAnalysisResponse()
            response.sources_by_section = {}
            response.confidence_scores = {}
            response.retrieval_method = retrieval_method
            
            if AnalysisType.COMPREHENSIVE in request.analysis_types:
                comprehensive_prompt = self._create_comprehensive_prompt(context_text)
                
                try:
                    analysis_result = call_openrouter_api(comprehensive_prompt, OPENROUTER_API_KEY, OPENAI_API_BASE)
                    parsed_sections = self._parse_comprehensive_response(analysis_result)
                    
                    response.document_summary = parsed_sections.get("summary", "")
                    response.key_clauses = parsed_sections.get("clauses", "")
                    response.risk_assessment = parsed_sections.get("risks", "")
                    response.timeline_deadlines = parsed_sections.get("timeline", "")
                    response.party_obligations = parsed_sections.get("obligations", "")
                    response.missing_clauses = parsed_sections.get("missing", "")
                    
                    response.overall_confidence = self._calculate_comprehensive_confidence(parsed_sections, len(search_results))
                    
                    for section in ["summary", "clauses", "risks", "timeline", "obligations", "missing"]:
                        response.sources_by_section[section] = source_info
                        response.confidence_scores[section] = response.overall_confidence
                    
                except Exception as e:
                    logger.error(f"Comprehensive analysis failed: {e}")
                    response.warnings.append(f"Comprehensive analysis failed: {str(e)}")
                    response.overall_confidence = 0.1
            
            else:
                for analysis_type in request.analysis_types:
                    section_result = self._process_individual_analysis(analysis_type, context_text, source_info)
                    
                    if analysis_type == AnalysisType.SUMMARY:
                        response.document_summary = section_result["content"]
                    elif analysis_type == AnalysisType.CLAUSES:
                        response.key_clauses = section_result["content"]
                    elif analysis_type == AnalysisType.RISKS:
                        response.risk_assessment = section_result["content"]
                    elif analysis_type == AnalysisType.TIMELINE:
                        response.timeline_deadlines = section_result["content"]
                    elif analysis_type == AnalysisType.OBLIGATIONS:
                        response.party_obligations = section_result["content"]
                    elif analysis_type == AnalysisType.MISSING_CLAUSES:
                        response.missing_clauses = section_result["content"]
                    
                    response.confidence_scores[analysis_type.value] = section_result["confidence"]
                    response.sources_by_section[analysis_type.value] = source_info
                
                confidences = list(response.confidence_scores.values())
                response.overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            response.processing_time = time.time() - start_time
            logger.info(f"Comprehensive analysis completed in {response.processing_time:.2f}s with confidence {response.overall_confidence:.2f}")
            
            return response
            
        except Exception as e:
            logger.error(f"Comprehensive analysis processing failed: {e}")
            return StructuredAnalysisResponse(
                warnings=[f"Analysis processing failed: {str(e)}"],
                processing_time=time.time() - start_time,
                overall_confidence=0.0,
                retrieval_method="error"
            )
    
    def _enhanced_document_specific_search(self, user_id: str, document_id: Optional[str], query: str, k: int = 15) -> Tuple[List, List[str], str]:
        all_results = []
        sources_searched = []
        retrieval_method = "enhanced_document_specific"
        
        try:
            container_manager = get_container_manager()
            if document_id:
                user_results = container_manager.enhanced_search_user_container(
                    user_id, query, "", k=k, document_id=document_id
                )
                sources_searched.append(f"document_{document_id}")
                logger.info(f"Document-specific search for {document_id}: {len(user_results)} results")
            else:
                user_results = container_manager.enhanced_search_user_container(
                    user_id, query, "", k=k
                )
                sources_searched.append("all_user_documents")
                logger.info(f"All documents search: {len(user_results)} results")
            
            for doc, score in user_results:
                doc.metadata['source_type'] = 'user_container'
                doc.metadata['search_scope'] = 'document_specific' if document_id else 'all_user_docs'
                all_results.append((doc, score))
            
            return all_results[:k], sources_searched, retrieval_method
            
        except Exception as e:
            logger.error(f"Error in document-specific search: {e}")
            return [], [], "error"
    
    def _create_comprehensive_prompt(self, context_text: str) -> str:
    """Create a comprehensive analysis prompt with strict anti-hallucination measures"""
    return f"""You are a legal document analyst providing structured analysis of legal documents.

CRITICAL INSTRUCTIONS - PREVENT HALLUCINATION:
1. **ONLY analyze what is EXPLICITLY written in the provided document**
2. **NEVER add information from general legal knowledge or assumptions**
3. **If information is not in the document, state: "Not specified in the document"**
4. **Copy the exact words from the document when making claims - use "..." for direct quotes**
5. **Each statement must be traceable to specific text in the document**

ANALYSIS FRAMEWORK:

## DOCUMENT SUMMARY
- Document type (contract, agreement, memo, etc.)
- Primary purpose and scope
- Parties involved (full legal names as stated)
- Effective date and term duration
- Governing law and jurisdiction
**Required: Copy the exact words from the document that identify each element**

## KEY CLAUSES ANALYSIS
Identify and copy the exact text of the following clauses if present:
- Termination provisions (notice period, conditions, penalties)
- Indemnification (who indemnifies whom, scope, exceptions)
- Limitation of liability (caps, exclusions, carve-outs)
- Confidentiality (duration, scope, exceptions)
- Payment terms (amounts, schedules, late fees)
- Intellectual property (ownership, licenses, restrictions)
- Dispute resolution (arbitration, mediation, court selection)
**For each clause: copy the exact words in quotes and provide section reference**

## RISK ASSESSMENT
Analyze only risks explicitly present in the document:
- **HIGH RISK**: Unlimited liability, one-sided indemnification, no termination rights
- **MEDIUM RISK**: Broad confidentiality, strict deadlines, significant penalties
- **LOW RISK**: Standard commercial terms, mutual obligations, clear exit rights
**Required: Copy the exact words from the document that create each risk**

## TIMELINE & DEADLINES
Extract all time-related provisions in chronological order:
- Contract commencement date
- Milestone deadlines
- Notice periods (termination, breach, cure)
- Renewal/expiration dates
- Performance deadlines
**Format: [Date/Deadline] - [Obligation] - "Copy exact words here" (Section X.X)**

## PARTY OBLIGATIONS
List all obligations by party as stated in the document:

**[Party A Name]**:
- Obligation 1: "Copy exact words from document" (Section reference)
- Obligation 2: "Copy exact words from document" (Section reference)

**[Party B Name]**:
- Obligation 1: "Copy exact words from document" (Section reference)
- Obligation 2: "Copy exact words from document" (Section reference)

## MISSING CLAUSES ANALYSIS
Identify standard legal provisions that are NOT present:
- Force majeure
- Severability
- Entire agreement
- Amendment procedures
- Assignment restrictions
- Warranty disclaimers
- Insurance requirements
**Only list if you've confirmed they're actually missing after checking the entire document**

## CRITICAL AMBIGUITIES
Identify any vague or undefined terms that could cause disputes:
- Undefined technical terms
- Ambiguous deadlines ("promptly", "reasonable time")
- Unclear scope definitions
- Missing calculation methods
**Copy the ambiguous language exactly as it appears in the document**

HALLUCINATION PREVENTION CHECKLIST:
Before making any claim, verify:
✓ Is this explicitly written in the document?
✓ Can I copy the exact words supporting this?
✓ Am I adding any external legal knowledge?
✓ Have I checked the entire document for this information?

DOCUMENT TEXT TO ANALYZE:
{context_text}

IMPORTANT REMINDERS:
- If you cannot find specific information, state: "Not specified in the provided document"
- Do not suggest what "should" be in the document based on legal standards
- Do not interpret or infer beyond what is explicitly written
- Every factual claim must include the exact words from the document
- If the document seems incomplete, note what sections are present vs missing
- When quoting: Copy the text exactly as it appears, word-for-word, inside quotation marks


HANDLING CONFLICTS:
- If documents contain conflicting information, present both views with citations
- Note the conflict explicitly: "Document A states X, while Document B states Y"

WHEN INFORMATION IS MISSING:
"Based on the provided documents, I cannot provide a complete answer. To provide thorough analysis, I would need documents containing: [specific missing elements]"

BEGIN ANALYSIS:"""

RESPONSE STYLE: {instruction}

CONVERSATION HISTORY:
{conversation_context}

DOCUMENT CONTEXT (ANALYZE THOROUGHLY):
{context_text}

USER QUESTION:
{questions}

RESPONSE APPROACH:
- **FIRST**: Identify what specific information the user is asking for. Do not reference any statute, case law, or principle unless it appears verbatim in the context.
- **SECOND**: Search the context thoroughly for that information  
- **THIRD**: Present any information found clearly and completely. At the end of your response, list all facts provided and their source documents for verification.
- **FOURTH**: Note what information is not available (if any)
- **ALWAYS**: Cite the source document for each fact provided

ADDITIONAL GUIDANCE:
- After fully answering based solely on the provided documents, if relevant key legal principles under Washington state law, any other U.S. state law, or U.S. federal law are not found in the sources, you may add a clearly labeled general legal principles disclaimer.
- This disclaimer must clearly state it is NOT based on the provided documents but represents general background knowledge of applicable Washington state, other state, and federal law.
- Do NOT use this disclaimer to answer the user’s question directly; it serves only as supplementary context.
- This disclaimer must explicitly state that these principles are not found in the provided documents but are usually relevant legal background.
- Format this disclaimer distinctly at the end of the response under a heading such as "GENERAL LEGAL PRINCIPLES DISCLAIMER."

RESPONSE:"""
    
    def _parse_comprehensive_response(self, response_text: str) -> Dict[str, str]:
        sections = {}
        section_markers = {
            "summary": ["## DOCUMENT SUMMARY", "# DOCUMENT SUMMARY"],
            "clauses": ["## KEY CLAUSES ANALYSIS", "# KEY CLAUSES ANALYSIS", "## KEY CLAUSES"],
            "risks": ["## RISK ASSESSMENT", "# RISK ASSESSMENT", "## RISKS"],
            "timeline": ["## TIMELINE & DEADLINES", "# TIMELINE & DEADLINES", "## TIMELINE"],
            "obligations": ["## PARTY OBLIGATIONS", "# PARTY OBLIGATIONS", "## OBLIGATIONS"],
            "missing": ["## MISSING CLAUSES ANALYSIS", "# MISSING CLAUSES ANALYSIS", "## MISSING CLAUSES"]
        }
        
        lines = response_text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line_strip = line.strip()
            
            section_found = None
            for section_key, markers in section_markers.items():
                if any(line_strip.startswith(marker) for marker in markers):
                    section_found = section_key
                    break
            
            if section_found:
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                current_section = section_found
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
        
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        for section_key in section_markers.keys():
            if section_key not in sections or not sections[section_key]:
                sections[section_key] = f"No {section_key.replace('_', ' ').title()} information found in the analysis."
        
        return sections
    
    def _process_individual_analysis(self, analysis_type: AnalysisType, context_text: str, source_info: List[Dict]) -> Dict:
        try:
            prompt = self.analysis_prompts.get(analysis_type.value, "Analyze this legal document.")
            full_prompt = f"{prompt}\n\nLEGAL DOCUMENT CONTEXT:\n{context_text}\n\nPlease provide a detailed analysis based ONLY on the provided context."
            
            result = call_openrouter_api(full_prompt, OPENROUTER_API_KEY, OPENAI_API_BASE)
            
            return {
                "content": result,
                "confidence": 0.7,
                "sources": source_info
            }
        except Exception as e:
            logger.error(f"Individual analysis failed for {analysis_type}: {e}")
            return {
                "content": f"Analysis failed for {analysis_type.value}: {str(e)}",
                "confidence": 0.1,
                "sources": []
            }
    
    def _calculate_comprehensive_confidence(self, parsed_sections: Dict[str, str], num_sources: int) -> float:
        try:
            successful_sections = sum(1 for content in parsed_sections.values() 
                                    if content and not content.startswith("No ") and len(content) > 50)
            section_factor = successful_sections / len(parsed_sections)
            
            avg_length = sum(len(content) for content in parsed_sections.values()) / len(parsed_sections)
            length_factor = min(1.0, avg_length / 200)
            
            source_factor = min(1.0, num_sources / 5)
            
            confidence = (section_factor * 0.5 + length_factor * 0.3 + source_factor * 0.2)
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
