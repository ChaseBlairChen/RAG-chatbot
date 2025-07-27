# Enhanced AI-Powered Document Analysis System
# This replaces the basic DocumentAnalysisEngine with intelligent AI analysis

from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import json
import requests
import re
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Set, Any
from dataclasses import dataclass
from enum import Enum
import io
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Document processing imports (keep existing)
try:
    import PyPDF2
    import docx
    from pdfplumber import PDF
except ImportError:
    print("Document processing libraries not installed. Run: pip install PyPDF2 python-docx pdfplumber")

# Enhanced Document Analysis Engine with AI-powered component identification
class EnhancedDocumentAnalysisEngine:
    def __init__(self):
        # Intelligent analysis types with AI-powered detection
        self.analysis_types = {
            'smart_summary': {
                'name': 'Smart Document Summary',
                'description': 'AI identifies document type and provides tailored summary',
                'ai_enhanced': True
            },
            'component_extraction': {
                'name': 'Intelligent Component Extraction',
                'description': 'AI identifies and extracts all document components automatically',
                'ai_enhanced': True
            },
            'risk_assessment': {
                'name': 'AI Risk Assessment',
                'description': 'AI analyzes potential risks and red flags in the document',
                'ai_enhanced': True
            },
            'clause_analysis': {
                'name': 'Smart Clause Analysis',
                'description': 'AI identifies, categorizes, and analyzes all clauses',
                'ai_enhanced': True
            },
            'compliance_check': {
                'name': 'Compliance Analysis',
                'description': 'AI checks document against common legal standards',
                'ai_enhanced': True
            },
            'missing_elements': {
                'name': 'Missing Elements Detection',
                'description': 'AI identifies what should be included but is missing',
                'ai_enhanced': True
            },
            'party_obligations': {
                'name': 'Party Obligations Mapping',
                'description': 'AI maps out obligations and responsibilities for each party',
                'ai_enhanced': True
            },
            'timeline_intelligence': {
                'name': 'Intelligent Timeline Analysis',
                'description': 'AI creates comprehensive timeline with critical dates',
                'ai_enhanced': True
            },
            'language_complexity': {
                'name': 'Language & Complexity Analysis',
                'description': 'AI analyzes readability and suggests simplifications',
                'ai_enhanced': True
            },
            'comparative_analysis': {
                'name': 'Document Comparison (if multiple uploaded)',
                'description': 'AI compares multiple versions or similar documents',
                'ai_enhanced': True
            }
        }
        
        # AI-powered document type detection patterns
        self.document_patterns = {
            'contract': {
                'keywords': ['agreement', 'contract', 'parties', 'consideration', 'terms', 'conditions'],
                'structure_indicators': ['whereas', 'now therefore', 'in witness whereof'],
                'ai_prompt_modifier': 'This appears to be a contract or agreement.'
            },
            'lease': {
                'keywords': ['lease', 'tenant', 'landlord', 'rent', 'premises', 'term'],
                'structure_indicators': ['monthly rent', 'security deposit', 'lease term'],
                'ai_prompt_modifier': 'This appears to be a lease agreement.'
            },
            'employment': {
                'keywords': ['employment', 'employee', 'employer', 'salary', 'benefits', 'termination'],
                'structure_indicators': ['job title', 'start date', 'compensation'],
                'ai_prompt_modifier': 'This appears to be an employment-related document.'
            },
            'policy': {
                'keywords': ['policy', 'procedure', 'guidelines', 'rules', 'regulations'],
                'structure_indicators': ['section', 'subsection', 'policy statement'],
                'ai_prompt_modifier': 'This appears to be a policy or procedure document.'
            },
            'legal_memo': {
                'keywords': ['memorandum', 'memo', 'legal opinion', 'analysis', 'recommendation'],
                'structure_indicators': ['issue', 'brief answer', 'conclusion'],
                'ai_prompt_modifier': 'This appears to be a legal memorandum or analysis.'
            },
            'compliance': {
                'keywords': ['compliance', 'regulatory', 'standards', 'requirements', 'audit'],
                'structure_indicators': ['requirement', 'standard', 'compliance with'],
                'ai_prompt_modifier': 'This appears to be a compliance-related document.'
            }
        }

    def detect_document_type_ai(self, document_text: str) -> Dict[str, Any]:
        """Use AI to intelligently detect document type and characteristics"""
        text_lower = document_text.lower()
        
        # Score each document type
        type_scores = {}
        for doc_type, patterns in self.document_patterns.items():
            score = 0
            
            # Keyword matching with context
            for keyword in patterns['keywords']:
                # Count occurrences with context weighting
                keyword_count = text_lower.count(keyword)
                score += keyword_count * 2
                
                # Bonus for keywords in title/header area (first 200 chars)
                if keyword in text_lower[:200]:
                    score += 5
            
            # Structure indicator matching
            for indicator in patterns['structure_indicators']:
                if indicator in text_lower:
                    score += 10
            
            type_scores[doc_type] = score
        
        # Determine most likely document type
        if type_scores:
            primary_type = max(type_scores, key=type_scores.get)
            confidence = min(type_scores[primary_type] / 20, 1.0)  # Normalize to 0-1
        else:
            primary_type = 'general'
            confidence = 0.5
        
        # Extract additional metadata using AI patterns
        metadata = self.extract_ai_metadata(document_text, primary_type)
        
        return {
            'primary_type': primary_type,
            'confidence': confidence,
            'type_scores': type_scores,
            'ai_prompt_modifier': self.document_patterns.get(primary_type, {}).get('ai_prompt_modifier', ''),
            'metadata': metadata
        }

    def extract_ai_metadata(self, text: str, doc_type: str) -> Dict[str, Any]:
        """Extract intelligent metadata based on document type"""
        metadata = {
            'word_count': len(text.split()),
            'character_count': len(text),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
            'estimated_reading_time': len(text.split()) / 200,  # avg reading speed
        }
        
        # Date extraction with context
        date_patterns = [
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
        ]
        
        dates_found = []
        for pattern in date_patterns:
            dates_found.extend(re.findall(pattern, text, re.IGNORECASE))
        
        metadata['dates_found'] = list(set(dates_found))
        metadata['date_count'] = len(dates_found)
        
        # Entity extraction (simplified)
        # Look for capitalized phrases that might be names/entities
        entity_pattern = r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b'
        potential_entities = list(set(re.findall(entity_pattern, text)))
        
        # Filter out common words
        common_words = {'The', 'This', 'That', 'And', 'Or', 'But', 'For', 'With', 'By', 'In', 'On', 'At', 'To', 'From'}
        entities = [e for e in potential_entities if e not in common_words and len(e) > 2]
        
        metadata['potential_entities'] = entities[:20]  # Limit to top 20
        
        # Complexity analysis
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        metadata['complexity_metrics'] = {
            'avg_sentence_length': avg_sentence_length,
            'complexity_level': 'high' if avg_sentence_length > 25 else 'medium' if avg_sentence_length > 15 else 'low',
            'sentence_count': len(sentences)
        }
        
        # Document-type specific metadata
        if doc_type == 'contract':
            metadata['contract_indicators'] = {
                'has_parties': bool(re.search(r'(party|parties|between|among)', text, re.IGNORECASE)),
                'has_consideration': bool(re.search(r'(consideration|payment|compensation)', text, re.IGNORECASE)),
                'has_signatures': bool(re.search(r'(signature|signed|execute)', text, re.IGNORECASE)),
                'has_termination': bool(re.search(r'(terminate|termination|end|expire)', text, re.IGNORECASE))
            }
        elif doc_type == 'lease':
            metadata['lease_indicators'] = {
                'has_rent_amount': bool(re.search(r'\$[\d,]+', text)),
                'has_lease_term': bool(re.search(r'(month|year|term)', text, re.IGNORECASE)),
                'has_deposit': bool(re.search(r'deposit', text, re.IGNORECASE))
            }
        
        return metadata

    def generate_ai_analysis_prompt(self, document_text: str, analysis_type: str) -> str:
        """Generate intelligent, context-aware prompts for AI analysis"""
        
        # Detect document type and characteristics
        doc_analysis = self.detect_document_type_ai(document_text)
        doc_type = doc_analysis['primary_type']
        confidence = doc_analysis['confidence']
        ai_modifier = doc_analysis['ai_prompt_modifier']
        
        # Base prompts for each analysis type
        base_prompts = {
            'smart_summary': f"""
You are an expert legal document analyst. {ai_modifier} (Confidence: {confidence:.1%})

Provide a comprehensive, intelligent summary that includes:

1. **Document Type & Purpose**: Clearly identify what type of document this is and its primary purpose
2. **Key Parties**: Identify all parties involved and their roles
3. **Main Provisions**: Summarize the most important terms, conditions, or provisions
4. **Critical Information**: Highlight dates, amounts, obligations, and deadlines
5. **Structure Overview**: Describe how the document is organized
6. **Notable Features**: Point out any unusual or particularly important clauses

Make this summary accessible to both legal professionals and laypeople.
""",

            'component_extraction': f"""
You are an expert legal document analyst. {ai_modifier} (Confidence: {confidence:.1%})

Systematically identify and extract ALL components of this document:

1. **Document Header/Title Information**
2. **Party Information** (names, addresses, roles)
3. **Recitals/Background** (whereas clauses, context)
4. **Main Body Sections** (numbered/lettered sections with titles)
5. **Definitions** (defined terms and their meanings)
6. **Rights and Obligations** (what each party must/can do)
7. **Financial Terms** (payments, amounts, fees)
8. **Dates and Deadlines** (effective dates, termination dates, milestones)
9. **Conditions and Contingencies** (if/then clauses)
10. **Boilerplate Clauses** (governing law, dispute resolution, etc.)
11. **Signature Blocks**
12. **Attachments/Exhibits** (referenced documents)

For each component, provide the actual text and explain its legal significance.
""",

            'risk_assessment': f"""
You are a senior legal risk analyst. {ai_modifier} (Confidence: {confidence:.1%})

Conduct a thorough risk assessment focusing on:

1. **HIGH RISK ITEMS**:
   - Unlimited liability exposure
   - Broad indemnification clauses
   - Automatic renewal terms
   - Unilateral termination rights
   - Vague or ambiguous language

2. **MEDIUM RISK ITEMS**:
   - Missing standard protections
   - Imbalanced obligations
   - Unclear dispute resolution
   - Inadequate insurance requirements

3. **OPERATIONAL RISKS**:
   - Compliance requirements
   - Performance standards
   - Notification obligations
   - Change management procedures

4. **FINANCIAL RISKS**:
   - Payment terms and penalties
   - Cost allocation
   - Currency/inflation risks

For each risk, provide:
- Specific clause or section reference
- Risk level (High/Medium/Low)
- Potential impact
- Recommended mitigation strategies
""",

            'clause_analysis': f"""
You are an expert contract attorney. {ai_modifier} (Confidence: {confidence:.1%})

Provide detailed clause-by-clause analysis:

1. **ESSENTIAL CLAUSES** (analyze each):
   - Parties and definitions
   - Scope of work/services
   - Payment terms
   - Term and termination
   - Intellectual property
   - Confidentiality
   - Limitation of liability
   - Indemnification
   - Governing law and dispute resolution

2. **CLAUSE QUALITY ASSESSMENT**:
   - Well-drafted vs. problematic clauses
   - Missing standard protections
   - Industry-specific requirements
   - Enforceability concerns

3. **CLAUSE INTERACTIONS**:
   - How clauses work together
   - Potential conflicts between clauses
   - Dependencies and cross-references

4. **RECOMMENDATIONS**:
   - Clauses that need revision
   - Missing clauses to add
   - Language improvements
""",

            'compliance_check': f"""
You are a compliance specialist and legal expert. {ai_modifier} (Confidence: {confidence:.1%})

Analyze this document for compliance with standard legal requirements:

1. **FORMATION REQUIREMENTS**:
   - Proper party identification
   - Consideration/mutual benefit
   - Clear terms and conditions
   - Proper execution/signatures

2. **INDUSTRY STANDARDS**:
   - Common industry practices
   - Regulatory compliance requirements
   - Professional standards adherence

3. **LEGAL SUFFICIENCY**:
   - Required disclosures
   - Statutory compliance
   - Consumer protection (if applicable)
   - Employment law compliance (if applicable)

4. **BEST PRACTICES**:
   - Standard protective clauses
   - Risk management provisions
   - Clear dispute resolution
   - Proper documentation

5. **GAPS AND RECOMMENDATIONS**:
   - Missing required elements
   - Compliance improvements needed
   - Risk mitigation suggestions
""",

            'missing_elements': f"""
You are a comprehensive legal document reviewer. {ai_modifier} (Confidence: {confidence:.1%})

Identify what's missing from this document by analyzing:

1. **STANDARD DOCUMENT ELEMENTS** typically found in this type of document:
   - Required legal provisions
   - Industry-standard clauses
   - Protective measures
   - Administrative provisions

2. **PARTY PROTECTIONS**:
   - What should protect each party's interests
   - Risk allocation mechanisms
   - Exit strategies and remedies

3. **OPERATIONAL CLARITY**:
   - Performance standards
   - Communication requirements
   - Change management procedures
   - Reporting obligations

4. **LEGAL SAFEGUARDS**:
   - Limitation of liability
   - Force majeure
   - Compliance requirements
   - Dispute resolution mechanisms

5. **PRACTICAL CONSIDERATIONS**:
   - Implementation details
   - Monitoring and oversight
   - Documentation requirements

For each missing element, explain why it's important and provide suggested language.
""",

            'party_obligations': f"""
You are an expert in contract obligation analysis. {ai_modifier} (Confidence: {confidence:.1%})

Create a comprehensive mapping of all party obligations:

1. **PARTY IDENTIFICATION**:
   - Name each party clearly
   - Define their role in the agreement
   - Note any subsidiaries/affiliates involved

2. **OBLIGATION MATRIX** for each party:
   - **Performance Obligations** (what they must do)
   - **Payment Obligations** (financial responsibilities)
   - **Compliance Obligations** (laws, regulations, standards)
   - **Reporting Obligations** (information sharing requirements)
   - **Maintenance Obligations** (ongoing responsibilities)

3. **TIMELINE ANALYSIS**:
   - When each obligation begins
   - Duration of obligations
   - Deadlines and milestones
   - Renewal/termination impacts

4. **MUTUAL OBLIGATIONS**:
   - Shared responsibilities
   - Cooperation requirements
   - Information sharing

5. **CONSEQUENCES**:
   - What happens if obligations aren't met
   - Remedies available
   - Termination triggers

Present this as a clear, actionable roadmap for each party.
""",

            'timeline_intelligence': f"""
You are a legal project manager and timeline specialist. {ai_modifier} (Confidence: {confidence:.1%})

Create an intelligent, comprehensive timeline analysis:

1. **CRITICAL DATES EXTRACTION**:
   - Effective date
   - Commencement dates
   - Deadlines and milestones
   - Renewal dates
   - Termination dates
   - Notice periods

2. **TIMELINE VISUALIZATION**:
   - Chronological order of all events
   - Dependencies between dates
   - Critical path analysis
   - Buffer periods and flexibility

3. **RISK ANALYSIS**:
   - Time-sensitive obligations
   - Penalty dates
   - Automatic triggers
   - Notice requirements

4. **CALENDAR INTEGRATION**:
   - Recommended calendar entries
   - Alert dates (advance warnings)
   - Recurring obligations
   - Annual requirements

5. **CONTINGENCY PLANNING**:
   - What if dates are missed
   - Extension procedures
   - Force majeure impacts
   - Modification processes

Present as both a chronological timeline and a practical management tool.
""",

            'language_complexity': f"""
You are a legal writing expert and communication specialist. {ai_modifier} (Confidence: {confidence:.1%})

Analyze the language and complexity of this document:

1. **READABILITY ANALYSIS**:
   - Overall complexity level
   - Average sentence length
   - Technical terminology density
   - Passive vs. active voice usage

2. **CLARITY ASSESSMENT**:
   - Clear vs. ambiguous language
   - Defined vs. undefined terms
   - Consistent terminology usage
   - Logical organization and flow

3. **ACCESSIBILITY**:
   - Comprehension level required
   - Industry jargon usage
   - Plain English opportunities
   - Client communication suitability

4. **IMPROVEMENT RECOMMENDATIONS**:
   - Simplification opportunities
   - Clarification needs
   - Reorganization suggestions
   - Definition improvements

5. **RISK FACTORS**:
   - Ambiguous language that could cause disputes
   - Overly complex provisions
   - Inconsistent terminology
   - Missing explanations

Provide specific examples and suggested rewrites for complex passages.
""",

            'comparative_analysis': f"""
You are a comparative legal document analyst. {ai_modifier} (Confidence: {confidence:.1%})

[Note: This analysis works best with multiple documents uploaded]

If multiple documents provided, compare them across:

1. **STRUCTURAL DIFFERENCES**:
   - Organization and format
   - Section numbering and headings
   - Length and detail level

2. **SUBSTANTIVE VARIATIONS**:
   - Different terms and conditions
   - Varying obligations and rights
   - Changed financial terms
   - Modified timelines

3. **RISK PROFILE CHANGES**:
   - Added or removed protections
   - Shifted liability allocation
   - Changed termination rights
   - Modified dispute resolution

4. **EVOLUTION ANALYSIS**:
   - Improvements or deteriorations
   - Industry standard alignment
   - Legal compliance updates

If only one document provided, compare against typical industry standards and best practices for this document type.

Highlight all significant differences and their implications.
"""
        }
        
        # Get the base prompt for the analysis type
        base_prompt = base_prompts.get(analysis_type, f"""
Analyze this legal document focusing on {analysis_type.replace('_', ' ').title()}.
{ai_modifier} (Confidence: {confidence:.1%})

Provide a thorough, professional analysis that would be valuable to legal professionals and their clients.
""")
        
        # Add document-specific context
        metadata = doc_analysis['metadata']
        context_info = f"""

**DOCUMENT CONTEXT**:
- Estimated Type: {doc_type.title()} (Confidence: {confidence:.1%})
- Length: {metadata['word_count']} words, {metadata['character_count']} characters
- Complexity: {metadata['complexity_metrics']['complexity_level'].title()}
- Reading Time: ~{metadata['estimated_reading_time']:.1f} minutes
"""
        
        if metadata.get('dates_found'):
            context_info += f"- Key Dates Found: {len(metadata['dates_found'])} dates identified\n"
        
        # Final prompt assembly
        complete_prompt = f"""{base_prompt}{context_info}

**DOCUMENT TO ANALYZE**:
{document_text}

**INSTRUCTIONS**:
- Be thorough and specific in your analysis
- Reference specific sections or clauses
- Provide actionable insights and recommendations
- Use professional legal terminology but explain complex concepts
- Structure your response with clear headings and bullet points
- Include practical implications for the parties involved

**ANALYSIS**:"""
        
        return complete_prompt

    def analyze_document_with_ai(self, document_text: str, analysis_type: str, custom_prompt: str = None) -> Dict:
        """Enhanced document analysis with AI-powered intelligence"""
        
        # If custom prompt provided, use it; otherwise generate intelligent prompt
        if custom_prompt:
            analysis_prompt = custom_prompt
            doc_analysis = {'primary_type': 'custom', 'confidence': 0.8}
        else:
            analysis_prompt = self.generate_ai_analysis_prompt(document_text, analysis_type)
            doc_analysis = self.detect_document_type_ai(document_text)
        
        # Extract comprehensive metadata
        metadata = self.extract_comprehensive_metadata(document_text, doc_analysis)
        
        return {
            'prompt': analysis_prompt,
            'analysis_type': analysis_type,
            'document_analysis': doc_analysis,
            'metadata': metadata,
            'ai_enhanced': True,
            'prompt_length': len(analysis_prompt),
            'confidence_score': doc_analysis.get('confidence', 0.5)
        }

    def extract_comprehensive_metadata(self, text: str, doc_analysis: Dict) -> Dict:
        """Extract comprehensive metadata with AI insights"""
        base_metadata = doc_analysis.get('metadata', {})
        
        # Add advanced analysis
        advanced_metadata = {
            'document_structure': self.analyze_document_structure(text),
            'content_density': self.analyze_content_density(text),
            'legal_language_indicators': self.identify_legal_language(text),
            'action_items': self.extract_action_items(text),
            'financial_references': self.extract_financial_info(text)
        }
        
        # Merge all metadata
        comprehensive_metadata = {**base_metadata, **advanced_metadata}
        
        return comprehensive_metadata

    def analyze_document_structure(self, text: str) -> Dict:
        """Analyze the structural elements of the document"""
        # Count sections, subsections, paragraphs, etc.
        sections = len(re.findall(r'^\d+\.|\n\d+\.', text, re.MULTILINE))
        subsections = len(re.findall(r'^\d+\.\d+|\n\d+\.\d+', text, re.MULTILINE))
        bullet_points = len(re.findall(r'^\s*[-•*]|\n\s*[-•*]', text, re.MULTILINE))
        
        return {
            'sections': sections,
            'subsections': subsections,
            'bullet_points': bullet_points,
            'has_structured_format': sections > 0 or bullet_points > 5,
            'organization_level': 'high' if sections > 5 else 'medium' if sections > 0 else 'low'
        }

    def analyze_content_density(self, text: str) -> Dict:
        """Analyze content density and information richness"""
        words = text.split()
        unique_words = set(word.lower().strip('.,!?;:') for word in words)
        
        return {
            'total_words': len(words),
            'unique_words': len(unique_words),
            'vocabulary_richness': len(unique_words) / max(len(words), 1),
            'information_density': 'high' if len(unique_words) / max(len(words), 1) > 0.7 else 'medium'
        }

    def identify_legal_language(self, text: str) -> Dict:
        """Identify legal language patterns and formality"""
        legal_terms = [
            'whereas', 'heretofore', 'hereinafter', 'pursuant', 'notwithstanding',
            'thereof', 'hereof', 'whereby', 'aforesaid', 'aforementioned',
            'covenant', 'warrant', 'represent', 'indemnify', 'liable'
        ]
        
        text_lower = text.lower()
        legal_term_count = sum(1 for term in legal_terms if term in text_lower)
        
        return {
            'legal_terms_found': legal_term_count,
            'formality_level': 'high' if legal_term_count > 10 else 'medium' if legal_term_count > 3 else 'low',
            'legal_language_density': legal_term_count / max(len(text.split()), 1) * 1000  # per 1000 words
        }

    def extract_action_items(self, text: str) -> List[str]:
        """Extract potential action items and obligations"""
        action_patterns = [
            r'shall\s+([^.]{1,100})',
            r'must\s+([^.]{1,100})',
            r'required\s+to\s+([^.]{1,100})',
            r'obligated\s+to\s+([^.]{1,100})',
            r'will\s+([^.]{1,100})',
            r'agree\s+to\s+([^.]{1,100})'
        ]
        
        action_items = []
        for pattern in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            action_items.extend([match.strip() for match in matches if len(match.strip()) > 10])
        
        return action_items[:20]  # Limit to top 20

    def extract_financial_info(self, text: str) -> Dict:
        """Extract financial information and amounts"""
        # Money patterns
        money_patterns = [
            r'\$\s*[\d,]+(?:\.\d{2})?',
            r'\b\d+\s*dollars?\b',
            r'\b\d+\s*USD\b'
        ]
        
        financial_refs = []
        for pattern in money_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            financial_refs.extend(matches)
        
        # Percentage patterns
        percentage_pattern = r'\b\d+(?:\.\d+)?%'
        percentages = re.findall(percentage_pattern, text)
        
        return {
            'financial_amounts': financial_refs[:10],  # Limit to first 10
            'percentages': percentages[:10],
            'has_financial_terms': len(financial_refs) > 0 or len(percentages) > 0
        }

    def get_analysis_types(self) -> Dict:
        """Return available analysis types with descriptions"""
        return {
            analysis_id: {
                'name': details['name'],
                'description': details['description'],
                'ai_enhanced': details['ai_enhanced']
            }
            for analysis_id, details in self.analysis_types.items()
        }

# Enhanced API integration functions
def call_openrouter_api_enhanced(prompt: str, api_key: str, api_base: str = "https://openrouter.ai/api/v1") -> str:
    """Enhanced API calling with better model selection for document analysis"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Enhanced Legal Document Analysis Assistant"
    }
    
    # Prioritize DeepSeek as primary choice (same as RAG system)
    models_to_try = [
        "deepseek/deepseek-chat-v3-0324:free",
        "microsoft/phi-3-medium-128k-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "anthropic/claude-3-sonnet",
        "openai/gpt-4-turbo-preview", 
        "anthropic/claude-3-haiku"
    ]
    
    for model in models_to_try:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,  # Lower temperature for more consistent analysis
                "max_tokens": 4000,
                "top_p": 0.9
            }
            response = requests.post(f"{api_base}/chat/completions", headers=headers, json=payload, timeout=120)
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and result['choices']:
                    content = result['choices'][0]['message']['content']
                    if content and content.strip():
                        return content.strip()
        except Exception as e:
            logger.error(f"Error with model {model}: {e}")
            continue
    
    return "I apologize, but I'm experiencing technical difficulties with the AI analysis service. Please try again."

# Example usage and integration with your existing FastAPI app:

# Replace your existing DocumentAnalysisEngine with this enhanced version
enhanced_document_analyzer = EnhancedDocumentAnalysisEngine()

# Enhanced document analysis endpoint (replace your existing one)
@app.post("/document-analysis-enhanced", response_model=LegalAnalysisResponse)
async def enhanced_document_analysis_endpoint(
    file: UploadFile = File(...),
    analysis_type: str = Form(...),
    session_id: Optional[str] = Form(None),
    response_style: str = Form("balanced"),
    custom_prompt: Optional[str] = Form(None)
):
    """Enhanced AI-powered document analysis with intelligent component identification"""
    cleanup_expired_conversations()
    
    session_id = session_id or str(uuid.uuid4())
    if session_id not in conversations:
        conversations[session_id] = {
            "messages": [],
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow()
        }
    else:
        conversations[session_id]["last_accessed"] = datetime.utcnow()

    try:
        # Process the document and extract text
        document_text, doc_type = DocumentProcessor.process_document(file)
        
        # Use enhanced AI analysis
        analysis_data = enhanced_document_analyzer.analyze_document_with_ai(
            document_text, 
            analysis_type, 
            custom_prompt
        )
        
        # Get API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return LegalAnalysisResponse(
                response=None,
                error="API configuration error. Please contact administrator.",
                context_found=False,
                sources=[],
                session_id=session_id,
                confidence_score=0.0,
                expand_available=False
            )

        # Call enhanced LLM with the AI-generated prompt
        api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        raw_response = call_openrouter_api_enhanced(analysis_data['prompt'], api_key, api_base)
        
        # Enhanced document info with AI insights
        document_info = {
            'filename': file.filename,
            'file_type': doc_type,
            'file_size': len(document_text),
            'ai_analysis': analysis_data['document_analysis'],
            'metadata': analysis_data['metadata']
        }
        
        # Enhanced analysis metadata
        analysis_metadata = {
            'analysis_type': analysis_type,
            'ai_enhanced': analysis_data['ai_enhanced'],
            'document_type_detected': analysis_data['document_analysis']['primary_type'],
            'detection_confidence': analysis_data['document_analysis']['confidence'],
            'processing_time': datetime.utcnow().isoformat(),
            'prompt_optimization': {
                'prompt_length': analysis_data['prompt_length'],
                'ai_generated': not bool(custom_prompt),
                'custom_prompt_used': bool(custom_prompt)
            }
        }
        
        # Calculate enhanced confidence score
        confidence_score = calculate_enhanced_confidence(document_text, analysis_data, raw_response)
        
        # Add AI insights to response
        enhanced_response = enhance_response_with_insights(raw_response, analysis_data, document_info)
        
        # Update conversation with enhanced context
        user_message = f"AI Document Analysis: {analysis_type} for {file.filename}"
        add_to_conversation(session_id, "user", user_message, document_info=document_info)
        add_to_conversation(session_id, "assistant", enhanced_response, document_info=document_info)
        
        return LegalAnalysisResponse(
            response=enhanced_response,
            error=None,
            context_found=True,
            sources=[],
            session_id=session_id,
            confidence_score=confidence_score,
            expand_available=True,
            document_info=document_info,
            analysis_metadata=analysis_metadata
        )
        
    except Exception as e:
        logger.error(f"Enhanced document analysis failed: {e}", exc_info=True)
        return LegalAnalysisResponse(
            response=None,
            error=f"Enhanced document analysis failed: {str(e)}",
            context_found=False,
            sources=[],
            session_id=session_id,
            confidence_score=0.0,
            expand_available=False
        )

def calculate_enhanced_confidence(document_text: str, analysis_data: Dict, response: str) -> float:
    """Calculate confidence score based on AI analysis quality"""
    base_confidence = analysis_data.get('confidence_score', 0.5)
    
    # Factor 1: Document type detection confidence
    detection_confidence = analysis_data['document_analysis']['confidence']
    
    # Factor 2: Document length and complexity appropriateness
    doc_length = len(document_text)
    length_factor = min(1.0, doc_length / 5000) if doc_length > 500 else 0.3
    
    # Factor 3: Response quality indicators
    response_length = len(response)
    response_factor = min(1.0, response_length / 1000) if response_length > 200 else 0.2
    
    # Factor 4: AI enhancement factor
    ai_enhancement_factor = 0.9 if analysis_data.get('ai_enhanced') else 0.6
    
    # Factor 5: Metadata richness
    metadata = analysis_data.get('metadata', {})
    metadata_richness = len([k for k, v in metadata.items() if v]) / max(len(metadata), 1)
    
    # Weighted combination
    confidence = (
        detection_confidence * 0.25 +
        length_factor * 0.15 +
        response_factor * 0.15 +
        ai_enhancement_factor * 0.25 +
        metadata_richness * 0.20
    )
    
    return min(1.0, max(0.1, confidence))

def enhance_response_with_insights(response: str, analysis_data: Dict, document_info: Dict) -> str:
    """Enhance the AI response with additional insights and metadata"""
    
    doc_analysis = analysis_data['document_analysis']
    metadata = analysis_data['metadata']
    
    # Add AI insights header
    insights_header = f"""
🤖 **AI Document Intelligence Summary**
- **Document Type**: {doc_analysis['primary_type'].title()} (Confidence: {doc_analysis['confidence']:.1%})
- **Complexity Level**: {metadata.get('complexity_metrics', {}).get('complexity_level', 'Unknown').title()}
- **Reading Time**: ~{metadata.get('estimated_reading_time', 0):.1f} minutes
- **Structure Quality**: {metadata.get('document_structure', {}).get('organization_level', 'Unknown').title()}

---

"""
    
    # Add the main AI analysis
    enhanced_response = insights_header + response
    
    # Add actionable insights footer
    action_items = metadata.get('action_items', [])
    financial_info = metadata.get('financial_references', {})
    
    footer_insights = "\n\n---\n\n📊 **Key AI Insights**:\n"
    
    if action_items:
        footer_insights += f"- **Action Items Detected**: {len(action_items)} potential obligations or requirements identified\n"
    
    if financial_info.get('has_financial_terms'):
        amounts = financial_info.get('financial_amounts', [])
        percentages = financial_info.get('percentages', [])
        footer_insights += f"- **Financial Terms**: {len(amounts)} monetary amounts and {len(percentages)} percentages found\n"
    
    if metadata.get('dates_found'):
        footer_insights += f"- **Important Dates**: {len(metadata['dates_found'])} dates identified for timeline tracking\n"
    
    # Add legal language analysis
    legal_lang = metadata.get('legal_language_indicators', {})
    if legal_lang:
        formality = legal_lang.get('formality_level', 'unknown')
        footer_insights += f"- **Language Formality**: {formality.title()} level legal language detected\n"
    
    footer_insights += f"\n💡 **Pro Tip**: This AI analysis used advanced document intelligence to tailor the analysis specifically for {doc_analysis['primary_type']} documents."
    
    return enhanced_response + footer_insights

# Enhanced analysis types endpoint
@app.get("/analysis-types-enhanced")
def get_enhanced_analysis_types():
    """Get available enhanced AI analysis types"""
    return {
        "analysis_types": enhanced_document_analyzer.get_analysis_types(),
        "ai_features": {
            "intelligent_document_detection": True,
            "context_aware_prompts": True,
            "component_identification": True,
            "risk_assessment": True,
            "compliance_checking": True,
            "comparative_analysis": True
        },
        "supported_document_types": [
            "contracts", "leases", "employment_documents", 
            "policies", "legal_memos", "compliance_documents"
        ]
    }

# Document type detection endpoint
@app.post("/detect-document-type")
async def detect_document_type_endpoint(file: UploadFile = File(...)):
    """AI-powered document type detection"""
    try:
        document_text, _ = DocumentProcessor.process_document(file)
        detection_result = enhanced_document_analyzer.detect_document_type_ai(document_text)
        
        return {
            "filename": file.filename,
            "detected_type": detection_result['primary_type'],
            "confidence": detection_result['confidence'],
            "type_scores": detection_result['type_scores'],
            "metadata": detection_result['metadata'],
            "recommendations": {
                "suggested_analysis": get_recommended_analysis_types(detection_result['primary_type']),
                "ai_insights": detection_result['ai_prompt_modifier']
            }
        }
    except Exception as e:
        logger.error(f"Document type detection failed: {e}")
        raise HTTPException(status_code=400, detail=f"Document type detection failed: {str(e)}")

def get_recommended_analysis_types(doc_type: str) -> List[str]:
    """Get recommended analysis types based on document type"""
    recommendations = {
        'contract': ['smart_summary', 'clause_analysis', 'risk_assessment', 'party_obligations'],
        'lease': ['smart_summary', 'timeline_intelligence', 'party_obligations', 'compliance_check'],
        'employment': ['smart_summary', 'compliance_check', 'risk_assessment', 'missing_elements'],
        'policy': ['smart_summary', 'compliance_check', 'language_complexity', 'missing_elements'],
        'legal_memo': ['smart_summary', 'component_extraction', 'language_complexity'],
        'compliance': ['smart_summary', 'compliance_check', 'risk_assessment', 'missing_elements']
    }
    return recommendations.get(doc_type, ['smart_summary', 'component_extraction', 'risk_assessment'])

# Batch analysis endpoint for multiple documents
@app.post("/batch-document-analysis")
async def batch_document_analysis_endpoint(
    files: List[UploadFile] = File(...),
    analysis_type: str = Form(...),
    session_id: Optional[str] = Form(None)
):
    """Analyze multiple documents with AI comparison"""
    session_id = session_id or str(uuid.uuid4())
    
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 documents allowed for batch analysis")
    
    try:
        batch_results = []
        all_document_texts = []
        
        # Process each document
        for file in files:
            document_text, doc_type = DocumentProcessor.process_document(file)
            all_document_texts.append({
                'filename': file.filename,
                'text': document_text,
                'type': doc_type
            })
            
            # Individual analysis
            analysis_data = enhanced_document_analyzer.analyze_document_with_ai(
                document_text, 
                analysis_type
            )
            
            batch_results.append({
                'filename': file.filename,
                'analysis_data': analysis_data,
                'document_info': {
                    'type': doc_type,
                    'size': len(document_text),
                    'ai_detected_type': analysis_data['document_analysis']['primary_type']
                }
            })
        
        # If multiple documents, add comparative analysis
        if len(files) > 1 and analysis_type == 'comparative_analysis':
            comparison_prompt = create_comparative_analysis_prompt(all_document_texts)
            
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
                comparative_response = call_openrouter_api_enhanced(comparison_prompt, api_key, api_base)
                
                batch_results.append({
                    'filename': 'COMPARATIVE_ANALYSIS',
                    'analysis_data': {
                        'prompt': comparison_prompt,
                        'analysis_type': 'comparative_analysis',
                        'ai_enhanced': True
                    },
                    'comparative_analysis': comparative_response
                })
        
        return {
            "session_id": session_id,
            "batch_analysis": batch_results,
            "document_count": len(files),
            "analysis_type": analysis_type,
            "processing_time": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=400, detail=f"Batch analysis failed: {str(e)}")

def create_comparative_analysis_prompt(documents: List[Dict]) -> str:
    """Create a prompt for comparing multiple documents"""
    
    prompt = """You are an expert legal document analyst conducting a comparative analysis.

**DOCUMENTS TO COMPARE**:

"""
    
    for i, doc in enumerate(documents, 1):
        prompt += f"""
**DOCUMENT {i}: {doc['filename']}**
Type: {doc['type']}
Content:
{doc['text'][:2000]}{'...' if len(doc['text']) > 2000 else ''}

---
"""
    
    prompt += """

**COMPARATIVE ANALYSIS INSTRUCTIONS**:

1. **STRUCTURAL COMPARISON**:
   - Compare document organization and formatting
   - Identify different approaches to similar content
   - Note variations in section structure

2. **SUBSTANTIVE DIFFERENCES**:
   - Compare key terms and conditions
   - Identify different obligations and rights
   - Note varying financial or timeline terms

3. **RISK PROFILE COMPARISON**:
   - Compare risk allocation between documents
   - Identify which document is more favorable to which party
   - Note different protective measures

4. **BEST PRACTICES ASSESSMENT**:
   - Which document follows better legal practices
   - Recommendations for harmonization
   - Suggested improvements for each document

5. **SUMMARY MATRIX**:
   Create a comparison table highlighting the key differences.

Provide a comprehensive comparative analysis that would help legal professionals understand the differences and make informed decisions.
"""
    
    return prompt

# Keep your existing RAG endpoints unchanged - they remain perfect as you mentioned
# Just add these enhanced document analysis capabilities alongside them

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting Enhanced Legal Document Analysis Agent with AI Intelligence on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
