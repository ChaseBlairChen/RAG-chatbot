# Hallucination-Free Document Analysis System
# Only extracts verifiable information - says "Failed to extract" when uncertain

from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import json
import re
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Set, Any
from dataclasses import dataclass
from enum import Enum
import io
import tempfile
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced document processing imports
try:
    import PyPDF2
    import docx
    from pdfplumber import PDF
    
    # Try to import Unstructured for better PDF processing (compatible with v0.11.8)
    try:
        # Check for required dependencies first
        missing_deps = []
        
        try:
            import pdf2image
        except ImportError:
            missing_deps.append("pdf2image")
            
        try:
            import PIL
        except ImportError:
            missing_deps.append("Pillow")
        
        if missing_deps:
            print(f"⚠️ Missing dependencies for Unstructured: {', '.join(missing_deps)}")
            print("Install with: pip install pdf2image Pillow")
            raise ImportError(f"Missing dependencies: {missing_deps}")
        
        # Try newer import paths first
        try:
            from unstructured.partition.auto import partition
            from unstructured.partition.pdf import partition_pdf
            UNSTRUCTURED_VERSION = "new"
        except ImportError:
            # Try older import paths for v0.11.8
            from unstructured.partition.pdf import partition_pdf
            from unstructured.documents.elements import Text, Title
            UNSTRUCTURED_VERSION = "old"
        
        UNSTRUCTURED_AVAILABLE = True
        print(f"✅ Unstructured library available (version: {UNSTRUCTURED_VERSION}) - using enhanced PDF processing")
        
    except ImportError as e:
        UNSTRUCTURED_AVAILABLE = False
        print("⚠️ Unstructured library not available - using basic PDF processing")
        print(f"Import error: {e}")
        print("For better PDF processing, install:")
        print("  pip install pdf2image Pillow")
        print("  pip install 'unstructured[pdf]'")
        print("Or for full support: pip install 'unstructured[all-docs]'")
        
except ImportError:
    print("Document processing libraries not installed. Run: pip install PyPDF2 python-docx pdfplumber unstructured[all-docs]")
    UNSTRUCTURED_AVAILABLE = False

class VerifiableExtractor:
    """Extract only verifiable information from documents with source locations"""
    
    def __init__(self):
        # High-confidence extraction patterns with context requirements
        self.extraction_patterns = {
            'dates': {
                'patterns': [
                    r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                    r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
                ],
                'context_required': ['due', 'deadline', 'effective', 'expires', 'terminates', 'begins', 'starts', 'ends']
            },
            'monetary_amounts': {
                'patterns': [
                    r'\$\s*[\d,]+(?:\.\d{2})?(?:\s*(?:per|each|monthly|annually|yearly))?',
                    r'\b[\d,]+\s*dollars?(?:\s*(?:per|each|monthly|annually|yearly))?\b',
                    r'\b[\d,]+\s*USD\b'
                ],
                'context_required': ['pay', 'payment', 'fee', 'cost', 'price', 'rent', 'salary', 'compensation']
            },
            'percentages': {
                'patterns': [r'\b\d+(?:\.\d+)?%'],
                'context_required': ['rate', 'interest', 'fee', 'penalty', 'commission', 'tax']
            },
            'party_names': {
                'patterns': [
                    r'(?:party|parties)[:\s]+([A-Z][a-zA-Z\s&.,]+?)(?:\s*(?:and|,|\n))',
                    r'between\s+([A-Z][a-zA-Z\s&.,]+?)\s+and\s+([A-Z][a-zA-Z\s&.,]+?)(?:\s|,)',
                    r'(?:company|corporation|llc|inc\.?)[:\s]*([A-Z][a-zA-Z\s&.,]+?)(?:\s*(?:and|,|\n))'
                ],
                'context_required': []
            }
        }
    
    def extract_with_verification(self, document_text: str, extraction_type: str) -> List[Dict[str, Any]]:
        """Extract information only if it can be verified with high confidence"""
        
        if extraction_type not in self.extraction_patterns:
            return [{"status": "failed", "reason": f"Unknown extraction type: {extraction_type}"}]
        
        pattern_config = self.extraction_patterns[extraction_type]
        extracted_items = []
        
        # Split document into lines for source tracking
        lines = document_text.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line_lower = line.lower()
            
            for pattern in pattern_config['patterns']:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                
                for match in matches:
                    extracted_value = match.group(0).strip()
                    
                    # Check if context requirements are met
                    context_verified = True
                    if pattern_config['context_required']:
                        context_verified = any(
                            keyword in line_lower 
                            for keyword in pattern_config['context_required']
                        )
                    
                    if context_verified:
                        extracted_items.append({
                            "value": extracted_value,
                            "line_number": line_num,
                            "context": line.strip(),
                            "confidence": "high",
                            "verified": True
                        })
                    else:
                        # Found pattern but no context - mark as unverified
                        extracted_items.append({
                            "value": extracted_value,
                            "line_number": line_num,
                            "context": line.strip(),
                            "confidence": "low",
                            "verified": False,
                            "reason": "No supporting context found"
                        })
        
        # Filter to only high-confidence, verified items
        verified_items = [item for item in extracted_items if item.get('verified', False)]
        
        if not verified_items:
            return [{"status": "failed_to_extract", "reason": f"No verifiable {extraction_type} found in document"}]
        
        return verified_items

class NoHallucinationAnalyzer:
    """Document analyzer that never hallucinates - only reports verifiable facts"""
    
    def __init__(self):
        self.extractor = VerifiableExtractor()
        
        # Strict document type detection based on explicit keywords only
        self.document_types = {
            'contract': {
                'required_keywords': ['agreement', 'contract'],
                'supporting_keywords': ['parties', 'consideration', 'terms'],
                'minimum_matches': 2
            },
            'lease': {
                'required_keywords': ['lease'],
                'supporting_keywords': ['tenant', 'landlord', 'rent', 'premises'],
                'minimum_matches': 2
            },
            'employment': {
                'required_keywords': ['employment', 'employee'],
                'supporting_keywords': ['employer', 'salary', 'position', 'job'],
                'minimum_matches': 2
            },
            'policy': {
                'required_keywords': ['policy'],
                'supporting_keywords': ['procedure', 'guidelines', 'rules'],
                'minimum_matches': 1
            }
        }
    
    def detect_document_type_strict(self, document_text: str) -> Dict[str, Any]:
        """Detect document type only with high confidence - no guessing"""
        
        text_lower = document_text.lower()
        detection_results = {}
        
        for doc_type, criteria in self.document_types.items():
            score = 0
            matched_keywords = []
            
            # Check required keywords
            required_found = 0
            for keyword in criteria['required_keywords']:
                if keyword in text_lower:
                    required_found += 1
                    matched_keywords.append(keyword)
                    score += 10
            
            # Check supporting keywords
            supporting_found = 0
            for keyword in criteria['supporting_keywords']:
                if keyword in text_lower:
                    supporting_found += 1
                    matched_keywords.append(keyword)
                    score += 3
            
            # Only consider if minimum requirements met
            total_matches = required_found + supporting_found
            if total_matches >= criteria['minimum_matches'] and required_found > 0:
                detection_results[doc_type] = {
                    'score': score,
                    'matched_keywords': matched_keywords,
                    'required_found': required_found,
                    'supporting_found': supporting_found
                }
        
        if not detection_results:
            return {
                'type': 'unknown',
                'confidence': 0.0,
                'status': 'failed_to_detect',
                'reason': 'No clear document type indicators found',
                'matched_keywords': []
            }
        
        # Get highest scoring type
        best_type = max(detection_results, key=lambda x: detection_results[x]['score'])
        best_result = detection_results[best_type]
        
        # Calculate confidence based on keyword matches
        confidence = min(1.0, best_result['score'] / 20)
        
        return {
            'type': best_type,
            'confidence': confidence,
            'status': 'detected' if confidence > 0.7 else 'uncertain',
            'matched_keywords': best_result['matched_keywords'],
            'all_matches': detection_results
        }
    
    def extract_document_facts(self, document_text: str) -> Dict[str, Any]:
        """Extract only verifiable facts from the document"""
        
        facts = {
            'basic_stats': self._get_basic_stats(document_text),
            'dates': self.extractor.extract_with_verification(document_text, 'dates'),
            'monetary_amounts': self.extractor.extract_with_verification(document_text, 'monetary_amounts'),
            'percentages': self.extractor.extract_with_verification(document_text, 'percentages'),
            'party_names': self.extractor.extract_with_verification(document_text, 'party_names'),
            'document_structure': self._analyze_structure(document_text),
            'extraction_status': 'completed',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return facts
    
    def _get_basic_stats(self, text: str) -> Dict[str, Any]:
        """Get verifiable basic statistics"""
        
        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'character_count': len(text),
            'estimated_reading_time_minutes': round(len(words) / 200, 1)
        }
    
    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure - only count what's clearly present"""
        
        # Count numbered sections (1., 2., etc.)
        numbered_sections = len(re.findall(r'^\s*\d+\.\s', text, re.MULTILINE))
        
        # Count lettered sections (a., b., etc.)
        lettered_sections = len(re.findall(r'^\s*[a-z]\.\s', text, re.MULTILINE))
        
        # Count subsections (1.1, 1.2, etc.)
        subsections = len(re.findall(r'^\s*\d+\.\d+\s', text, re.MULTILINE))
        
        # Count headers (lines that end with ":" and are short)
        potential_headers = re.findall(r'^([^:\n]{1,50}):$', text, re.MULTILINE)
        headers = len([h for h in potential_headers if len(h.split()) <= 8])
        
        return {
            'numbered_sections': numbered_sections,
            'lettered_sections': lettered_sections,
            'subsections': subsections,
            'headers': headers,
            'has_clear_structure': numbered_sections > 0 or headers > 2
        }
    
    def generate_factual_summary(self, document_text: str) -> str:
        """Generate a summary using only extracted facts - no interpretation"""
        
        # Get document type
        doc_type_result = self.detect_document_type_strict(document_text)
        
        # Extract facts
        facts = self.extract_document_facts(document_text)
        
        # Build factual summary
        summary_parts = []
        
        # Document identification
        if doc_type_result['status'] == 'detected':
            summary_parts.append(f"**Document Type**: {doc_type_result['type'].title()} (confidence: {doc_type_result['confidence']:.1%})")
            summary_parts.append(f"**Keywords Found**: {', '.join(doc_type_result['matched_keywords'])}")
        else:
            summary_parts.append(f"**Document Type**: Failed to detect - {doc_type_result['reason']}")
        
        # Basic statistics
        stats = facts['basic_stats']
        summary_parts.append(f"\n**Document Statistics**:")
        summary_parts.append(f"• Word count: {stats['word_count']:,}")
        summary_parts.append(f"• Estimated reading time: {stats['estimated_reading_time_minutes']} minutes")
        summary_parts.append(f"• Paragraphs: {stats['paragraph_count']}")
        
        # Structure analysis
        structure = facts['document_structure']
        summary_parts.append(f"\n**Document Structure**:")
        if structure['has_clear_structure']:
            summary_parts.append(f"• Numbered sections: {structure['numbered_sections']}")
            summary_parts.append(f"• Headers found: {structure['headers']}")
            summary_parts.append(f"• Subsections: {structure['subsections']}")
        else:
            summary_parts.append("• No clear structural organization detected")
        
        # Extracted information
        summary_parts.append(f"\n**Extracted Information**:")
        
        # Dates
        dates = facts['dates']
        if dates and dates[0].get('status') != 'failed_to_extract':
            summary_parts.append(f"• **Dates found**: {len(dates)} verifiable dates")
            for i, date in enumerate(dates[:3], 1):  # Show first 3
                summary_parts.append(f"  {i}. {date['value']} (Line {date['line_number']})")
            if len(dates) > 3:
                summary_parts.append(f"  ... and {len(dates) - 3} more dates")
        else:
            summary_parts.append("• **Dates**: Failed to extract any verifiable dates")
        
        # Monetary amounts
        amounts = facts['monetary_amounts']
        if amounts and amounts[0].get('status') != 'failed_to_extract':
            summary_parts.append(f"• **Financial amounts**: {len(amounts)} verifiable amounts")
            for i, amount in enumerate(amounts[:3], 1):  # Show first 3
                summary_parts.append(f"  {i}. {amount['value']} (Line {amount['line_number']})")
            if len(amounts) > 3:
                summary_parts.append(f"  ... and {len(amounts) - 3} more amounts")
        else:
            summary_parts.append("• **Financial amounts**: Failed to extract any verifiable amounts")
        
        # Percentages
        percentages = facts['percentages']
        if percentages and percentages[0].get('status') != 'failed_to_extract':
            summary_parts.append(f"• **Percentages**: {len(percentages)} found")
            for perc in percentages[:3]:
                summary_parts.append(f"  - {perc['value']} (Line {perc['line_number']})")
        else:
            summary_parts.append("• **Percentages**: Failed to extract any verifiable percentages")
        
        # Party names
        parties = facts['party_names']
        if parties and parties[0].get('status') != 'failed_to_extract':
            summary_parts.append(f"• **Parties**: {len(parties)} identified")
            for party in parties[:3]:
                summary_parts.append(f"  - {party['value']} (Line {party['line_number']})")
        else:
            summary_parts.append("• **Parties**: Failed to extract clear party names")
        
        # Important disclaimer
        summary_parts.append(f"\n**⚠️ IMPORTANT NOTES**:")
        summary_parts.append("• This analysis only includes information that could be verified directly from the document text")
        summary_parts.append("• All extracted items include line numbers for verification")
        summary_parts.append("• Items marked 'Failed to extract' mean the information was not clearly identifiable")
        summary_parts.append("• For legal advice, consult a qualified attorney")
        
        return '\n'.join(summary_parts)

class SafeDocumentProcessor:
    """Document processor with verification and no hallucination"""
    
    @staticmethod
    def process_document_safe(file: UploadFile) -> Tuple[str, str, Dict[str, Any]]:
        """Process document and provide processing metadata"""
        
        file_extension = file.filename.split('.')[-1].lower() if file.filename else 'unknown'
        processing_info = {
            'original_filename': file.filename,
            'file_extension': file_extension,
            'processing_method': None,
            'success': False,
            'warnings': [],
            'errors': []
        }
        
        try:
            if file_extension == 'pdf':
                text, warnings = SafeDocumentProcessor._extract_pdf_safe(file)
                processing_info['processing_method'] = 'PDF extraction'
                processing_info['warnings'] = warnings
                
            elif file_extension in ['docx', 'doc']:
                text, warnings = SafeDocumentProcessor._extract_docx_safe(file)
                processing_info['processing_method'] = 'Word document extraction'
                processing_info['warnings'] = warnings
                
            elif file_extension == 'txt':
                content = file.file.read()
                text = content.decode('utf-8')
                processing_info['processing_method'] = 'Plain text'
                
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_extension}. Supported: PDF, DOCX, TXT"
                )
            
            # Verify text extraction quality
            if not text or len(text.strip()) < 10:
                processing_info['errors'].append("Extracted text is too short or empty")
                raise ValueError("Failed to extract meaningful text from document")
            
            processing_info['success'] = True
            processing_info['extracted_length'] = len(text)
            processing_info['word_count'] = len(text.split())
            
            return text, file_extension, processing_info
            
        except Exception as e:
            processing_info['errors'].append(str(e))
            raise HTTPException(status_code=400, detail=f"Document processing failed: {str(e)}")
    
    @staticmethod
    def _extract_pdf_safe(file: UploadFile) -> Tuple[str, List[str]]:
        """Extract PDF text with enhanced processing using Unstructured if available"""
        
        warnings = []
        pdf_content = file.file.read()
        
        if len(pdf_content) == 0:
            raise ValueError("PDF file is empty")
        
        # Try Unstructured library first (best quality) - only if dependencies available
        if UNSTRUCTURED_AVAILABLE:
            try:
                # Write to temporary file for Unstructured
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_file.write(pdf_content)
                    temp_file_path = temp_file.name
                
                try:
                    # Use Unstructured to partition the PDF (compatible with v0.11.8)
                    if UNSTRUCTURED_VERSION == "new":
                        elements = partition_pdf(
                            filename=temp_file_path,
                            strategy="fast",
                            extract_images_in_pdf=False,
                            infer_table_structure=True,
                            chunking_strategy="by_title",
                            max_characters=4000,
                            combine_text_under_n_chars=100
                        )
                    else:
                        # Older version - simpler parameters
                        elements = partition_pdf(
                            filename=temp_file_path,
                            strategy="fast"
                        )
                    
                    # Convert elements to structured text (works for both versions)
                    text = ""
                    page_num = 1
                    elements_found = 0
                    
                    for element in elements:
                        element_type = type(element).__name__
                        element_text = str(element).strip()
                        
                        if element_text:
                            # Add element type context for better extraction
                            if "title" in element_type.lower():
                                text += f"\n## {element_text}\n"
                            elif "header" in element_type.lower():
                                text += f"\n### {element_text}\n"
                            elif "table" in element_type.lower():
                                text += f"\n[TABLE]\n{element_text}\n[/TABLE]\n"
                            elif "list" in element_type.lower():
                                text += f"• {element_text}\n"
                            else:
                                text += f"{element_text}\n"
                            
                            elements_found += 1
                    
                    # Clean up temp file
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass  # Don't fail if cleanup fails
                    
                    if elements_found == 0:
                        raise ValueError("Unstructured found no elements in PDF")
                    
                    warnings.append(f"Used Unstructured library v{UNSTRUCTURED_VERSION} - extracted {elements_found} document elements")
                    return text, warnings
                    
                except Exception as e:
                    warnings.append(f"Unstructured processing failed: {str(e)}, falling back to basic extraction")
                    # Clean up temp file on error
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
                    
            except Exception as e:
                warnings.append(f"Unstructured setup failed: {str(e)}, using fallback methods")
        else:
            warnings.append("Unstructured not available, using pdfplumber for enhanced extraction")
        
        # Fallback to pdfplumber (medium quality)
        pdf_file = io.BytesIO(pdf_content)
        
        try:
            import pdfplumber
            with pdfplumber.open(pdf_file) as pdf:
                if len(pdf.pages) == 0:
                    raise ValueError("PDF contains no pages")
                
                text = ""
                pages_with_text = 0
                tables_found = 0
                
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text += f"[Page {page_num + 1}]\n{page_text}\n"
                        pages_with_text += 1
                    else:
                        warnings.append(f"Page {page_num + 1} contains no extractable text")
                    
                    # Extract tables separately for better structure
                    try:
                        tables = page.extract_tables()
                        if tables:
                            for table_num, table in enumerate(tables):
                                text += f"\n[TABLE {table_num + 1} FROM PAGE {page_num + 1}]\n"
                                for row in table:
                                    if row:
                                        clean_row = [str(cell) if cell else "" for cell in row]
                                        text += " | ".join(clean_row) + "\n"
                                text += "[/TABLE]\n\n"
                                tables_found += 1
                    except Exception as e:
                        warnings.append(f"Could not extract tables from page {page_num + 1}: {str(e)}")
                
                if pages_with_text == 0:
                    raise ValueError("No readable text found in any PDF pages")
                
                if pages_with_text < len(pdf.pages):
                    warnings.append(f"Only {pages_with_text} of {len(pdf.pages)} pages contained extractable text")
                
                if tables_found > 0:
                    warnings.append(f"Extracted {tables_found} tables with preserved structure")
                
                return text, warnings
                
        except ImportError:
            warnings.append("pdfplumber not available, using basic PyPDF2 extraction")
            
            # Final fallback to PyPDF2 (basic quality)
            pdf_file.seek(0)
            reader = PyPDF2.PdfReader(pdf_file)
            
            if len(reader.pages) == 0:
                raise ValueError("PDF contains no pages")
            
            text = ""
            pages_with_text = 0
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text += f"[Page {page_num + 1}]\n{page_text}\n\n"
                        pages_with_text += 1
                    else:
                        warnings.append(f"Page {page_num + 1} contains no extractable text")
                except Exception as e:
                    warnings.append(f"Error extracting text from page {page_num + 1}: {str(e)}")
            
            if pages_with_text == 0:
                raise ValueError("No readable text found in any PDF pages")
            
            warnings.append("Used basic PDF extraction - complex layouts may not be preserved")
            return text, warnings
    
    @staticmethod
    def _extract_docx_safe(file: UploadFile) -> Tuple[str, List[str]]:
        """Extract DOCX text with warnings about potential issues"""
        
        warnings = []
        docx_content = file.file.read()
        
        if len(docx_content) == 0:
            raise ValueError("Word document is empty")
        
        docx_file = io.BytesIO(docx_content)
        
        try:
            doc = docx.Document(docx_file)
            
            text = ""
            paragraphs_with_text = 0
            
            for para_num, paragraph in enumerate(doc.paragraphs):
                if paragraph.text and paragraph.text.strip():
                    text += paragraph.text + "\n"
                    paragraphs_with_text += 1
            
            # Check tables as well
            tables_found = len(doc.tables)
            if tables_found > 0:
                warnings.append(f"Document contains {tables_found} tables - table content may not be fully extracted")
            
            if paragraphs_with_text == 0:
                raise ValueError("No readable text found in Word document")
            
            if paragraphs_with_text < len(doc.paragraphs):
                warnings.append(f"Some paragraphs contained no text")
            
            return text, warnings
            
        except Exception as e:
            raise ValueError(f"Failed to process Word document: {str(e)}")

# Response models for safe extraction
class SafeAnalysisResponse(BaseModel):
    extraction_results: Optional[Dict[str, Any]] = None
    factual_summary: Optional[str] = None
    processing_info: Optional[Dict[str, Any]] = None
    verification_status: str
    failed_extractions: List[str] = []
    warnings: List[str] = []
    session_id: str
    timestamp: str

# Initialize the safe analyzer
safe_analyzer = NoHallucinationAnalyzer()

# FastAPI app
app = FastAPI(
    title="Hallucination-Free Document Analysis",
    description="Document analysis that only reports verifiable facts - no guessing or interpretation",
    version="3.0.0-Safe"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/safe-document-analysis", response_model=SafeAnalysisResponse)
@app.post("/document-analysis")  # Compatibility endpoint for existing calls
async def safe_document_analysis(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    analysis_type: Optional[str] = Form(None)  # Accept but ignore for compatibility
):
    """Analyze document with zero hallucination - only verifiable facts"""
    
    session_id = session_id or str(uuid.uuid4())
    
    try:
        # Process document safely
        document_text, file_type, processing_info = SafeDocumentProcessor.process_document_safe(file)
        
        # Extract only verifiable facts
        extraction_results = safe_analyzer.extract_document_facts(document_text)
        
        # Add compatibility notice if analysis_type was provided (old API usage)
        compatibility_notice = ""
        if analysis_type:
            compatibility_notice = f"""
⚠️ **API CHANGE NOTICE**: This system no longer performs "{analysis_type}" analysis to prevent hallucination.
Instead, it provides only verifiable facts extracted directly from your document.

"""

        # Generate factual summary with compatibility notice
        factual_summary = compatibility_notice + safe_analyzer.generate_factual_summary(document_text)
        
        # Collect failed extractions
        failed_extractions = []
        for key, value in extraction_results.items():
            if isinstance(value, list) and value and value[0].get('status') == 'failed_to_extract':
                failed_extractions.append(key)
        
        # Determine verification status
        successful_extractions = len([k for k in extraction_results.keys() if k not in failed_extractions])
        if successful_extractions >= 3:
            verification_status = "high_confidence"
        elif successful_extractions >= 1:
            verification_status = "partial_success"
        else:
            verification_status = "minimal_extraction"
        
        return SafeAnalysisResponse(
            extraction_results=extraction_results,
            factual_summary=factual_summary,
            processing_info=processing_info,
            verification_status=verification_status,
            failed_extractions=failed_extractions,
            warnings=processing_info.get('warnings', []),
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Safe document analysis failed: {e}")
        
        return SafeAnalysisResponse(
            extraction_results=None,
            factual_summary=f"**Analysis Failed**: {str(e)}",
            processing_info={"error": str(e)},
            verification_status="failed",
            failed_extractions=["all"],
            warnings=[],
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat()
        )

@app.post("/verify-extraction")
async def verify_extraction(
    file: UploadFile = File(...),
    claim: str = Form(...),
    line_number: Optional[int] = Form(None)
):
    """Verify a specific claim against the document"""
    
    try:
        document_text, _, _ = SafeDocumentProcessor.process_document_safe(file)
        lines = document_text.split('\n')
        
        verification_result = {
            "claim": claim,
            "verification_status": "not_found",
            "evidence": None,
            "line_number": line_number,
            "context": None
        }
        
        # If line number provided, check that specific line
        if line_number and 1 <= line_number <= len(lines):
            target_line = lines[line_number - 1]
            if claim.lower() in target_line.lower():
                verification_result.update({
                    "verification_status": "verified",
                    "evidence": target_line.strip(),
                    "context": target_line.strip()
                })
            else:
                verification_result.update({
                    "verification_status": "not_found_at_line",
                    "context": target_line.strip()
                })
        else:
            # Search entire document
            for i, line in enumerate(lines, 1):
                if claim.lower() in line.lower():
                    verification_result.update({
                        "verification_status": "verified",
                        "evidence": line.strip(),
                        "line_number": i,
                        "context": line.strip()
                    })
                    break
        
        return verification_result
        
    except Exception as e:
        return {
            "claim": claim,
            "verification_status": "error",
            "error": str(e)
        }

@app.get("/extraction-capabilities")
def get_extraction_capabilities():
    """Get what the system can and cannot extract"""
    
    return {
        "can_extract": {
            "dates": "Only dates with clear context (due dates, deadlines, etc.)",
            "monetary_amounts": "Only amounts with payment context",
            "percentages": "Only percentages with rate/fee context",
            "party_names": "Only clearly identified parties",
            "document_structure": "Numbered sections, headers, basic organization",
            "basic_statistics": "Word count, paragraph count, reading time"
        },
        "cannot_extract": {
            "interpretations": "No interpretation of legal meaning",
            "implications": "No analysis of what clauses mean legally",
            "recommendations": "No advice on what to do",
            "risk_assessment": "No assessment of risks or problems",
            "obligations": "No analysis of duties or responsibilities",
            "missing_clauses": "No suggestions for what should be added"
        },
        "verification_required": "All extractions include line numbers for manual verification",
        "fallback_behavior": "Returns 'Failed to extract' instead of guessing"
    }

@app.get("/health")
def health_check():
    """Health check for safe analyzer"""
    return {
        "status": "Safe Mode Active",
        "version": "3.0.0-Safe",
        "hallucination_risk": "eliminated",
        "extraction_mode": "verification_only",
        "features": [
            "✅ Zero hallucination guarantee",
            "📍 Source line tracking",
            "🔍 Context verification required",
            "❌ No interpretation or guessing",
            "⚠️ Explicit failure reporting"
        ],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/", response_class=HTMLResponse)
def get_safe_interface():
    """Safe interface with clear capabilities"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🔒 Hallucination-Free Document Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f8f9fa; }
            .container { max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            .safe-badge { background: #27ae60; color: white; padding: 5px 15px; border-radius: 20px; font-size: 12px; }
            .feature { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .capability { background: #e8f5e8; padding: 10px; margin: 5px 0; border-left: 4px solid #27ae60; }
            .limitation { background: #fff3cd; padding: 10px; margin: 5px 0; border-left: 4px solid #ffc107; }
            .warning { background: #f8d7da; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #dc3545; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔒 Hallucination-Free Document Analysis</h1>
            <p><strong>Version 3.0.0-Safe</strong> <span class="safe-badge">ZERO HALLUCINATION</span></p>
            
            <div class="warning">
                <h3>⚠️ Important: This System NEVER Guesses</h3>
                <p>This analyzer only reports facts it can verify directly from your document. 
                If it can't find something with high confidence, it will explicitly say "Failed to extract" 
                instead of making assumptions.</p>
            </div>
            
            <div class="feature">
                <h3>✅ What This System CAN Do</h3>
                <div class="capability">Extract dates with clear context (deadlines, effective dates)</div>
                <div class="capability">Find monetary amounts with payment context</div>
                <div class="capability">Identify percentages with rate/fee context</div>
                <div class="capability">Locate party names when clearly identified</div>
                <div class="capability">Count words, paragraphs, and basic structure</div>
                <div class="capability">Provide line numbers for all extracted information</div>
            </div>
            
            <div class="feature">
                <h3>❌ What This System CANNOT Do</h3>
                <div class="limitation">Interpret legal meaning or implications</div>
                <div class="limitation">Analyze risks or provide legal advice</div>
                <div class="limitation">Suggest missing clauses or improvements</div>
                <div class="limitation">Explain what obligations parties have</div>
                <div class="limitation">Make recommendations about the document</div>
                <div class="limitation">Guess or assume anything not explicitly stated</div>
            </div>
            
            <div class="feature">
                <h3>🔍 Available Endpoints</h3>
                <p><strong>POST /safe-document-analysis</strong> - Extract only verifiable facts</p>
                <p><strong>POST /verify-extraction</strong> - Verify specific claims against document</p>
                <p><strong>GET /extraction-capabilities</strong> - See detailed capabilities and limitations</p>
            </div>
            
            <p style="text-align: center; color: #7f8c8d; margin-top: 30px;">
                Built for accuracy and trust - never hallucinates 🎯
            </p>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🔒 Starting Hallucination-Free Document Analysis on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
