"""Document processing service"""
import os
import io
import logging
import tempfile
from typing import Tuple, List
from ..config import FeatureFlags
from ..core.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)

class SafeDocumentProcessor:
    """Safe document processor for various file types with enhanced processing capabilities"""
    
    @staticmethod
    def process_document_safe(file) -> Tuple[str, int, List[str]]:
        """
        Process uploaded document safely with enhanced processing
        Returns: (content, pages_processed, warnings)
        """
        warnings = []
        content = ""
        pages_processed = 0
        
        try:
            filename = getattr(file, 'filename', 'unknown')
            file_ext = os.path.splitext(filename)[1].lower()
            file_content = file.file.read()
            
            if file_ext == '.txt':
                content = file_content.decode('utf-8', errors='ignore')
                pages_processed = SafeDocumentProcessor._estimate_pages_from_text(content)
            elif file_ext == '.pdf':
                content, pages_processed = SafeDocumentProcessor._process_pdf_enhanced(file_content, warnings)
            elif file_ext == '.docx':
                content, pages_processed = SafeDocumentProcessor._process_docx_enhanced(file_content, warnings)
            else:
                try:
                    content = file_content.decode('utf-8', errors='ignore')
                    pages_processed = SafeDocumentProcessor._estimate_pages_from_text(content)
                    warnings.append(f"File type {file_ext} processed as plain text")
                except Exception as e:
                    warnings.append(f"Could not process file: {str(e)}")
                    content = "Unable to process this file type"
                    pages_processed = 0
            
            file.file.seek(0)
            
        except Exception as e:
            warnings.append(f"Error processing document: {str(e)}")
            content = "Error processing document"
            pages_processed = 0
        
        return content, pages_processed, warnings
    
    @staticmethod
    def _estimate_pages_from_text(text: str) -> int:
        """Fixed page estimation based on content analysis"""
        if not text:
            return 0
        
        # Enhanced page estimation logic
        word_count = len(text.split())
        char_count = len(text)
        line_count = len(text.split('\n'))
        
        # Average words per page: 250-500 (legal documents tend to be dense)
        pages_by_words = max(1, word_count // 350)
        
        # Average characters per page: 1500-3000 (including spaces)
        pages_by_chars = max(1, char_count // 2000)
        
        # For documents with many line breaks (structured content)
        pages_by_lines = max(1, line_count // 50)
        
        # Use the median of the three estimates for better accuracy
        estimates = [pages_by_words, pages_by_chars, pages_by_lines]
        estimates.sort()
        estimated_pages = estimates[1]  # median
        
        # Ensure reasonable bounds
        return max(1, min(estimated_pages, 1000))
    
    @staticmethod
    def _process_pdf_enhanced(file_content: bytes, warnings: List[str]) -> Tuple[str, int]:
        """Enhanced PDF processing with Unstructured.io fallback"""
        # Try Unstructured.io first for best results
        if FeatureFlags.UNSTRUCTURED_AVAILABLE:
            try:
                from unstructured.partition.auto import partition
                
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_file.write(file_content)
                    temp_file_path = temp_file.name
                
                try:
                    # Use Unstructured.io for advanced processing
                    elements = partition(filename=temp_file_path)
                    
                    # Extract text and structure
                    text_content = ""
                    page_count = 0
                    
                    for element in elements:
                        if hasattr(element, 'text') and element.text:
                            text_content += element.text + "\n"
                        
                        # Try to get page information
                        if hasattr(element, 'metadata') and element.metadata:
                            if 'page_number' in element.metadata:
                                page_count = max(page_count, element.metadata['page_number'])
                    
                    # Clean up temp file
                    os.unlink(temp_file_path)
                    
                    if page_count == 0:
                        page_count = SafeDocumentProcessor._estimate_pages_from_text(text_content)
                    
                    return text_content, page_count
                    
                except Exception as e:
                    os.unlink(temp_file_path)
                    warnings.append(f"Unstructured.io processing failed: {str(e)}, falling back to PyMuPDF")
                    
            except Exception as e:
                warnings.append(f"Unstructured.io setup failed: {str(e)}")
        
        # Fallback to existing PDF processing
        return SafeDocumentProcessor._process_pdf(file_content, warnings)
    
    @staticmethod
    def _process_pdf(file_content: bytes, warnings: List[str]) -> Tuple[str, int]:
        """Process PDF content with enhanced page counting"""
        try:
            if FeatureFlags.PYMUPDF_AVAILABLE:
                try:
                    import fitz
                    doc = fitz.open(stream=file_content, filetype="pdf")
                    text_content = ""
                    pages = len(doc)
                    for page_num in range(pages):
                        page = doc.load_page(page_num)
                        text_content += page.get_text()
                    doc.close()
                    return text_content, pages
                except Exception as e:
                    warnings.append(f"PyMuPDF error: {str(e)}")
            
            if FeatureFlags.PDFPLUMBER_AVAILABLE:
                try:
                    import pdfplumber
                    with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                        text_content = ""
                        pages = len(pdf.pages)
                        for page in pdf.pages:
                            text_content += page.extract_text() or ""
                    return text_content, pages
                except Exception as e:
                    warnings.append(f"pdfplumber error: {str(e)}")
            
            warnings.append("No PDF processing libraries available. Install PyMuPDF or pdfplumber.")
            return "PDF processing not available", 0
            
        except Exception as e:
            warnings.append(f"Error processing PDF: {str(e)}")
            return "Error processing PDF", 0
    
    @staticmethod
    def _process_docx_enhanced(file_content: bytes, warnings: List[str]) -> Tuple[str, int]:
        """Enhanced DOCX processing with better page estimation"""
        if FeatureFlags.UNSTRUCTURED_AVAILABLE:
            try:
                from unstructured.partition.auto import partition
                
                with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
                    temp_file.write(file_content)
                    temp_file_path = temp_file.name
                
                try:
                    # Use Unstructured.io for advanced processing
                    elements = partition(filename=temp_file_path)
                    
                    text_content = ""
                    for element in elements:
                        if hasattr(element, 'text') and element.text:
                            text_content += element.text + "\n"
                    
                    os.unlink(temp_file_path)
                    
                    pages_estimated = SafeDocumentProcessor._estimate_pages_from_text(text_content)
                    return text_content, pages_estimated
                    
                except Exception as e:
                    os.unlink(temp_file_path)
                    warnings.append(f"Unstructured.io DOCX processing failed: {str(e)}")
                    
            except Exception as e:
                warnings.append(f"Unstructured.io DOCX setup failed: {str(e)}")
        
        # Fallback to existing DOCX processing
        return SafeDocumentProcessor._process_docx(file_content, warnings)
    
    @staticmethod
    def _process_docx(file_content: bytes, warnings: List[str]) -> Tuple[str, int]:
        """Process DOCX content with enhanced page estimation"""
        try:
            try:
                from docx import Document
                doc = Document(io.BytesIO(file_content))
                text_content = ""
                for paragraph in doc.paragraphs:
                    text_content += paragraph.text + "\n"
                
                # Enhanced page estimation for DOCX
                pages_estimated = SafeDocumentProcessor._estimate_pages_from_text(text_content)
                return text_content, pages_estimated
                
            except ImportError:
                warnings.append("python-docx not available. Install with: pip install python-docx")
                return "DOCX processing not available", 0
            except Exception as e:
                warnings.append(f"Error processing DOCX: {str(e)}")
                return "Error processing DOCX", 0
                
        except Exception as e:
            warnings.append(f"Error processing DOCX: {str(e)}")
            return "Error processing DOCX", 0
