"""Enhanced document processing service with OCR support"""
import os
import io
import logging
import tempfile
from typing import Tuple, List
import hashlib
from ..config import FeatureFlags
from ..core.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)

class SafeDocumentProcessor:
    """Safe document processor for various file types with enhanced processing capabilities"""
    
    @staticmethod
    def process_document_safe(file) -> Tuple[str, int, List[str]]:
        """
        Process uploaded document safely with enhanced processing and OCR fallback
        Returns: (content, pages_processed, warnings)
        """
        warnings = []
        content = ""
        pages_processed = 0
        
        try:
            filename = getattr(file, 'filename', 'unknown')
            file_ext = os.path.splitext(filename)[1].lower()
            file_content = file.file.read()
            
            logger.info(f"Processing file: {filename}, extension: {file_ext}, size: {len(file_content)} bytes")
            
            if file_ext == '.txt':
                content = file_content.decode('utf-8', errors='ignore')
                pages_processed = SafeDocumentProcessor._estimate_pages_from_text(content)
                logger.info(f"Text file processed: {len(content)} chars, {pages_processed} pages")
            elif file_ext == '.pdf':
                # Enhanced PDF processing with multiple fallbacks
                content, pages_processed = SafeDocumentProcessor._process_pdf_multi_method(file_content, warnings)
                logger.info(f"PDF processed: {len(content)} chars, {pages_processed} pages")
            elif file_ext == '.docx':
                content, pages_processed = SafeDocumentProcessor._process_docx_enhanced(file_content, warnings)
                logger.info(f"DOCX processed: {len(content)} chars, {pages_processed} pages")
            else:
                try:
                    content = file_content.decode('utf-8', errors='ignore')
                    pages_processed = SafeDocumentProcessor._estimate_pages_from_text(content)
                    warnings.append(f"File type {file_ext} processed as plain text")
                except Exception as e:
                    warnings.append(f"Could not process file: {str(e)}")
                    content = "Unable to process this file type"
                    pages_processed = 0
            
            # Log extraction quality
            logger.info(f"Content preview: {content[:200]}..." if len(content) > 200 else f"Content: {content}")
            
            # Validate extraction quality
            if not SafeDocumentProcessor._validate_extraction(content, filename):
                warnings.append("Low quality extraction detected - may need manual review")
                logger.warning(f"Low quality extraction for {filename}")
            
            file.file.seek(0)
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}", exc_info=True)
            warnings.append(f"Error processing document: {str(e)}")
            content = "Error processing document"
            pages_processed = 0
        
        logger.info(f"Final result: {len(content)} chars, {pages_processed} pages, {len(warnings)} warnings")
        return content, pages_processed, warnings
    
    @staticmethod
    def _validate_extraction(text: str, filename: str) -> bool:
        """Validate if extraction was successful"""
        if len(text) < 100:
            logger.warning(f"Extraction too short for {filename}: {len(text)} chars")
            return False
        
        # Check for common extraction failures
        if text.count('�') > 10:  # Unicode errors
            logger.warning(f"Many unicode errors in {filename}")
            return False
        
        # Check if mostly whitespace
        if len(text.strip()) < len(text) * 0.1:
            logger.warning(f"Mostly whitespace in {filename}")
            return False
        
        return True
    
    @staticmethod
    def _process_pdf_multi_method(file_content: bytes, warnings: List[str]) -> Tuple[str, int]:
        """Process PDF with multiple methods and OCR fallback"""
        
        # Method 1: Try Unstructured.io first (best quality)
        if FeatureFlags.UNSTRUCTURED_AVAILABLE:
            try:
                content, pages = SafeDocumentProcessor._process_pdf_unstructured(file_content, warnings)
                if content and len(content) > 100:
                    logger.info("✅ PDF processed with Unstructured.io")
                    return content, pages
            except Exception as e:
                warnings.append(f"Unstructured.io failed: {str(e)}")
        
        # Method 2: Try PyMuPDF with layout preservation
        if FeatureFlags.PYMUPDF_AVAILABLE:
            try:
                content, pages = SafeDocumentProcessor._process_pdf_pymupdf_enhanced(file_content, warnings)
                if content and len(content) > 100:
                    logger.info("✅ PDF processed with PyMuPDF")
                    return content, pages
            except Exception as e:
                warnings.append(f"PyMuPDF failed: {str(e)}")
        
        # Method 3: Try pdfplumber
        if FeatureFlags.PDFPLUMBER_AVAILABLE:
            try:
                content, pages = SafeDocumentProcessor._process_pdf_pdfplumber(file_content, warnings)
                if content and len(content) > 100:
                    logger.info("✅ PDF processed with pdfplumber")
                    return content, pages
            except Exception as e:
                warnings.append(f"pdfplumber failed: {str(e)}")
        
        # Method 4: OCR fallback for scanned PDFs
        if FeatureFlags.OCR_AVAILABLE:
            try:
                content, pages = SafeDocumentProcessor._process_pdf_with_ocr(file_content, warnings)
                if content and len(content) > 100:
                    logger.info("✅ PDF processed with OCR")
                    return content, pages
            except Exception as e:
                warnings.append(f"OCR failed: {str(e)}")
        
        warnings.append("All PDF processing methods failed")
        return "PDF processing failed - please try a different format", 0
    
    @staticmethod
    def _process_pdf_pymupdf_enhanced(file_content: bytes, warnings: List[str]) -> Tuple[str, int]:
        """Enhanced PyMuPDF processing with better text extraction"""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(stream=file_content, filetype="pdf")
            all_text = []
            page_count = len(doc)  # Get actual page count
            
            for page_num in range(page_count):
                page = doc.load_page(page_num)
                
                # Try different extraction methods
                # Method 1: Get text with layout
                text = page.get_text("dict")
                page_text = SafeDocumentProcessor._reconstruct_layout(text)
                
                # If text is too short, might be scanned or have issues
                if len(page_text.strip()) < 50:
                    # Method 2: Simple text extraction
                    page_text = page.get_text()
                    
                    # If still too short, flag for OCR
                    if len(page_text.strip()) < 50:
                        warnings.append(f"Page {page_num + 1} has little text - may need OCR")
                
                all_text.append(f"\n--- Page {page_num + 1} ---\n{page_text}")
            
            doc.close()
            
            # Join all text
            final_text = "\n".join(all_text)
            return final_text, page_count
            
        except Exception as e:
            warnings.append(f"PyMuPDF enhanced extraction failed: {str(e)}")
            raise
    
    @staticmethod
    def _reconstruct_layout(text_dict):
        """Reconstruct text layout from PyMuPDF dict output"""
        blocks = []
        
        for block in sorted(text_dict.get("blocks", []), key=lambda b: (b.get("bbox", [0])[1], b.get("bbox", [0])[0])):
            if block.get("type") == 0:  # Text block
                block_lines = []
                
                for line in block.get("lines", []):
                    line_text = []
                    for span in line.get("spans", []):
                        span_text = span.get("text", "")
                        if span_text.strip():
                            line_text.append(span_text)
                    
                    if line_text:
                        block_lines.append(" ".join(line_text))
                
                if block_lines:
                    blocks.append("\n".join(block_lines))
        
        return "\n\n".join(blocks)
    
    @staticmethod
    def _process_pdf_with_ocr(file_content: bytes, warnings: List[str]) -> Tuple[str, int]:
        """Process PDF using OCR for scanned documents"""
        try:
            import pytesseract
            from pdf2image import convert_from_bytes
            from PIL import Image
            
            # Convert PDF to images
            images = convert_from_bytes(file_content, dpi=300)
            all_text = []
            
            for i, image in enumerate(images):
                try:
                    # Preprocess image for better OCR
                    image = SafeDocumentProcessor._preprocess_image_for_ocr(image)
                    
                    # Perform OCR
                    text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
                    
                    if text.strip():
                        all_text.append(f"\n--- Page {i + 1} (OCR) ---\n{text}")
                    else:
                        warnings.append(f"Page {i + 1}: No text found with OCR")
                        
                except Exception as e:
                    warnings.append(f"OCR failed for page {i + 1}: {str(e)}")
            
            if not all_text:
                raise ValueError("No text extracted with OCR")
                
            return "\n".join(all_text), len(images)
            
        except ImportError:
            warnings.append("OCR libraries not installed. Install pytesseract and pdf2image")
            raise
        except Exception as e:
            warnings.append(f"OCR processing failed: {str(e)}")
            raise
    
    @staticmethod
    def _preprocess_image_for_ocr(image):
        """Preprocess image to improve OCR accuracy"""
        try:
            from PIL import ImageEnhance, ImageFilter
            
            # Convert to grayscale
            image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Remove noise
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            return image
        except Exception:
            return image  # Return original if preprocessing fails
    
    @staticmethod
    def _process_pdf_pdfplumber(file_content: bytes, warnings: List[str]) -> Tuple[str, int]:
        """Process PDF using pdfplumber"""
        try:
            import pdfplumber
            
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                all_text = []
                
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    
                    # Also try to extract tables
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            table_text = "\n".join(["\t".join([str(cell) if cell else "" for cell in row]) for row in table])
                            text += f"\n\n[Table]\n{table_text}\n"
                    
                    if text.strip():
                        all_text.append(f"\n--- Page {i + 1} ---\n{text}")
                
                return "\n".join(all_text), len(pdf.pages)
                
        except Exception as e:
            warnings.append(f"pdfplumber failed: {str(e)}")
            raise
    
    @staticmethod
    def _process_pdf_unstructured(file_content: bytes, warnings: List[str]) -> Tuple[str, int]:
        """Process PDF using Unstructured.io"""
        try:
            from unstructured.partition.auto import partition
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            try:
                elements = partition(filename=temp_file_path, strategy="hi_res")
                
                text_content = []
                page_count = 0
                
                for element in elements:
                    if hasattr(element, 'text') and element.text:
                        text_content.append(element.text)
                    
                    if hasattr(element, 'metadata') and element.metadata:
                        if hasattr(element.metadata, 'page_number'):
                            page_count = max(page_count, element.metadata.page_number)
                
                os.unlink(temp_file_path)
                
                final_text = "\n\n".join(text_content)
                if page_count == 0:
                    page_count = SafeDocumentProcessor._estimate_pages_from_text(final_text)
                
                return final_text, page_count
                
            except Exception as e:
                os.unlink(temp_file_path)
                raise
                
        except Exception as e:
            warnings.append(f"Unstructured.io processing failed: {str(e)}")
            raise
    
    @staticmethod
    def _estimate_pages_from_text(text: str) -> int:
        """Fixed page estimation based on content analysis"""
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for page estimation")
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
        
        # Log the estimation for debugging
        logger.debug(f"Page estimation: words={word_count}, chars={char_count}, lines={line_count}")
        logger.debug(f"Estimates: by_words={pages_by_words}, by_chars={pages_by_chars}, by_lines={pages_by_lines}")
        logger.debug(f"Final estimate: {estimated_pages} pages")
        
        # Ensure reasonable bounds
        return max(1, min(estimated_pages, 1000))
    
    @staticmethod
    def _process_pdf(file_content: bytes, warnings: List[str]) -> Tuple[str, int]:
        """Legacy PDF processing method - redirect to multi-method"""
        return SafeDocumentProcessor._process_pdf_multi_method(file_content, warnings)
    
    @staticmethod
    def _process_pdf_enhanced(file_content: bytes, warnings: List[str]) -> Tuple[str, int]:
        """Legacy enhanced PDF processing - redirect to multi-method"""
        return SafeDocumentProcessor._process_pdf_multi_method(file_content, warnings)
    
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
                    
                    text_content = []
                    for element in elements:
                        if hasattr(element, 'text') and element.text:
                            text_content.append(element.text)
                    
                    os.unlink(temp_file_path)
                    
                    final_text = "\n\n".join(text_content)
                    pages_estimated = SafeDocumentProcessor._estimate_pages_from_text(final_text)
                    return final_text, pages_estimated
                    
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
                text_content = []
                
                # Extract text from paragraphs
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        text_content.append(paragraph.text)
                
                # Also extract text from tables
                for table in doc.tables:
                    for row in table.rows:
                        row_text = []
                        for cell in row.cells:
                            if cell.text.strip():
                                row_text.append(cell.text.strip())
                        if row_text:
                            text_content.append("\t".join(row_text))
                
                final_text = "\n\n".join(text_content)
                pages_estimated = SafeDocumentProcessor._estimate_pages_from_text(final_text)
                return final_text, pages_estimated
                
            except ImportError:
                warnings.append("python-docx not available. Install with: pip install python-docx")
                return "DOCX processing not available", 0
            except Exception as e:
                warnings.append(f"Error processing DOCX: {str(e)}")
                return "Error processing DOCX", 0
                
        except Exception as e:
            warnings.append(f"Error processing DOCX: {str(e)}")
            return "Error processing DOCX", 0
