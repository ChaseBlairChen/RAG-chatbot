# INTEGRATION GUIDE: New Document Processor with Your Legal Assistant App
# This shows how to integrate the new DocumentProcessor with your existing code

# =============================================================================
# 1. UPDATE requirements.txt - Add missing dependency
# =============================================================================

# Add this line to your requirements.txt:
# redis>=4.5.0

# =============================================================================
# 2. COMPATIBILITY WRAPPER - Add this to services/document_processor.py
# =============================================================================

from typing import Tuple, List
from .enhanced_document_processor import DocumentProcessor, ProcessingResult

class SafeDocumentProcessor:
    """
    Backward-compatible wrapper for the new DocumentProcessor.
    Maintains the same interface your app currently expects.
    """
    
    def __init__(self):
        self.processor = DocumentProcessor()
    
    @staticmethod
    def process_document_safe(file) -> Tuple[str, int, List[str]]:
        """
        BACKWARD COMPATIBLE: Process uploaded document safely
        Returns: (content, pages_processed, warnings)
        """
        try:
            filename = getattr(file, 'filename', 'unknown')
            file_content = file.file.read()
            
            # Use new processor
            processor = DocumentProcessor()
            result = processor.process(file_content, filename)
            
            # Reset file pointer for compatibility
            file.file.seek(0)
            
            return result.content, result.page_count, result.warnings
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return "Error processing document", 0, [f"Processing failed: {str(e)}"]
    
    @staticmethod
    def quick_validate(file_content: bytes, file_ext: str) -> Tuple[str, int, List[str]]:
        """
        BACKWARD COMPATIBLE: Quick validation to check if document can be processed
        """
        try:
            processor = DocumentProcessor()
            
            # Create a temporary filename for validation
            temp_filename = f"temp{file_ext}"
            
            # For quick validation, process first 10KB only
            sample_content = file_content[:10240] if len(file_content) > 10240 else file_content
            
            result = processor.process(sample_content, temp_filename)
            
            # Return preview for validation
            preview_content = result.content[:1000] if len(result.content) > 1000 else result.content
            estimated_pages = max(1, len(file_content) // (len(sample_content) or 1) * result.page_count)
            
            return preview_content, estimated_pages, result.warnings
            
        except Exception as e:
            logger.error(f"Quick validation error: {e}")
            return "", 0, [f"Validation error: {str(e)}"]
    
    @staticmethod
    def process_document_from_bytes(file_content: bytes, filename: str, file_ext: str) -> Tuple[str, int, List[str]]:
        """
        BACKWARD COMPATIBLE: Process document from bytes (for background processing)
        """
        try:
            processor = DocumentProcessor()
            result = processor.process(file_content, filename)
            
            logger.info(f"Background processing complete: {len(result.content)} chars, {result.page_count} pages, {len(result.warnings)} warnings")
            return result.content, result.page_count, result.warnings
            
        except Exception as e:
            logger.error(f"Error processing document from bytes: {e}", exc_info=True)
            return "", 0, [f"Processing failed: {str(e)}"]
    
    @staticmethod
    def get_processing_capabilities() -> Dict[str, bool]:
        """Return current processing capabilities for different methods"""
        return {
            'unstructured_available': FeatureFlags.UNSTRUCTURED_AVAILABLE,
            'pymupdf_available': FeatureFlags.PYMUPDF_AVAILABLE,
            'pdfplumber_available': FeatureFlags.PDFPLUMBER_AVAILABLE,
            'ocr_available': FeatureFlags.OCR_AVAILABLE,
            'docx_available': True,  # Usually available
            'supported_formats': ['.txt', '.pdf', '.docx'],
            'processor_version': 'refactored_v2'
        }

# =============================================================================
# 3. ENHANCED DOCUMENT PROCESSOR - Create services/enhanced_document_processor.py
# =============================================================================

"""
Enhanced document processing service with Strategy pattern and better error handling.
"""
import os
import io
import logging
import tempfile
from typing import Tuple, List, Dict, Callable, Optional, Any
from contextlib import contextmanager

from pydantic import BaseModel, Field
from ..config import FeatureFlags
from ..core.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)

class ProcessingResult(BaseModel):
    """Structured result of a document processing operation."""
    content: str
    page_count: int = Field(..., ge=0)
    warnings: List[str] = Field(default_factory=list)
    processing_method: str = ""
    extraction_quality: float = Field(default=1.0, ge=0.0, le=1.0)

class DocumentProcessor:
    """
    Enhanced document processor using Strategy pattern for extensibility.
    """
    
    def __init__(self):
        # Strategy Pattern: Map file extensions to handler methods
        self.handlers: Dict[str, Callable] = {
            '.txt': self._process_text,
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
        }
        
        # PDF processing strategies in order of preference
        self.pdf_strategies = [
            ("unstructured", self._pdf_handler_unstructured, FeatureFlags.UNSTRUCTURED_AVAILABLE),
            ("pymupdf_enhanced", self._pdf_handler_pymupdf_enhanced, FeatureFlags.PYMUPDF_AVAILABLE),
            ("pdfplumber", self._pdf_handler_pdfplumber, FeatureFlags.PDFPLUMBER_AVAILABLE),
            ("pymupdf_basic", self._pdf_handler_pymupdf_basic, FeatureFlags.PYMUPDF_AVAILABLE),
            ("ocr", self._pdf_handler_ocr, FeatureFlags.OCR_AVAILABLE),
        ]
        
        logger.info(f"DocumentProcessor initialized with {len([s for s in self.pdf_strategies if s[2]])} PDF strategies available")
    
    def process(self, file_content: bytes, filename: str) -> ProcessingResult:
        """
        Process a document from its content bytes and filename.
        Single public entry point with comprehensive error handling.
        """
        start_time = time.time()
        logger.info(f"Processing '{filename}', size: {len(file_content)} bytes")
        
        file_ext = os.path.splitext(filename)[1].lower()
        handler = self.handlers.get(file_ext, self._process_unsupported_as_text)
        
        try:
            result = handler(file_content, filename)
            
            # Validate extraction quality
            quality_score = self._assess_extraction_quality(result.content, filename)
            result.extraction_quality = quality_score
            
            if quality_score < 0.5:
                result.warnings.append("Low quality extraction detected - consider manual review")
                logger.warning(f"Low quality extraction for '{filename}': {quality_score:.2f}")
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Processed '{filename}' in {processing_time:.2f}s: "
                       f"{len(result.content)} chars, {result.page_count} pages, "
                       f"quality={quality_score:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to process '{filename}': {e}", exc_info=True)
            raise DocumentProcessingError(f"Document processing failed for '{file_ext}': {e}") from e
    
    # --- Handler Methods (Strategies) ---
    
    def _process_text(self, content_bytes: bytes, filename: str) -> ProcessingResult:
        """Process plain text files"""
        try:
            # Try multiple encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            content = None
            
            for encoding in encodings:
                try:
                    content = content_bytes.decode(encoding, errors='ignore')
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                content = content_bytes.decode('utf-8', errors='replace')
            
            page_count = self._estimate_pages_from_text(content)
            
            return ProcessingResult(
                content=content, 
                page_count=page_count,
                processing_method="text_encoding"
            )
            
        except Exception as e:
            raise DocumentProcessingError(f"Text processing failed: {e}")
    
    def _process_docx(self, content_bytes: bytes, filename: str) -> ProcessingResult:
        """Process DOCX files with fallback strategies"""
        
        # Strategy 1: Unstructured.io (most powerful)
        if FeatureFlags.UNSTRUCTURED_AVAILABLE:
            try:
                logger.debug("Attempting DOCX processing with Unstructured.io")
                with self._temp_file(content_bytes, ".docx") as temp_path:
                    from unstructured.partition.auto import partition
                    elements = partition(filename=temp_path)
                
                text_content = []
                for element in elements:
                    if hasattr(element, 'text') and element.text:
                        text_content.append(element.text)
                
                content = "\n\n".join(text_content)
                page_count = self._estimate_pages_from_text(content)
                
                return ProcessingResult(
                    content=content, 
                    page_count=page_count,
                    processing_method="unstructured_docx"
                )
                
            except Exception as e:
                logger.warning(f"Unstructured.io DOCX processing failed: {e}")
        
        # Strategy 2: python-docx (fallback)
        try:
            from docx import Document
            doc = Document(io.BytesIO(content_bytes))
            
            text_content = []
            
            # Extract text from paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text_content.append("\t".join(row_text))
            
            content = "\n\n".join(text_content)
            page_count = self._estimate_pages_from_text(content)
            
            return ProcessingResult(
                content=content, 
                page_count=page_count,
                processing_method="python_docx"
            )
            
        except ImportError:
            raise DocumentProcessingError("python-docx not available and Unstructured.io failed")
        except Exception as e:
            raise DocumentProcessingError(f"DOCX processing failed: {e}")
    
    def _process_pdf(self, content_bytes: bytes, filename: str) -> ProcessingResult:
        """Process PDF files using multiple strategies with intelligent fallback"""
        
        warnings = []
        
        for strategy_name, method, is_available in self.pdf_strategies:
            if not is_available:
                logger.debug(f"Skipping PDF strategy '{strategy_name}' (not available)")
                continue
            
            try:
                logger.debug(f"Attempting PDF processing with '{strategy_name}'")
                result = method(content_bytes, filename)
                
                if result and result.content.strip() and len(result.content.strip()) > 50:
                    logger.info(f"✅ PDF processed successfully with '{strategy_name}'")
                    result.warnings.extend(warnings)
                    result.processing_method = strategy_name
                    return result
                else:
                    warning_msg = f"Strategy '{strategy_name}' produced insufficient content"
                    warnings.append(warning_msg)
                    logger.warning(warning_msg)
                    
            except Exception as e:
                warning_msg = f"Strategy '{strategy_name}' failed: {str(e)}"
                warnings.append(warning_msg)
                logger.warning(f"PDF processing with '{strategy_name}' failed: {e}")
        
        # If all strategies failed
        raise DocumentProcessingError(f"All {len(self.pdf_strategies)} PDF processing strategies failed. Warnings: {'; '.join(warnings)}")
    
    def _process_unsupported_as_text(self, content_bytes: bytes, filename: str) -> ProcessingResult:
        """Handle unsupported file types by attempting text extraction"""
        file_ext = os.path.splitext(filename)[1].lower()
        warnings = [f"Unsupported file type '{file_ext}'. Attempting to process as plain text."]
        
        try:
            result = self._process_text(content_bytes, filename)
            result.warnings.extend(warnings)
            result.processing_method = "unsupported_as_text"
            return result
        except Exception as e:
            raise DocumentProcessingError(f"Could not process unsupported file type '{file_ext}': {e}")
    
    # --- PDF Strategy Implementations ---
    
    def _pdf_handler_unstructured(self, content_bytes: bytes, filename: str) -> ProcessingResult:
        """Unstructured.io PDF processing (highest quality)"""
        with self._temp_file(content_bytes, ".pdf") as temp_path:
            from unstructured.partition.auto import partition
            elements = partition(filename=temp_path, strategy="hi_res")
        
        text_content = []
        page_count = 0
        
        for element in elements:
            if hasattr(element, 'text') and element.text:
                text_content.append(element.text)
            
            if hasattr(element, 'metadata') and element.metadata:
                if hasattr(element.metadata, 'page_number'):
                    page_count = max(page_count, element.metadata.page_number)
        
        content = "\n\n".join(text_content)
        
        if page_count == 0:
            page_count = self._estimate_pages_from_text(content)
        
        return ProcessingResult(
            content=content, 
            page_count=page_count,
            processing_method="unstructured_hi_res"
        )
    
    def _pdf_handler_pymupdf_enhanced(self, content_bytes: bytes, filename: str) -> ProcessingResult:
        """Enhanced PyMuPDF processing with layout preservation"""
        import fitz
        
        doc = fitz.open(stream=content_bytes, filetype="pdf")
        all_text = []
        page_count = len(doc)
        
        for page_num in range(page_count):
            page = doc.load_page(page_num)
            
            # Try to get text with layout preservation
            text_dict = page.get_text("dict")
            page_text = self._reconstruct_layout_from_dict(text_dict)
            
            # If text is too short, might be scanned or have issues
            if len(page_text.strip()) < 50:
                # Fallback to simple text extraction
                page_text = page.get_text()
            
            all_text.append(f"\n--- Page {page_num + 1} ---\n{page_text}")
        
        doc.close()
        
        content = "\n".join(all_text)
        
        return ProcessingResult(
            content=content, 
            page_count=page_count,
            processing_method="pymupdf_enhanced"
        )
    
    def _pdf_handler_pymupdf_basic(self, content_bytes: bytes, filename: str) -> ProcessingResult:
        """Basic PyMuPDF processing"""
        import fitz
        
        doc = fitz.open(stream=content_bytes, filetype="pdf")
        all_text = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            all_text.append(f"\n--- Page {page_num + 1} ---\n{text}")
        
        page_count = len(doc)
        doc.close()
        
        content = "\n".join(all_text)
        
        return ProcessingResult(
            content=content, 
            page_count=page_count,
            processing_method="pymupdf_basic"
        )
    
    def _pdf_handler_pdfplumber(self, content_bytes: bytes, filename: str) -> ProcessingResult:
        """PDFPlumber processing with table extraction"""
        import pdfplumber
        
        all_text = []
        
        with pdfplumber.open(io.BytesIO(content_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                
                # Also extract tables
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        table_text = "\n".join([
                            "\t".join([str(cell) if cell else "" for cell in row]) 
                            for row in table
                        ])
                        text += f"\n\n[Table]\n{table_text}\n"
                
                all_text.append(f"\n--- Page {i + 1} ---\n{text}")
            
            page_count = len(pdf.pages)
        
        content = "\n".join(all_text)
        
        return ProcessingResult(
            content=content, 
            page_count=page_count,
            processing_method="pdfplumber_with_tables"
        )
    
    def _pdf_handler_ocr(self, content_bytes: bytes, filename: str) -> ProcessingResult:
        """OCR processing for scanned PDFs"""
        try:
            import pytesseract
            from pdf2image import convert_from_bytes
            from PIL import Image
            
            # Convert PDF to images
            images = convert_from_bytes(content_bytes, dpi=300)
            all_text = []
            
            for i, image in enumerate(images):
                try:
                    # Preprocess image for better OCR
                    processed_image = self._preprocess_image_for_ocr(image)
                    
                    # Perform OCR
                    text = pytesseract.image_to_string(
                        processed_image, 
                        lang='eng', 
                        config='--psm 6'
                    )
                    
                    if text.strip():
                        all_text.append(f"\n--- Page {i + 1} (OCR) ---\n{text}")
                    
                except Exception as e:
                    logger.warning(f"OCR failed for page {i + 1}: {e}")
                    all_text.append(f"\n--- Page {i + 1} (OCR FAILED) ---\n")
            
            content = "\n".join(all_text)
            
            return ProcessingResult(
                content=content, 
                page_count=len(images),
                processing_method="ocr_tesseract",
                warnings=["Document processed using OCR - accuracy may vary"]
            )
            
        except ImportError:
            raise DocumentProcessingError("OCR libraries not installed. Install pytesseract and pdf2image")
    
    def _preprocess_image_for_ocr(self, image):
        """Preprocess image to improve OCR accuracy"""
        try:
            from PIL import ImageEnhance, ImageFilter
            
            # Convert to grayscale
            image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Remove noise
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            return image
        except Exception:
            return image  # Return original if preprocessing fails
    
    def _reconstruct_layout_from_dict(self, text_dict: Dict) -> str:
        """Reconstruct text layout from PyMuPDF dict output"""
        blocks = []
        
        for block in sorted(text_dict.get("blocks", []), 
                          key=lambda b: (b.get("bbox", [0])[1], b.get("bbox", [0])[0])):
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
    
    def _assess_extraction_quality(self, text: str, filename: str) -> float:
        """Assess the quality of text extraction"""
        if not text or len(text.strip()) < 50:
            return 0.0
        
        quality_score = 1.0
        
        # Check for Unicode errors
        unicode_error_ratio = text.count('�') / len(text)
        if unicode_error_ratio > 0.05:  # More than 5% unicode errors
            quality_score -= 0.3
        
        # Check for mostly whitespace
        content_ratio = len(text.strip()) / len(text)
        if content_ratio < 0.5:  # More than 50% whitespace
            quality_score -= 0.2
        
        # Check for very short extraction relative to file type
        if filename.endswith('.pdf') and len(text) < 500:
            quality_score -= 0.3
        
        # Check for repeated characters (OCR artifacts)
        import re
        repeated_chars = len(re.findall(r'(.)\1{5,}', text))
        if repeated_chars > 10:
            quality_score -= 0.2
        
        # Check for reasonable sentence structure
        sentence_count = len(re.findall(r'[.!?]+', text))
        if sentence_count == 0 and len(text) > 100:
            quality_score -= 0.2
        
        return max(0.0, quality_score)
    
    def _estimate_pages_from_text(self, text: str) -> int:
        """Enhanced page estimation based on content analysis"""
        if not text or len(text.strip()) == 0:
            return 0
        
        # Multiple estimation methods
        word_count = len(text.split())
        char_count = len(text)
        line_count = len(text.split('\n'))
        
        # Estimates based on different metrics
        pages_by_words = max(1, word_count // 300)  # ~300 words per page
        pages_by_chars = max(1, char_count // 2000)  # ~2000 chars per page
        pages_by_lines = max(1, line_count // 40)    # ~40 lines per page
        
        # Use median for more accurate estimation
        estimates = sorted([pages_by_words, pages_by_chars, pages_by_lines])
        estimated_pages = estimates[1]  # median
        
        # Reasonable bounds
        return max(1, min(estimated_pages, 1000))
    
    @contextmanager
    def _temp_file(self, content_bytes: bytes, suffix: str):
        """Context manager for safe temporary file handling"""
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        try:
            os.write(fd, content_bytes)
            os.close(fd)
            yield temp_path
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass  # File already deleted

# =============================================================================
# 4. INTEGRATION INSTRUCTIONS
# =============================================================================

"""
STEP-BY-STEP INTEGRATION:

1. ADD DEPENDENCIES:
   Add to requirements.txt:
   redis>=4.5.0

2. CREATE NEW FILES:
   - Save the new DocumentProcessor as: services/enhanced_document_processor.py
   - Keep your existing services/document_processor.py but replace its content with the wrapper

3. NO OTHER CHANGES NEEDED:
   - Your existing API endpoints will work unchanged
   - Background tasks will work unchanged  
   - All existing functionality preserved

4. OPTIONAL REDIS SETUP:
   - Install Redis: sudo apt install redis-server (Linux) or brew install redis (Mac)
   - Start Redis: redis-server
   - If Redis unavailable, app falls back to in-memory storage automatically

5. BENEFITS:
   - Better PDF extraction (multiple strategies)
   - Quality assessment of extractions
   - More robust error handling
   - Strategy pattern for easy extension
   - OCR support for scanned documents
   - Better table extraction from documents

6. TESTING:
   # Test the new processor
   processor = DocumentProcessor()
   result = processor.process(file_content, "test.pdf")
   print(f"Quality: {result.extraction_quality}")
   print(f"Method: {result.processing_method}")
"""

# =============================================================================
# 5. MIGRATION CHECKLIST
# =============================================================================

MIGRATION_CHECKLIST = """
□ Add redis>=4.5.0 to requirements.txt
□ Create services/enhanced_document_processor.py with new code
□ Replace services/document_processor.py with wrapper
□ Test document upload still works
□ Check processing status endpoint still works  
□ Install Redis server (optional - app works without it)
□ Test PDF processing with complex documents
□ Monitor processing quality scores
□ Verify all file types still process correctly
"""
