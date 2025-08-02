# Legal Assistant API - Developer Guide & Troubleshooting

## 🏗️ Architecture Overview

The application follows a clean modular architecture:

```
legal_assistant/
├── api/           # API endpoints
├── core/          # Core functionality (auth, dependencies, exceptions)
├── models/        # Data models (Pydantic)
├── processors/    # Business logic processors
├── services/      # Service layer (AI, documents, RAG)
├── storage/       # State management
└── utils/         # Utility functions
```

## 🔧 Common Issues & Which Files to Edit

### 1. **Document Upload/Processing Issues**

**Problem**: Users report documents aren't being "read" properly or content isn't found

**Files to edit**:
- `services/document_processor.py` - Main document processing logic
- `services/container_manager.py` - Document storage and chunking
- `utils/text_processing.py` - Text chunking algorithms

**Current issues & solutions**:

1. **Poor PDF extraction**:
   - Current: Basic PyMuPDF/pdfplumber extraction
   - Solution: Add OCR support for scanned PDFs
   ```python
   # In services/document_processor.py, add:
   import pytesseract
   from PIL import Image
   
   def _process_pdf_with_ocr(self, file_content: bytes, warnings: List[str]) -> Tuple[str, int]:
       # Add OCR processing for scanned PDFs
   ```

2. **Chunking breaks context**:
   - Current: Fixed-size chunking (1500 chars)
   - Solution: Implement sliding window with better overlap
   ```python
   # In services/container_manager.py, modify add_document_to_container:
   text_splitter = RecursiveCharacterTextSplitter(
       chunk_size=1000,  # Smaller chunks
       chunk_overlap=200,  # More overlap
       separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", " ", ""]
   )
   ```

3. **Metadata not properly stored**:
   - Current: Limited metadata extraction
   - Solution: Enhanced metadata with better indexing

### 2. **Search/Retrieval Issues**

**Problem**: Relevant content not found even when it exists

**Files to edit**:
- `services/rag_service.py` - RAG implementation
- `services/container_manager.py` - Search logic
- `processors/query_processor.py` - Query processing

**Solutions**:

1. **Improve search accuracy**:
   ```python
   # In services/container_manager.py, enhance search:
   def enhanced_search_user_container(self, user_id: str, query: str, ...):
       # Add hybrid search (keyword + semantic)
       keyword_results = self._keyword_search(query)
       semantic_results = user_db.similarity_search_with_score(query, k=k*2)
       
       # Rerank results using cross-encoder
       from sentence_transformers import CrossEncoder
       reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
       
       # Combine and rerank
   ```

2. **Better query expansion**:
   ```python
   # In services/rag_service.py:
   def expand_query(query: str) -> List[str]:
       # Add synonyms, related terms
       # Use WordNet or domain-specific expansions
   ```

### 3. **Citation/Source Attribution Issues**

**Problem**: Incorrect or missing citations

**Files to edit**:
- `utils/formatting.py` - Format context for LLM
- `processors/query_processor.py` - Response generation

**Solution**:
```python
# In utils/formatting.py, enhance source tracking:
def format_context_for_llm(results_with_scores: List[Tuple], max_length: int = 3000):
    # Add unique IDs to each chunk
    # Include page numbers, sections, and exact locations
    
    context_part = f"""
    [SOURCE_ID: {unique_id}]
    [DOCUMENT: {display_source}]
    [PAGE: {page}]
    [SECTION: {section}]
    [RELEVANCE: {score:.2f}]
    
    {content}
    """
```

### 4. **AI Response Quality Issues**

**Problem**: AI not following instructions or hallucinating

**Files to edit**:
- `processors/query_processor.py` - Prompt engineering
- `services/ai_service.py` - AI model configuration

**Solution**: Enhanced prompts (already partially implemented)

### 5. **Performance Issues**

**Problem**: Slow document processing or retrieval

**Files to edit**:
- `services/container_manager.py` - Batch processing
- `config.py` - Adjust batch sizes and limits

## 📚 Enhanced Document Processing Solution

Here's a comprehensive solution to improve document reading:

### Step 1: Install Additional Dependencies

Add to `requirements.txt`:
```
# OCR support
pytesseract==0.3.10
pdf2image==1.16.3
Pillow==10.1.0

# Better text extraction
unstructured[pdf]==0.10.30
langchain-unstructured==0.1.0

# Reranking
sentence-transformers==2.6.1
rank-bm25==0.2.2
```

### Step 2: Create Enhanced Document Processor

Create `services/enhanced_document_processor.py`:

```python
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import fitz  # PyMuPDF
from typing import Tuple, List
import re

class EnhancedDocumentProcessor:
    @staticmethod
    def process_pdf_advanced(file_content: bytes) -> Tuple[str, int, List[str]]:
        """Enhanced PDF processing with OCR fallback"""
        warnings = []
        all_text = []
        
        try:
            # First try: PyMuPDF with layout preservation
            doc = fitz.open(stream=file_content, filetype="pdf")
            
            for page_num, page in enumerate(doc):
                # Extract text with layout
                text = page.get_text("dict")
                page_text = EnhancedDocumentProcessor._reconstruct_layout(text)
                
                # If text is too short, might be scanned
                if len(page_text.strip()) < 50:
                    warnings.append(f"Page {page_num + 1} appears to be scanned, using OCR")
                    # Convert to image and OCR
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    ocr_text = pytesseract.image_to_string(img)
                    page_text = ocr_text if ocr_text else page_text
                
                all_text.append(f"--- Page {page_num + 1} ---\n{page_text}")
            
            doc.close()
            
        except Exception as e:
            warnings.append(f"Advanced extraction failed: {e}")
            # Fallback to basic extraction
            
        return "\n\n".join(all_text), len(all_text), warnings
    
    @staticmethod
    def _reconstruct_layout(text_dict):
        """Reconstruct text layout from PyMuPDF dict"""
        blocks = []
        for block in text_dict.get("blocks", []):
            if block["type"] == 0:  # Text block
                block_text = []
                for line in block.get("lines", []):
                    line_text = []
                    for span in line.get("spans", []):
                        line_text.append(span.get("text", ""))
                    block_text.append(" ".join(line_text))
                blocks.append("\n".join(block_text))
        return "\n\n".join(blocks)
```

### Step 3: Implement Hybrid Search

Create `services/hybrid_search.py`:

```python
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np

class HybridSearch:
    def __init__(self):
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def search(self, query: str, documents: List[Document], k: int = 10):
        # Step 1: BM25 keyword search
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Step 2: Semantic search (existing)
        # ... your existing semantic search ...
        
        # Step 3: Combine scores
        combined_scores = 0.3 * bm25_scores + 0.7 * semantic_scores
        
        # Step 4: Rerank top results
        top_indices = np.argsort(combined_scores)[-k*2:][::-1]
        pairs = [[query, documents[i].page_content] for i in top_indices]
        rerank_scores = self.reranker.predict(pairs)
        
        # Return reranked results
        reranked_indices = np.argsort(rerank_scores)[::-1][:k]
        return [(documents[top_indices[i]], rerank_scores[i]) for i in reranked_indices]
```

### Step 4: Better Chunking Strategy

Update `services/container_manager.py`:

```python
def intelligent_chunking(self, text: str, doc_type: str = "general") -> List[str]:
    """Intelligent chunking based on document structure"""
    
    if doc_type == "legal":
        # Legal document patterns
        section_patterns = [
            r'\n\s*(?:Section|SECTION|Article|ARTICLE)\s+\d+',
            r'\n\s*\d+\.\d+\s+[A-Z]',  # Numbered sections
            r'\n\s*\([a-z]\)\s+',  # Subsections
        ]
        
        chunks = []
        for pattern in section_patterns:
            sections = re.split(pattern, text)
            if len(sections) > 1:
                # Found sections, chunk by section
                for section in sections:
                    if len(section) > 2000:
                        # Section too large, sub-chunk
                        sub_chunks = self.sliding_window_chunk(section, 1000, 200)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(section)
                return chunks
    
    # Fallback to sliding window
    return self.sliding_window_chunk(text, 1000, 200)

def sliding_window_chunk(self, text: str, chunk_size: int, overlap: int) -> List[str]:
    """Sliding window chunking with sentence boundaries"""
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        if current_size + sentence_size > chunk_size and current_chunk:
            # Save current chunk
            chunks.append('. '.join(current_chunk) + '.')
            
            # Start new chunk with overlap
            overlap_sentences = []
            overlap_size = 0
            for s in reversed(current_chunk):
                if overlap_size + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_size += len(s)
                else:
                    break
            
            current_chunk = overlap_sentences + [sentence]
            current_size = overlap_size + sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    
    return chunks
```

## 🚀 Quick Fixes for Document Reading Issues

### 1. **Immediate Fix** - Increase chunk overlap:
```python
# In services/container_manager.py
chunk_overlap=500  # Increase from 300
```

### 2. **Add Document Validation**:
```python
# In services/document_processor.py
def validate_extraction(text: str, filename: str) -> bool:
    """Validate if extraction was successful"""
    if len(text) < 100:
        logger.warning(f"Extraction too short for {filename}")
        return False
    
    # Check for common extraction failures
    if text.count('�') > 10:  # Unicode errors
        return False
    
    return True
```

### 3. **Improve Metadata Storage**:
```python
# In services/container_manager.py
doc_metadata['content_hash'] = hashlib.md5(chunk.encode()).hexdigest()
doc_metadata['chunk_position'] = f"{start_idx}/{total_chunks}"
doc_metadata['previous_chunk_id'] = previous_chunk_id
doc_metadata['next_chunk_id'] = next_chunk_id
```

## 📊 Monitoring & Debugging

### Add Logging for Document Processing:
```python
# In services/document_processor.py
logger.info(f"Processing {filename}: {file_size} bytes")
logger.info(f"Extracted {len(content)} characters, {pages} pages")
logger.info(f"Text sample: {content[:200]}...")
```

### Add Search Debugging:
```python
# In services/container_manager.py
logger.info(f"Search query: '{query}'")
logger.info(f"Found {len(results)} results")
for i, (doc, score) in enumerate(results[:3]):
    logger.info(f"Result {i}: Score={score:.3f}, Content={doc.page_content[:100]}...")
```

## 🏥 Health Checks

Add to `api/routers/admin.py`:
```python
@router.get("/document-processing-health")
async def check_processing_health():
    """Check document processing capabilities"""
    return {
        "pdf_processors": {
            "pymupdf": FeatureFlags.PYMUPDF_AVAILABLE,
            "pdfplumber": FeatureFlags.PDFPLUMBER_AVAILABLE,
            "unstructured": FeatureFlags.UNSTRUCTURED_AVAILABLE,
            "ocr": check_ocr_available(),
        },
        "chunking_methods": [
            "semantic_bert",
            "sliding_window",
            "recursive_character",
            "legal_structure"
        ],
        "search_methods": [
            "semantic",
            "keyword_bm25",
            "hybrid",
            "reranked"
        ]
    }
```

## 🎯 Testing Document Processing

Create `tests/test_document_processing.py`:
```python
def test_document_extraction():
    """Test document extraction quality"""
    test_files = ["sample.pdf", "scanned.pdf", "complex_layout.pdf"]
    
    for file in test_files:
        content, pages, warnings = processor.process_document(file)
        
        assert len(content) > 100, f"Extraction failed for {file}"
        assert pages > 0, f"No pages detected in {file}"
        
        # Check for common issues
        assert content.count('�') < 10, "Unicode errors in extraction"
        assert not content.startswith("Error"), "Error in extraction"
```

## 💡 Best Practices

1. **Always validate extracted content** before storing
2. **Log extraction metrics** for debugging
3. **Use multiple extraction methods** with fallbacks
4. **Store extraction metadata** for troubleshooting
5. **Implement content deduplication** to avoid redundancy
6. **Use hybrid search** for better retrieval
7. **Monitor chunk sizes** and adjust based on document types

## 🆘 Emergency Fixes

If users can't find content that exists:

1. **Reduce chunk size** to 1000 characters
2. **Increase chunk overlap** to 500 characters  
3. **Clear and rebuild** the vector database
4. **Enable debug logging** for search queries
5. **Use multiple search strategies** in parallel

Front end 
frontend/
├── src/
│   ├── App.tsx (← you already have this!)
│   ├── main.tsx
│   ├── index.css
│   ├── components/
│   │   ├── auth/
│   │   │   └── LoginScreen.tsx
│   │   ├── layout/
│   │   │   ├── AppHeader.tsx
│   │   │   ├── BackendWarning.tsx
│   │   │   ├── TabNavigation.tsx
│   │   │   └── DisconnectedView.tsx
│   │   ├── chat/
│   │   │   ├── ChatTab.tsx
│   │   │   ├── MessageList.tsx
│   │   │   ├── MessageItem.tsx
│   │   │   └── ChatInput.tsx
│   │   ├── upload/
│   │   │   ├── UploadTab.tsx
│   │   │   ├── UploadZone.tsx
│   │   │   ├── UploadQueue.tsx
│   │   │   ├── UploadStatus.tsx
│   │   │   └── UploadResults.tsx
│   │   ├── documents/
│   │   │   ├── DocumentsTab.tsx
│   │   │   └── DocumentItem.tsx
│   │   ├── analysis/
│   │   │   ├── AnalysisTab.tsx
│   │   │   ├── AnalysisToolCard.tsx
│   │   │   └── DocumentSelector.tsx
│   │   ├── results/
│   │   │   ├── ResultsTab.tsx
│   │   │   └── AnalysisResult.tsx
│   │   └── common/
│   │       ├── EmptyState.tsx
│   │       └── LoadingSpinner.tsx
│   ├── contexts/
│   │   ├── AuthContext.tsx
│   │   └── BackendContext.tsx
│   ├── hooks/
│   │   ├── useChat.ts
│   │   ├── useDocuments.ts
│   │   └── useAnalysis.ts
│   ├── services/
│   │   └── api.ts
│   ├── types/
│   │   └── index.ts
│   └── utils/
│       ├── constants.ts
│       ├── markdown.ts
│       ├── fileValidation.ts
│       └── helpers.ts
├── package.json
├── vite.config.ts
└── tsconfig.json



