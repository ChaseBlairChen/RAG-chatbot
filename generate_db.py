import os
import shutil
import glob
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import fitz  # PyMuPDF - better PDF extraction
import re

# Use home directory to avoid permission issues
CHROMA_PATH = os.path.join(os.getcwd(), "my_chroma_db")
DOCUMENTS_FOLDER = "documents"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    if not documents:
        print("No documents found to process.")
        return
    
    # Quality check before chunking
    documents = validate_and_clean_documents(documents)
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents() -> List[Document]:
    """Load all PDF and Markdown documents with better PDF handling"""
    if not os.path.exists(DOCUMENTS_FOLDER):
        raise FileNotFoundError(f"Documents folder '{DOCUMENTS_FOLDER}' not found.")
    
    all_documents = []
    
    # Get all PDF files
    pdf_files = glob.glob(os.path.join(DOCUMENTS_FOLDER, "*.pdf"))
    pdf_files.extend(glob.glob(os.path.join(DOCUMENTS_FOLDER, "**/*.pdf"), recursive=True))
    
    # Get all Markdown files
    md_files = glob.glob(os.path.join(DOCUMENTS_FOLDER, "*.md"))
    md_files.extend(glob.glob(os.path.join(DOCUMENTS_FOLDER, "**/*.md"), recursive=True))
    
    print(f"Found {len(pdf_files)} PDF files and {len(md_files)} Markdown files")
    
    # Load PDF files with enhanced PyMuPDF processing
    for pdf_file in pdf_files:
        try:
            print(f"Loading PDF: {pdf_file}")
            docs = load_pdf_with_enhanced_extraction(pdf_file)
            all_documents.extend(docs)
            print(f"Successfully loaded {len(docs)} sections from {pdf_file}")
        except Exception as e:
            print(f"Error loading PDF {pdf_file}: {e}")
            # Fallback to basic extraction
            try:
                print(f"Trying basic extraction for {pdf_file}")
                docs = load_pdf_with_pymupdf_basic(pdf_file)
                all_documents.extend(docs)
                print(f"Basic extraction successful: {len(docs)} pages")
            except Exception as e2:
                print(f"Both extraction methods failed for {pdf_file}: {e2}")
    
    # Load Markdown files
    for md_file in md_files:
        try:
            print(f"Loading Markdown: {md_file}")
            loader = TextLoader(md_file, encoding='utf-8')
            documents = loader.load()
            
            for doc in documents:
                doc.metadata["source"] = md_file
                doc.metadata["file_type"] = "markdown"
            
            all_documents.extend(documents)
            print(f"Successfully loaded {md_file}")
        except Exception as e:
            print(f"Error loading Markdown {md_file}: {e}")
    
    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents

def load_pdf_with_enhanced_extraction(pdf_path: str) -> List[Document]:
    """Enhanced PDF extraction that creates logical sections"""
    documents = []
    
    try:
        pdf_document = fitz.open(pdf_path)
        file_name = os.path.basename(pdf_path)
        
        # Extract text from all pages first
        all_text = ""
        page_breaks = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text()
            
            if page_text.strip():
                page_breaks.append(len(all_text))
                all_text += page_text + "\n\n--- PAGE BREAK ---\n\n"
        
        pdf_document.close()
        
        # Clean the text
        all_text = clean_pdf_text_enhanced(all_text)
        
        # Split into logical sections
        sections = create_logical_sections(all_text, file_name, page_breaks)
        
        for section in sections:
            if len(section['content'].strip()) > 100:  # Only substantial sections
                doc = Document(
                    page_content=section['content'],
                    metadata={
                        "source": pdf_path,
                        "file_name": file_name,
                        "file_type": "pdf",
                        "section_title": section['title'],
                        "page_number": section['page'],
                        "total_pages": len(pdf_document),
                        "extraction_method": "enhanced_pymupdf",
                        "section_type": section['type']
                    }
                )
                documents.append(doc)
        
    except Exception as e:
        print(f"Enhanced PDF extraction failed for {pdf_path}: {e}")
        raise e
    
    return documents

def load_pdf_with_pymupdf_basic(pdf_path: str) -> List[Document]:
    """Basic PDF extraction as fallback"""
    documents = []
    
    try:
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text = page.get_text()
            text = clean_pdf_text(text)
            
            if len(text.strip()) > 50:
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "file_type": "pdf",
                        "page_number": page_num + 1,
                        "total_pages": len(pdf_document),
                        "extraction_method": "basic_pymupdf",
                        "file_name": os.path.basename(pdf_path)
                    }
                )
                documents.append(doc)
        
        pdf_document.close()
        
    except Exception as e:
        print(f"Basic PDF extraction failed for {pdf_path}: {e}")
        raise e
    
    return documents

def create_logical_sections(text: str, file_name: str, page_breaks: List[int]) -> List[dict]:
    """Create logical sections from PDF text"""
    sections = []
    
    # Remove page break markers for processing
    clean_text = text.replace("--- PAGE BREAK ---", "")
    
    # Look for section headers (various patterns)
    header_patterns = [
        r'^([A-Z][A-Z\s&,.-]{10,}?)(?=\n)',  # ALL CAPS headers
        r'^((?:[A-Z][a-z]+ )*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:?)(?=\n)',  # Title Case headers
        r'^(\d+\.\s+[A-Z][^.]*?)(?=\n)',  # Numbered sections
        r'^([A-Z]\.\s+[A-Z][^.]*?)(?=\n)',  # Letter sections (A., B., etc.)
    ]
    
    # Find all potential headers
    headers_found = []
    for pattern in header_patterns:
        matches = re.finditer(pattern, clean_text, re.MULTILINE)
        for match in matches:
            headers_found.append({
                'title': match.group(1).strip(),
                'start': match.start(),
                'end': match.end()
            })
    
    # Sort headers by position
    headers_found.sort(key=lambda x: x['start'])
    
    # Create sections
    if not headers_found:
        # No clear headers found, split by paragraphs
        paragraphs = clean_text.split('\n\n')
        current_section = ""
        section_count = 1
        
        for para in paragraphs:
            para = para.strip()
            if len(para) > 50:
                if len(current_section) > 800:  # Start new section
                    sections.append({
                        'title': f"Section {section_count}",
                        'content': current_section.strip(),
                        'type': 'paragraph_group',
                        'page': estimate_page_number(len(current_section), page_breaks)
                    })
                    current_section = para
                    section_count += 1
                else:
                    current_section += "\n\n" + para
        
        # Add final section
        if current_section.strip():
            sections.append({
                'title': f"Section {section_count}",
                'content': current_section.strip(),
                'type': 'paragraph_group',
                'page': estimate_page_number(len(current_section), page_breaks)
            })
    
    else:
        # Use found headers to create sections
        for i, header in enumerate(headers_found):
            section_start = header['start']
            section_end = headers_found[i + 1]['start'] if i + 1 < len(headers_found) else len(clean_text)
            
            section_content = clean_text[section_start:section_end].strip()
            
            if len(section_content) > 100:
                sections.append({
                    'title': header['title'],
                    'content': section_content,
                    'type': 'header_section',
                    'page': estimate_page_number(section_start, page_breaks)
                })
    
    return sections

def estimate_page_number(text_position: int, page_breaks: List[int]) -> int:
    """Estimate page number based on text position"""
    if not page_breaks:
        return 1
    
    for i, break_pos in enumerate(page_breaks):
        if text_position <= break_pos:
            return i + 1
    
    return len(page_breaks)

def clean_pdf_text_enhanced(text: str) -> str:
    """Enhanced PDF text cleaning for structured documents"""
    # Remove page break markers temporarily
    text = text.replace("--- PAGE BREAK ---", "\n\n")
    
    # Remove excessive whitespace but preserve structure
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    # Clean up common artifacts
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip obvious page numbers and headers/footers
        if re.match(r'^Page \d+ of \d+$', line):
            continue
        if re.match(r'^Community Services Division$', line) and len(line) < 30:
            continue
        if re.match(r'^\d+$', line) and len(line) < 4:
            continue
        
        # Keep bullet points and form elements
        if line:
            cleaned_lines.append(line)
    
    # Rejoin text
    text = '\n'.join(cleaned_lines)
    
    # Normalize spacing
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def clean_pdf_text(text: str) -> str:
    """Basic PDF text cleaning (fallback)"""
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    # Remove standalone page numbers and headers/footers
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip likely page numbers (single numbers on their own line)
        if re.match(r'^\d+$', line) and len(line) < 4:
            continue
        # Skip very short lines that are likely artifacts (but keep bullet points)
        if len(line) < 3 and not re.match(r'^[•\-\*]', line):
            continue
        cleaned_lines.append(line)
    
    # Rejoin and normalize spacing
    text = '\n'.join(cleaned_lines)
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
    
    return text.strip()

def validate_and_clean_documents(documents: List[Document]) -> List[Document]:
    """Validate document quality for RAG"""
    clean_documents = []
    
    for doc in documents:
        content = doc.page_content.strip()
        
        # Skip documents that are too short or likely garbage
        if len(content) < 50:  # Lowered threshold for structured sections
            print(f"Skipping short document from {doc.metadata.get('source', 'unknown')}")
            continue
            
        # Check for garbled text (too many non-alphanumeric characters)
        alpha_ratio = sum(c.isalnum() or c.isspace() for c in content) / len(content)
        if alpha_ratio < 0.6:  # Lowered threshold for forms/structured content
            print(f"Skipping likely garbled document from {doc.metadata.get('source', 'unknown')}")
            continue
            
        # Add word count metadata for retrieval scoring
        word_count = len(content.split())
        doc.metadata["word_count"] = word_count
        
        clean_documents.append(doc)
    
    print(f"Kept {len(clean_documents)} out of {len(documents)} documents after quality check")
    return clean_documents

def split_text(documents: List[Document]) -> List[Document]:
    """Split documents into chunks optimized for RAG with better handling of structured content"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,  # Smaller chunks for structured content
        chunk_overlap=50,  # Less overlap to preserve structure
        length_function=len,
        add_start_index=True,
        # Better separators for forms and structured content
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    # Add chunk-specific metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
        
        # For structured content, try to identify key information
        content = chunk.page_content
        
        # Check if chunk contains form information
        if re.search(r'Form\s+I-\d+', content, re.IGNORECASE):
            form_match = re.search(r'Form\s+(I-\d+[A-Z]*)', content, re.IGNORECASE)
            if form_match:
                chunk.metadata["contains_form"] = form_match.group(1)
        
        # Check if chunk contains definitions
        if re.search(r'(is defined as|means|refers to)', content, re.IGNORECASE):
            chunk.metadata["content_type"] = "definition"
        
        # Check if chunk contains procedures
        if re.search(r'(must|shall|should|may|process|procedure)', content, re.IGNORECASE):
            chunk.metadata["content_type"] = "procedure"
        
        # Extract first meaningful sentence as summary
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 10]
        if sentences:
            chunk.metadata["summary"] = sentences[0][:200] + "..."
    
    # Show sample chunk information
    if chunks:
        print("\nSample chunk:")
        print(f"Content preview: {chunks[0].page_content[:200]}...")
        print(f"Metadata: {chunks[0].metadata}")
    
    return chunks

def save_to_chroma(chunks: List[Document]):
    """Save document chunks to Chroma vector store with RAG optimizations"""
    if os.path.exists(CHROMA_PATH):
        print(f"Deleting existing {CHROMA_PATH} folder")
        shutil.rmtree(CHROMA_PATH)
    
    os.makedirs(CHROMA_PATH, exist_ok=True)
    
    # Use a better embedding model for RAG
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",  # Better than MiniLM
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # Better for similarity search
    )
    
    try:
        batch_size = 25  # Smaller batches for stability with structured content
        total_chunks = len(chunks)
        
        print(f"Processing {total_chunks} chunks in batches of {batch_size}")
        
        db = None
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            print(f"Processing batch {batch_num}/{(total_chunks + batch_size - 1) // batch_size}")
            
            if db is None:
                db = Chroma.from_documents(
                    batch, 
                    embedding_function, 
                    persist_directory=CHROMA_PATH
                )
            else:
                db.add_documents(batch)
        
        if db:
            db.persist()
            print(f"Successfully saved {total_chunks} chunks to {CHROMA_PATH}")
            
            # Enhanced summary for structured RAG
            pdf_chunks = [c for c in chunks if c.metadata.get("file_type") == "pdf"]
            md_chunks = [c for c in chunks if c.metadata.get("file_type") == "markdown"]
            form_chunks = [c for c in chunks if c.metadata.get("contains_form")]
            procedure_chunks = [c for c in chunks if c.metadata.get("content_type") == "procedure"]
            
            avg_chunk_size = sum(c.metadata.get("chunk_size", 0) for c in chunks) / len(chunks)
            
            print(f"Enhanced RAG Database Summary:")
            print(f"- {len(pdf_chunks)} PDF chunks, {len(md_chunks)} Markdown chunks")
            print(f"- {len(form_chunks)} chunks contain forms")
            print(f"- {len(procedure_chunks)} chunks contain procedures")
            print(f"- Average chunk size: {avg_chunk_size:.0f} characters")
            print(f"- Total unique sources: {len(set(c.metadata.get('source') for c in chunks))}")
        
    except Exception as e:
        print(f"Error creating Chroma DB: {e}")
        raise

def verify_installation():
    """Verify that required packages are installed"""
    missing_packages = []
    
    try:
        import fitz
        print("✓ PyMuPDF is available")
    except ImportError:
        print("✗ PyMuPDF not found. Install with: pip install PyMuPDF")
        missing_packages.append("PyMuPDF")
    
    try:
        from langchain_community.document_loaders import PyPDFLoader
        print("✓ LangChain PDF loader is available")
    except ImportError:
        print("✗ LangChain PDF loader not available")
        missing_packages.append("langchain_community")
    
    return len(missing_packages) == 0

if __name__ == "__main__":
    if not verify_installation():
        print("Please install missing dependencies and try again.")
        print("pip install PyMuPDF langchain_community")
        exit(1)
    
    main()
