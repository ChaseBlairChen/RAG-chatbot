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
    
    # Load PDF files with PyMuPDF (better extraction)
    for pdf_file in pdf_files:
        try:
            print(f"Loading PDF: {pdf_file}")
            docs = load_pdf_with_pymupdf(pdf_file)
            all_documents.extend(docs)
            print(f"Successfully loaded {len(docs)} pages from {pdf_file}")
        except Exception as e:
            print(f"Error loading PDF {pdf_file}: {e}")
            # Fallback to PyPDFLoader if PyMuPDF fails
            try:
                print(f"Trying fallback loader for {pdf_file}")
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(pdf_file)
                documents = loader.load()
                for doc in documents:
                    doc.metadata["source"] = pdf_file
                    doc.metadata["file_type"] = "pdf"
                    doc.metadata["extraction_method"] = "pypdf_fallback"
                all_documents.extend(documents)
                print(f"Fallback successful: {len(documents)} pages")
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

def load_pdf_with_pymupdf(pdf_path: str) -> List[Document]:
    """Load PDF using PyMuPDF for better text extraction"""
    documents = []
    
    try:
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            
            # Extract text with better formatting
            text = page.get_text()
            
            # Clean up common PDF extraction issues
            text = clean_pdf_text(text)
            
            if len(text.strip()) > 50:  # Only include pages with substantial content
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "file_type": "pdf",
                        "page_number": page_num + 1,
                        "total_pages": len(pdf_document),
                        "extraction_method": "pymupdf",
                        "file_name": os.path.basename(pdf_path)
                    }
                )
                documents.append(doc)
        
        pdf_document.close()
        
    except Exception as e:
        print(f"PyMuPDF extraction failed for {pdf_path}: {e}")
        raise e
    
    return documents

def clean_pdf_text(text: str) -> str:
    """Clean common PDF extraction artifacts"""
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
        if len(content) < 100:
            print(f"Skipping short document from {doc.metadata.get('source', 'unknown')}")
            continue
            
        # Check for garbled text (too many non-alphanumeric characters)
        alpha_ratio = sum(c.isalnum() or c.isspace() for c in content) / len(content)
        if alpha_ratio < 0.7:
            print(f"Skipping likely garbled document from {doc.metadata.get('source', 'unknown')}")
            continue
            
        # Add word count metadata for retrieval scoring
        word_count = len(content.split())
        doc.metadata["word_count"] = word_count
        
        clean_documents.append(doc)
    
    print(f"Kept {len(clean_documents)} out of {len(documents)} documents after quality check")
    return clean_documents

def split_text(documents: List[Document]) -> List[Document]:
    """Split documents into chunks optimized for RAG"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Smaller chunks for better retrieval precision
        chunk_overlap=100,  # Reasonable overlap for context
        length_function=len,
        add_start_index=True,
        # Better separators for structured content
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    # Add chunk-specific metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
        
        # Extract first sentence as summary for better retrieval
        sentences = chunk.page_content.split('. ')
        if sentences:
            chunk.metadata["first_sentence"] = sentences[0][:200] + "..."
    
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
        batch_size = 50  # Smaller batches for stability
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
            
            # Enhanced summary for RAG
            pdf_chunks = [c for c in chunks if c.metadata.get("file_type") == "pdf"]
            md_chunks = [c for c in chunks if c.metadata.get("file_type") == "markdown"]
            avg_chunk_size = sum(c.metadata.get("chunk_size", 0) for c in chunks) / len(chunks)
            
            print(f"RAG Database Summary:")
            print(f"- {len(pdf_chunks)} PDF chunks, {len(md_chunks)} Markdown chunks")
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
