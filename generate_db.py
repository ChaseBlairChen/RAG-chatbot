import os
import shutil
import glob
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # Updated import
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain.schema import Document
import fitz  # PyMuPDF
import re

# IMPORTANT: Use the same embedding model as your app.py
CHROMA_PATH = os.path.join(os.getcwd(), "my_chroma_db")
DOCUMENTS_FOLDER = "documents"

def main():
    print("Starting document ingestion process...")
    
    # Check if documents folder exists
    if not os.path.exists(DOCUMENTS_FOLDER):
        print(f"Creating documents folder: {DOCUMENTS_FOLDER}")
        os.makedirs(DOCUMENTS_FOLDER)
        print("Please add your PDF and Markdown files to the 'documents' folder and run this script again.")
        return
    
    # List files in documents folder
    all_files = []
    for ext in ['*.pdf', '*.md', '*.txt']:
        all_files.extend(glob.glob(os.path.join(DOCUMENTS_FOLDER, ext)))
        all_files.extend(glob.glob(os.path.join(DOCUMENTS_FOLDER, "**", ext), recursive=True))
    
    print(f"Found {len(all_files)} files in documents folder:")
    for file in all_files:
        print(f"  - {file}")
    
    if not all_files:
        print("No documents found! Please add PDF, Markdown, or text files to the 'documents' folder.")
        return
    
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    if not documents:
        print("No documents were successfully loaded.")
        return
    
    print(f"Loaded {len(documents)} documents")
    
    # Quality check before chunking
    documents = validate_and_clean_documents(documents)
    if not documents:
        print("No documents passed validation.")
        return
        
    chunks = split_text(documents)
    if not chunks:
        print("No chunks were created.")
        return
        
    save_to_chroma(chunks)

def load_documents() -> List[Document]:
    """Load all supported document types"""
    if not os.path.exists(DOCUMENTS_FOLDER):
        raise FileNotFoundError(f"Documents folder '{DOCUMENTS_FOLDER}' not found.")
    
    all_documents = []
    
    # Get all supported files
    pdf_files = glob.glob(os.path.join(DOCUMENTS_FOLDER, "*.pdf"))
    pdf_files.extend(glob.glob(os.path.join(DOCUMENTS_FOLDER, "**/*.pdf"), recursive=True))
    
    md_files = glob.glob(os.path.join(DOCUMENTS_FOLDER, "*.md"))
    md_files.extend(glob.glob(os.path.join(DOCUMENTS_FOLDER, "**/*.md"), recursive=True))
    
    txt_files = glob.glob(os.path.join(DOCUMENTS_FOLDER, "*.txt"))
    txt_files.extend(glob.glob(os.path.join(DOCUMENTS_FOLDER, "**/*.txt"), recursive=True))
    
    print(f"Found {len(pdf_files)} PDF files, {len(md_files)} Markdown files, {len(txt_files)} text files")
    
    # Load PDF files
    for pdf_file in pdf_files:
        try:
            print(f"Loading PDF: {pdf_file}")
            docs = load_pdf_simple(pdf_file)
            all_documents.extend(docs)
            print(f"Successfully loaded {len(docs)} pages from {pdf_file}")
        except Exception as e:
            print(f"Error loading PDF {pdf_file}: {e}")
    
    # Load Markdown files
    for md_file in md_files:
        try:
            print(f"Loading Markdown: {md_file}")
            docs = load_text_file(md_file, "markdown")
            all_documents.extend(docs)
            print(f"Successfully loaded {md_file}")
        except Exception as e:
            print(f"Error loading Markdown {md_file}: {e}")
    
    # Load text files
    for txt_file in txt_files:
        try:
            print(f"Loading text file: {txt_file}")
            docs = load_text_file(txt_file, "text")
            all_documents.extend(docs)
            print(f"Successfully loaded {txt_file}")
        except Exception as e:
            print(f"Error loading text file {txt_file}: {e}")
    
    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents

def load_pdf_simple(pdf_path: str) -> List[Document]:
    """Simple, reliable PDF loading"""
    documents = []
    
    try:
        pdf_document = fitz.open(pdf_path)
        file_name = os.path.basename(pdf_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text = page.get_text()
            
            # Clean the text
            text = clean_text_simple(text)
            
            # Only keep pages with substantial content
            if len(text.strip()) > 100:
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "file_name": file_name,
                        "file_type": "pdf",
                        "page_number": page_num + 1,
                        "total_pages": len(pdf_document)
                    }
                )
                documents.append(doc)
        
        pdf_document.close()
        
    except Exception as e:
        print(f"Failed to load PDF {pdf_path}: {e}")
        raise e
    
    return documents

def load_text_file(file_path: str, file_type: str) -> List[Document]:
    """Load text or markdown files"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Clean the content
        content = clean_text_simple(content)
        
        if len(content.strip()) > 50:
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "file_name": os.path.basename(file_path),
                    "file_type": file_type
                }
            )
            return [doc]
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        raise e
    
    return []

def clean_text_simple(text: str) -> str:
    """Simple but effective text cleaning"""
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Remove obvious page artifacts
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip page numbers and short artifacts
        if re.match(r'^\d+$', line) and len(line) < 4:
            continue
        if len(line) < 2:
            continue
            
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()

def validate_and_clean_documents(documents: List[Document]) -> List[Document]:
    """Basic document validation"""
    clean_documents = []
    
    for doc in documents:
        content = doc.page_content.strip()
        
        # Skip documents that are too short
        if len(content) < 50:
            print(f"Skipping short document from {doc.metadata.get('source', 'unknown')}")
            continue
        
        # Basic quality check
        words = content.split()
        if len(words) < 10:
            print(f"Skipping document with too few words from {doc.metadata.get('source', 'unknown')}")
            continue
        
        # Add word count
        doc.metadata["word_count"] = len(words)
        clean_documents.append(doc)
    
    print(f"Kept {len(clean_documents)} out of {len(documents)} documents after validation")
    return clean_documents

def split_text(documents: List[Document]) -> List[Document]:
    """Split documents into chunks for RAG"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    # Add chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
    
    if chunks:
        print(f"\nFirst chunk preview:")
        print(f"Content: {chunks[0].page_content[:200]}...")
        print(f"Metadata: {chunks[0].metadata}")
    
    return chunks

def save_to_chroma(chunks: List[Document]):
    """Save chunks to Chroma database"""
    if os.path.exists(CHROMA_PATH):
        print(f"Deleting existing database: {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)
    
    os.makedirs(os.path.dirname(CHROMA_PATH), exist_ok=True)
    
    # CRITICAL: Use the same embedding model as app.py
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        print(f"Creating Chroma database with {len(chunks)} chunks...")
        
        # Process in smaller batches
        batch_size = 20
        db = None
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            try:
                if db is None:
                    # Create the database with first batch
                    db = Chroma.from_documents(
                        batch, 
                        embedding_function, 
                        persist_directory=CHROMA_PATH
                    )
                    print(f"Created database with first batch")
                else:
                    # Add subsequent batches
                    db.add_documents(batch)
                    print(f"Added batch {batch_num}")
                    
            except Exception as e:
                print(f"Error processing batch {batch_num}: {e}")
                raise e
        
        if db:
            print("Persisting database...")
            db.persist()
            print(f"✅ Successfully created database at {CHROMA_PATH}")
            
            # Test the database
            test_search(db)
            
            # Print summary
            unique_sources = set(chunk.metadata.get('source', 'unknown') for chunk in chunks)
            pdf_chunks = sum(1 for chunk in chunks if chunk.metadata.get('file_type') == 'pdf')
            text_chunks = len(chunks) - pdf_chunks
            avg_chunk_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
            
            print(f"\n📊 Database Summary:")
            print(f"  • Total chunks: {len(chunks)}")
            print(f"  • PDF chunks: {pdf_chunks}")
            print(f"  • Text/MD chunks: {text_chunks}")
            print(f"  • Unique sources: {len(unique_sources)}")
            print(f"  • Average chunk size: {avg_chunk_size:.0f} characters")
            print(f"  • Database location: {CHROMA_PATH}")
            
        else:
            raise Exception("Failed to create database")
            
    except Exception as e:
        print(f"❌ Error creating Chroma database: {e}")
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        raise

def test_search(db):
    """Test the database with a simple search"""
    try:
        print("\n🔍 Testing database search...")
        test_results = db.similarity_search("test", k=3)
        print(f"  • Search returned {len(test_results)} results")
        
        if test_results:
            first_result = test_results[0]
            print(f"  • First result preview: {first_result.page_content[:100]}...")
            print(f"  • First result metadata: {first_result.metadata}")
        
        return True
    except Exception as e:
        print(f"  • Search test failed: {e}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    missing = []
    
    try:
        import fitz
        print("✅ PyMuPDF available")
    except ImportError:
        print("❌ PyMuPDF missing")
        missing.append("PyMuPDF")
    
    try:
        from langchain_chroma import Chroma
        print("✅ langchain_chroma available")
    except ImportError:
        print("❌ langchain_chroma missing")
        missing.append("langchain_chroma")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        print("✅ langchain_huggingface available")
    except ImportError:
        print("❌ langchain_huggingface missing")
        missing.append("langchain_huggingface")
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True

if __name__ == "__main__":
    print("=== Document Ingestion Script ===\n")
    
    if not check_dependencies():
        exit(1)
    
    try:
        main()
        print("\n✅ Ingestion completed successfully!")
        print(f"Your database is ready at: {CHROMA_PATH}")
        print("You can now start your RAG application.")
        
    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        exit(1)
