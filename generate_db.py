import os
import shutil
import glob
from typing import List
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Use home directory to avoid permission issues
CHROMA_PATH = os.path.join(os.getcwd(), "my_chroma_db")
DOCUMENTS_FOLDER = "documents"  # Change this to your folder path

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    if not documents:
        print("No documents found to process.")
        return
    
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents() -> List[Document]:
    """Load all PDF and Markdown documents from the specified folder"""
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
    
    # Load PDF files
    for pdf_file in pdf_files:
        try:
            print(f"Loading PDF: {pdf_file}")
            loader = PyPDFLoader(pdf_file)
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata["source"] = pdf_file
                doc.metadata["file_type"] = "pdf"
            
            all_documents.extend(documents)
            print(f"Successfully loaded {len(documents)} pages from {pdf_file}")
        except Exception as e:
            print(f"Error loading PDF {pdf_file}: {e}")
    
    # Load Markdown files
    for md_file in md_files:
        try:
            print(f"Loading Markdown: {md_file}")
            loader = TextLoader(md_file, encoding='utf-8')
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata["source"] = md_file
                doc.metadata["file_type"] = "markdown"
            
            all_documents.extend(documents)
            print(f"Successfully loaded {md_file}")
        except Exception as e:
            print(f"Error loading Markdown {md_file}: {e}")
    
    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents

def split_text(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increased chunk size for better context
        chunk_overlap=200,  # Increased overlap for better continuity
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    # Show sample chunk information
    if chunks:
        print("\nSample chunk:")
        print(f"Content preview: {chunks[0].page_content[:200]}...")
        print(f"Metadata: {chunks[0].metadata}")
    
    return chunks

def save_to_chroma(chunks: List[Document]):
    """Save document chunks to Chroma vector store"""
    if os.path.exists(CHROMA_PATH):
        print(f"Deleting existing {CHROMA_PATH} folder")
        shutil.rmtree(CHROMA_PATH)
    
    os.makedirs(CHROMA_PATH, exist_ok=True)
    
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    try:
        # Process in batches to avoid memory issues with large document sets
        batch_size = 100
        total_chunks = len(chunks)
        
        print(f"Processing {total_chunks} chunks in batches of {batch_size}")
        
        db = None
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            print(f"Processing batch {batch_num}/{(total_chunks + batch_size - 1) // batch_size}")
            
            if db is None:
                # Create the database with the first batch
                db = Chroma.from_documents(
                    batch, 
                    embedding_function, 
                    persist_directory=CHROMA_PATH
                )
            else:
                # Add subsequent batches to the existing database
                db.add_documents(batch)
        
        if db:
            db.persist()
            print(f"Successfully saved {total_chunks} chunks to {CHROMA_PATH}")
            
            # Print summary by file type
            pdf_chunks = [c for c in chunks if c.metadata.get("file_type") == "pdf"]
            md_chunks = [c for c in chunks if c.metadata.get("file_type") == "markdown"]
            print(f"Summary: {len(pdf_chunks)} PDF chunks, {len(md_chunks)} Markdown chunks")
        
    except Exception as e:
        print(f"Error creating Chroma DB: {e}")
        raise

def verify_installation():
    """Verify that required packages are installed"""
    try:
        import pypdf
        print("✓ PyPDF is available")
    except ImportError:
        print("✗ PyPDF not found. Install with: pip install pypdf")
        return False
    
    try:
        from langchain_community.document_loaders import PyPDFLoader
        print("✓ LangChain PDF loader is available")
    except ImportError:
        print("✗ LangChain PDF loader not available")
        return False
    
    return True

if __name__ == "__main__":
    # Verify required packages
    if not verify_installation():
        print("Please install missing dependencies and try again.")
        exit(1)
    
    main()
