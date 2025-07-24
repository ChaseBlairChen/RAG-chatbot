import os
import shutil
import glob
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PATH = os.path.join(os.getcwd(), "my_chroma_db")
DOCUMENTS_FOLDER = "documents"

def main():
    print("=== Enhanced Document Ingestion Script ===\n")
    
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

def check_unstructured_availability():
    """Check if Unstructured is available"""
    try:
        from unstructured.partition.pdf import partition_pdf
        from unstructured.cleaners.core import clean_extra_whitespace, clean_dashes
        print("✅ Unstructured library available - using enhanced PDF processing")
        return True
    except ImportError:
        print("ℹ️  Unstructured library not available - using PyMuPDF fallback")
        print("   Install with: pip install unstructured[pdf]")
        return False

def load_pdf_with_unstructured(pdf_path: str) -> List[Document]:
    """Load PDF using Unstructured with advanced processing"""
    try:
        from unstructured.partition.pdf import partition_pdf
        from unstructured.cleaners.core import clean_extra_whitespace, clean_dashes
        
        logger.info(f"Processing {pdf_path} with Unstructured...")
        
        # Partition PDF with advanced options
        elements = partition_pdf(
            filename=pdf_path,
            # Basic options that work reliably
            strategy="fast",  # Use "hi_res" for better quality but slower processing
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=1000,
            new_after_n_chars=800,
            combine_text_under_n_chars=50,
        )
        
        documents = []
        file_name = os.path.basename(pdf_path)
        
        # Group elements into logical sections
        current_section = ""
        current_title = "Introduction"
        section_count = 1
        
        for i, element in enumerate(elements):
            # Clean the text
            text = str(element)
            text = clean_extra_whitespace(text)
            text = clean_dashes(text)
            
            # Skip very short elements
            if len(text.strip()) < 20:
                continue
            
            # Get element type
            element_type = element.category if hasattr(element, 'category') else 'text'
            
            # Handle different element types
            if element_type in ["Title", "Header"]:
                # Save previous section if it has content
                if current_section.strip() and len(current_section.strip()) > 100:
                    doc = Document(
                        page_content=current_section.strip(),
                        metadata={
                            "source": pdf_path,
                            "file_name": file_name,
                            "file_type": "pdf",
                            "section_title": current_title,
                            "processing_method": "unstructured",
                            "element_type": "section",
                            "section_number": section_count
                        }
                    )
                    documents.append(doc)
                    section_count += 1
                
                # Start new section
                current_title = text.strip()
                current_section = f"{text}\n\n"
                
            elif element_type == "Table":
                # Save current section first
                if current_section.strip() and len(current_section.strip()) > 100:
                    doc = Document(
                        page_content=current_section.strip(),
                        metadata={
                            "source": pdf_path,
                            "file_name": file_name,
                            "file_type": "pdf",
                            "section_title": current_title,
                            "processing_method": "unstructured",
                            "element_type": "section",
                            "section_number": section_count
                        }
                    )
                    documents.append(doc)
                    section_count += 1
                    current_section = ""
                
                # Create separate document for table
                table_content = f"TABLE from section: {current_title}\n\n{text}"
                doc = Document(
                    page_content=table_content,
                    metadata={
                        "source": pdf_path,
                        "file_name": file_name,
                        "file_type": "pdf",
                        "section_title": f"Table - {current_title}",
                        "processing_method": "unstructured",
                        "element_type": "table",
                        "section_number": section_count
                    }
                )
                documents.append(doc)
                section_count += 1
                
            else:
                # Regular content (text, list items, etc.)
                current_section += f"{text}\n\n"
        
        # Don't forget the last section
        if current_section.strip() and len(current_section.strip()) > 100:
            doc = Document(
                page_content=current_section.strip(),
                metadata={
                    "source": pdf_path,
                    "file_name": file_name,
                    "file_type": "pdf",
                    "section_title": current_title,
                    "processing_method": "unstructured",
                    "element_type": "section",
                    "section_number": section_count
                }
            )
            documents.append(doc)
        
        logger.info(f"Extracted {len(documents)} structured sections from {pdf_path}")
        return documents
        
    except Exception as e:
        logger.error(f"Unstructured processing failed for {pdf_path}: {e}")
        raise e

def load_pdf_with_pymupdf(pdf_path: str) -> List[Document]:
    """Fallback PDF processing with PyMuPDF"""
    try:
        import fitz
        
        logger.info(f"Processing {pdf_path} with PyMuPDF...")
        documents = []
        pdf_document = fitz.open(pdf_path)
        file_name = os.path.basename(pdf_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text = page.get_text()
            text = clean_text_simple(text)
            
            if len(text.strip()) > 100:
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "file_name": file_name,
                        "file_type": "pdf",
                        "page_number": page_num + 1,
                        "total_pages": len(pdf_document),
                        "processing_method": "pymupdf",
                        "element_type": "page"
                    }
                )
                documents.append(doc)
        
        pdf_document.close()
        logger.info(f"Extracted {len(documents)} pages from {pdf_path}")
        return documents
        
    except Exception as e:
        logger.error(f"PyMuPDF processing failed for {pdf_path}: {e}")
        raise e

def load_text_file(file_path: str, file_type: str) -> List[Document]:
    """Load text or markdown files"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content = clean_text_simple(content)
        
        if len(content.strip()) > 50:
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "file_name": os.path.basename(file_path),
                    "file_type": file_type,
                    "processing_method": "direct_load"
                }
            )
            return [doc]
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise e
    
    return []

def clean_text_simple(text: str) -> str:
    """Simple but effective text cleaning"""
    import re
    
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

def generate_data_store():
    """Main function to create the vector database"""
    
    # Check what processing methods are available
    use_unstructured = check_unstructured_availability()
    
    documents = []
    
    # Load PDF files
    pdf_files = glob.glob(os.path.join(DOCUMENTS_FOLDER, "*.pdf"))
    pdf_files.extend(glob.glob(os.path.join(DOCUMENTS_FOLDER, "**/*.pdf"), recursive=True))
    
    for pdf_file in pdf_files:
        try:
            if use_unstructured:
                docs = load_pdf_with_unstructured(pdf_file)
            else:
                docs = load_pdf_with_pymupdf(pdf_file)
            
            documents.extend(docs)
            print(f"✅ Processed {pdf_file}: {len(docs)} sections/pages")
            
        except Exception as e:
            print(f"❌ Error processing {pdf_file}: {e}")
            # Try fallback method
            if use_unstructured:
                try:
                    print(f"   Trying PyMuPDF fallback for {pdf_file}")
                    docs = load_pdf_with_pymupdf(pdf_file)
                    documents.extend(docs)
                    print(f"✅ Fallback successful: {len(docs)} pages")
                except Exception as e2:
                    print(f"❌ Fallback also failed: {e2}")
    
    # Load text and markdown files
    for ext, file_type in [('*.md', 'markdown'), ('*.txt', 'text')]:
        files = glob.glob(os.path.join(DOCUMENTS_FOLDER, ext))
        files.extend(glob.glob(os.path.join(DOCUMENTS_FOLDER, "**", ext), recursive=True))
        
        for file_path in files:
            try:
                docs = load_text_file(file_path, file_type)
                documents.extend(docs)
                print(f"✅ Loaded {file_path}")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
    
    if not documents:
        print("❌ No documents were successfully loaded.")
        return
    
    print(f"\n📚 Total documents loaded: {len(documents)}")
    
    # Validate documents
    documents = validate_documents(documents)
    if not documents:
        print("❌ No documents passed validation.")
        return
    
    # Split into chunks
    chunks = split_documents(documents)
    if not chunks:
        print("❌ No chunks were created.")
        return
    
    # Create vector database
    create_vector_database(chunks)

def validate_documents(documents: List[Document]) -> List[Document]:
    """Validate document quality"""
    clean_documents = []
    
    for doc in documents:
        content = doc.page_content.strip()
        
        # Skip documents that are too short
        if len(content) < 50:
            logger.info(f"Skipping short document from {doc.metadata.get('source', 'unknown')}")
            continue
        
        # Basic quality check
        words = content.split()
        if len(words) < 10:
            logger.info(f"Skipping document with too few words from {doc.metadata.get('source', 'unknown')}")
            continue
        
        # Add word count
        doc.metadata["word_count"] = len(words)
        clean_documents.append(doc)
    
    print(f"📋 Validation: kept {len(clean_documents)} out of {len(documents)} documents")
    return clean_documents

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks for RAG"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✂️  Split into {len(chunks)} chunks")
    
    # Add chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
    
    if chunks:
        print(f"\n📄 First chunk preview:")
        print(f"   Content: {chunks[0].page_content[:150]}...")
        print(f"   Metadata: {chunks[0].metadata}")
    
    return chunks

def create_vector_database(chunks: List[Document]):
    """Create the Chroma vector database"""
    
    # Remove existing database
    if os.path.exists(CHROMA_PATH):
        print(f"🗑️  Removing existing database: {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)
    
    os.makedirs(os.path.dirname(CHROMA_PATH), exist_ok=True)
    
    # Use same embedding model as app.py
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        print(f"🔄 Creating vector database with {len(chunks)} chunks...")
        
        # Process in batches for stability
        batch_size = 20
        db = None
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            print(f"   Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            if db is None:
                # Create database with first batch
                db = Chroma.from_documents(
                    batch, 
                    embedding_function, 
                    persist_directory=CHROMA_PATH
                )
            else:
                # Add subsequent batches
                db.add_documents(batch)
        
        if db:
            # Database persists automatically with persist_directory
            print("💾 Database persisted automatically")
            
            # Test the database
            test_results = db.similarity_search("test", k=3)
            print(f"✅ Database created successfully!")
            print(f"   Test search returned {len(test_results)} results")
            
            # Print summary
            print_summary(chunks)
            
        else:
            raise Exception("Failed to create database")
            
    except Exception as e:
        print(f"❌ Error creating database: {e}")
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        raise

def print_summary(chunks: List[Document]):
    """Print a summary of the created database"""
    unique_sources = set(chunk.metadata.get('source', 'unknown') for chunk in chunks)
    pdf_chunks = sum(1 for chunk in chunks if chunk.metadata.get('file_type') == 'pdf')
    text_chunks = len(chunks) - pdf_chunks
    avg_chunk_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
    
    # Count processing methods
    unstructured_chunks = sum(1 for chunk in chunks if chunk.metadata.get('processing_method') == 'unstructured')
    pymupdf_chunks = sum(1 for chunk in chunks if chunk.metadata.get('processing_method') == 'pymupdf')
    
    print(f"\n📊 Database Summary:")
    print(f"   • Total chunks: {len(chunks)}")
    print(f"   • PDF chunks: {pdf_chunks}")
    print(f"   • Text/MD chunks: {text_chunks}")
    print(f"   • Unique sources: {len(unique_sources)}")
    print(f"   • Average chunk size: {avg_chunk_size:.0f} characters")
    print(f"   • Enhanced processing: {unstructured_chunks} chunks")
    print(f"   • Standard processing: {pymupdf_chunks} chunks")
    print(f"   • Database location: {CHROMA_PATH}")

def check_dependencies():
    """Check required dependencies"""
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
    
    # Optional dependency
    try:
        from unstructured.partition.pdf import partition_pdf
        print("✅ unstructured available (enhanced PDF processing)")
    except ImportError:
        print("ℹ️  unstructured not available (will use basic PDF processing)")
        print("   Install with: pip install 'unstructured[pdf]'")
    
    if missing:
        print(f"\n❌ Missing required packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True

if __name__ == "__main__":
    if not check_dependencies():
        exit(1)
    
    try:
        main()
        print("\n🎉 Ingestion completed successfully!")
        print(f"Your database is ready at: {CHROMA_PATH}")
        print("You can now start your RAG application with: uvicorn app:app --host 0.0.0.0 --port 8000")
        
    except Exception as e:
        print(f"\n💥 Ingestion failed: {e}")
        exit(1)
