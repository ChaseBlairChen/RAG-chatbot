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
import sqlite3
import chromadb
from chromadb.config import Settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PATH = os.path.join(os.getcwd(), "my_chroma_db")
DOCUMENTS_FOLDER = "documents"

def fix_chroma_database():
    """Fix Chroma database schema issues"""
    try:
        # Remove corrupted database
        if os.path.exists(CHROMA_PATH):
            logger.info(f"Removing corrupted database: {CHROMA_PATH}")
            shutil.rmtree(CHROMA_PATH)
        
        # Create directory
        os.makedirs(CHROMA_PATH, exist_ok=True)
        
        # Initialize a new ChromaDB client with proper settings
        client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Test client
        client.heartbeat()
        logger.info("✅ ChromaDB client initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to fix Chroma database: {e}")
        return False

def main():
    print("=== Fixed Document Ingestion Script ===\n")
    
    # Fix database issues first
    if not fix_chroma_database():
        print("❌ Failed to fix database issues")
        return
    
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

def extract_entities_and_aliases(text: str, filename: str) -> dict:
    """Enhanced entity extraction with alias mapping for better retrieval"""
    entities = {
        'bills': [],
        'acts': [],
        'organizations': [],
        'people': [],
        'aliases': {}  # Map aliases to canonical names
    }
    
    # Common bill aliases and their canonical names
    bill_aliases = {
        'one big beautiful bill': ['Inflation Reduction Act', 'IRA', 'H.R.5376'],
        'build back better': ['Build Back Better Act', 'BBB', 'H.R.5376'],
        'infrastructure bill': ['Infrastructure Investment and Jobs Act', 'IIJA', 'H.R.3684'],
        'chips act': ['CHIPS and Science Act', 'H.R.4346'],
        'climate bill': ['Inflation Reduction Act', 'Climate provisions'],
    }
    
    # Extract potential bill names and numbers
    import re
    
    # H.R./S. bill patterns
    hr_pattern = r'\b(?:H\.R\.|S\.)\s*\d+(?:-\d+)?\b'
    hr_matches = re.findall(hr_pattern, text, re.IGNORECASE)
    entities['bills'].extend(hr_matches)
    
    # Act patterns
    act_pattern = r'\b([A-Z][a-zA-Z\s]+(?:Act|Bill|Resolution))\b'
    act_matches = re.findall(act_pattern, text)
    entities['acts'].extend([act.strip() for act in act_matches if len(act.strip()) > 5])
    
    # Add filename-based context
    if 'inflation' in filename.lower() or 'ira' in filename.lower():
        entities['acts'].append('Inflation Reduction Act')
        entities['aliases']['one big beautiful bill'] = 'Inflation Reduction Act'
    
    if 'infrastructure' in filename.lower():
        entities['acts'].append('Infrastructure Investment and Jobs Act')
        entities['aliases']['infrastructure bill'] = 'Infrastructure Investment and Jobs Act'
    
    # Merge common aliases
    for alias, canonical_names in bill_aliases.items():
        if any(name.lower() in text.lower() for name in canonical_names):
            entities['aliases'][alias] = canonical_names[0]
    
    return entities

def load_pdf_with_unstructured(pdf_path: str) -> List[Document]:
    """Load PDF using Unstructured with enhanced entity extraction"""
    try:
        from unstructured.partition.pdf import partition_pdf
        from unstructured.cleaners.core import clean_extra_whitespace, clean_dashes
        
        logger.info(f"Processing {pdf_path} with Unstructured...")
        
        # Partition PDF with advanced options
        elements = partition_pdf(
            filename=pdf_path,
            strategy="fast",
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=1000,
            new_after_n_chars=800,
            combine_text_under_n_chars=50,
        )
        
        documents = []
        file_name = os.path.basename(pdf_path)
        
        # Extract entities from the entire document first
        full_text = " ".join([str(element) for element in elements])
        entities = extract_entities_and_aliases(full_text, file_name)
        
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
                            "section_number": section_count,
                            "entities": entities,  # Add entity metadata
                            "aliases": entities.get('aliases', {})
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
                            "section_number": section_count,
                            "entities": entities,
                            "aliases": entities.get('aliases', {})
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
                        "section_number": section_count,
                        "entities": entities,
                        "aliases": entities.get('aliases', {})
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
                    "section_number": section_count,
                    "entities": entities,
                    "aliases": entities.get('aliases', {})
                }
            )
            documents.append(doc)
        
        logger.info(f"Extracted {len(documents)} structured sections from {pdf_path}")
        return documents
        
    except Exception as e:
        logger.error(f"Unstructured processing failed for {pdf_path}: {e}")
        raise e

def load_pdf_with_pymupdf(pdf_path: str) -> List[Document]:
    """Fallback PDF processing with PyMuPDF and entity extraction"""
    try:
        import fitz
        
        logger.info(f"Processing {pdf_path} with PyMuPDF...")
        documents = []
        pdf_document = fitz.open(pdf_path)
        file_name = os.path.basename(pdf_path)
        
        # Extract full text for entity analysis
        full_text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            full_text += page.get_text() + "\n"
        
        entities = extract_entities_and_aliases(full_text, file_name)
        
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
                        "element_type": "page",
                        "entities": entities,
                        "aliases": entities.get('aliases', {})
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
    """Load text or markdown files with entity extraction"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content = clean_text_simple(content)
        file_name = os.path.basename(file_path)
        entities = extract_entities_and_aliases(content, file_name)
        
        if len(content.strip()) > 50:
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "file_name": file_name,
                    "file_type": file_type,
                    "processing_method": "direct_load",
                    "entities": entities,
                    "aliases": entities.get('aliases', {})
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
    """Main function to create the vector database with proper error handling"""
    
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
    """Split documents into chunks for RAG with entity preservation"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,  # Increased overlap for better entity context
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✂️  Split into {len(chunks)} chunks")
    
    # Add chunk metadata and preserve entity information
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
        
        # Preserve entity metadata from parent document
        if 'entities' in chunk.metadata:
            # Add entity aliases to chunk content for better matching
            aliases = chunk.metadata.get('aliases', {})
            if aliases:
                alias_text = "\nKnown aliases: " + ", ".join([f"{alias} ({canonical})" for alias, canonical in aliases.items()])
                chunk.page_content += alias_text
    
    if chunks:
        print(f"\n📄 First chunk preview:")
        print(f"   Content: {chunks[0].page_content[:150]}...")
        print(f"   Metadata: {chunks[0].metadata}")
    
    return chunks

def create_vector_database(chunks: List[Document]):
    """Create the Chroma vector database with proper error handling"""
    
    try:
        # Use same embedding model as app.py
        embedding_function = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        print(f"🔄 Creating vector database with {len(chunks)} chunks...")
        
        # Create database with proper settings
        db = Chroma.from_documents(
            chunks,
            embedding_function,
            persist_directory=CHROMA_PATH,
            collection_name="default"
        )
        
        print("💾 Database created successfully")
        
        # Test the database
        test_results = db.similarity_search("test", k=3)
        print(f"✅ Database created successfully!")
        print(f"   Test search returned {len(test_results)} results")
        
        # Print summary
        print_summary(chunks)
        
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
    
    # Count entities
    total_entities = sum(len(chunk.metadata.get('entities', {}).get('acts', [])) for chunk in chunks)
    
    print(f"\n📊 Database Summary:")
    print(f"   • Total chunks: {len(chunks)}")
    print(f"   • PDF chunks: {pdf_chunks}")
    print(f"   • Text/MD chunks: {text_chunks}")
    print(f"   • Unique sources: {len(unique_sources)}")
    print(f"   • Average chunk size: {avg_chunk_size:.0f} characters")
    print(f"   • Enhanced processing: {unstructured_chunks} chunks")
    print(f"   • Standard processing: {pymupdf_chunks} chunks")
    print(f"   • Total entities extracted: {total_entities}")
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
    
    try:
        import chromadb
        print("✅ chromadb available")
    except ImportError:
        print("❌ chromadb missing")
        missing.append("chromadb")
    
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
