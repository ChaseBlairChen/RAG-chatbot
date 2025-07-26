# --- generate_db.py ---
import os
import sys  # Added this import
import shutil
import glob
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import logging
import chromadb
from chromadb.config import Settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SUGGESTION: Change database folder name to avoid conflicts ---
# Using an absolute path can also help prevent issues.
CHROMA_PATH = os.path.abspath(os.path.join(os.getcwd(), "chromadb-database")) # Changed from "my_chroma_db"
# --- END SUGGESTION ---
DOCUMENTS_FOLDER = "documents"

def fix_chroma_database():
    """Fix Chroma database schema issues with aggressive cleanup and client reset."""
    try:
        print(f"[INIT] Intended database path is: {CHROMA_PATH}")
        
        # CRITICAL: Reset ChromaDB's internal client cache first
        # This is the key fix for your error
        try:
            import chromadb
            from chromadb.api.client import SharedSystemClient
            
            # Reset the shared system client cache
            if hasattr(SharedSystemClient, '_identifer_to_system'):
                SharedSystemClient._identifer_to_system.clear()
                print("[INIT] Cleared ChromaDB client cache")
            
            # Alternative approach - clear the entire module cache
            if 'chromadb.api.client' in sys.modules:
                delattr(sys.modules['chromadb.api.client'], '_identifer_to_system') if hasattr(sys.modules['chromadb.api.client'], '_identifer_to_system') else None
                print("[INIT] Reset ChromaDB module state")
                
        except Exception as cache_clear_error:
            print(f"[INIT] Cache clear attempt failed (this might be OK): {cache_clear_error}")
        
        # --- AGGRESSIVE CLEANUP ---
        if os.path.exists(CHROMA_PATH):
            print(f"[INIT] Found existing directory at {CHROMA_PATH}. Removing it to ensure a clean start.")
            try:
                # Standard removal
                shutil.rmtree(CHROMA_PATH)
                print(f"[INIT] Successfully removed {CHROMA_PATH} with rmtree.")
            except Exception as e1:
                print(f"[INIT] rmtree failed: {e1}")
                # Fallback: Try removing files individually if rmtree fails (e.g., due to locks)
                try:
                    print("[INIT] Trying individual file/directory removal...")
                    import glob
                    for item in glob.glob(os.path.join(CHROMA_PATH, "*")):
                        try:
                            if os.path.isfile(item) or os.path.islink(item):
                                os.unlink(item)
                            elif os.path.isdir(item):
                                shutil.rmtree(item, ignore_errors=True)
                        except Exception as e_item:
                            print(f"[INIT] Failed to remove item {item}: {e_item}")
                    # Try to remove the now (hopefully) empty directory
                    try:
                        os.rmdir(CHROMA_PATH)
                        print(f"[INIT] Successfully removed {CHROMA_PATH} via individual item removal.")
                    except Exception as e_rmdir:
                         print(f"[INIT] Failed to remove directory {CHROMA_PATH} after item removal: {e_rmdir}")
                         # If even rmdir fails, it's a critical issue.
                         raise RuntimeError(f"Could not finalize removal of {CHROMA_PATH}") from e_rmdir

                except Exception as e2:
                    print(f"[INIT] Individual file removal also failed: {e2}")
                    # Critical failure - cannot proceed if path exists and cannot be removed
                    raise RuntimeError(f"CRITICAL: Could not remove existing database path {CHROMA_PATH}. Cannot proceed with creation. Error 1: {e1}. Error 2: {e2}")
        # --- END AGGRESSIVE CLEANUP ---

        # Create directory cleanly
        os.makedirs(CHROMA_PATH, exist_ok=True)
        print(f"[INIT] Created clean database directory: {CHROMA_PATH}")

        # Longer delay to allow OS and ChromaDB to release handles
        import time
        time.sleep(1.0)  # Increased from 0.2 to 1.0 seconds

        # DON'T test the client here - let the main creation process handle it
        # This avoids creating a client with different settings than what will be used later
        print("✅ Database path prepared successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to fix Chroma database: {e}")
        return False

def main():
    print("=== Fixed Document Ingestion Script ===")
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

# --- SUGGESTION 1 & 2: Enhanced Alias Expansion ---
# Updated to ensure aliases are correctly built and added to metadata.
# Simplified alias mapping to one canonical name per alias.
def extract_entities_and_aliases(text: str, filename: str) -> dict:
    """Enhanced entity extraction with alias mapping for better retrieval"""
    entities = {
        'bills': [],
        'acts': [],
        'organizations': [],
        'people': [],
        'aliases': {}  # Map alias -> single canonical name
    }

    # Common bill aliases and their primary canonical name
    # Ensure the mapping is from alias -> canonical name (string, not list)
    bill_aliases = {
        'one big beautiful bill': 'Inflation Reduction Act',
        'build back better': 'Build Back Better Act', # Simplified for consistency
        'infrastructure bill': 'Infrastructure Investment and Jobs Act',
        'chips act': 'CHIPS and Science Act',
        'climate bill': 'Inflation Reduction Act', # Maps to the same canonical name
        # Add more aliases as discovered
    }

    # Extract potential bill names and numbers
    import re
    # H.R./S. bill patterns
    hr_pattern = r'\b(?:H\.R\.|S\.)\s*\d+(?:-\d+)?\b'
    hr_matches = re.findall(hr_pattern, text, re.IGNORECASE)
    entities['bills'].extend(hr_matches)

    # Act patterns (improved to be less greedy)
    act_pattern = r'\b([A-Z][a-zA-Z\s]*(?:Act|Law))\b' # Match Acts and Laws
    act_matches = re.findall(act_pattern, text)
    # Filter out very short matches and add
    entities['acts'].extend([act.strip() for act in act_matches if len(act.strip()) > 5 and act.strip() != "Act"])

    # Add filename-based context and map aliases
    # Check for keywords in filename to trigger alias mapping
    lower_filename = filename.lower()
    if 'inflation' in lower_filename or 'ira' in lower_filename:
        entities['acts'].append('Inflation Reduction Act')
        # Add alias mapping if not already present or override
        entities['aliases']['one big beautiful bill'] = 'Inflation Reduction Act'
        entities['aliases']['climate bill'] = 'Inflation Reduction Act' # Add this one too if filename matches

    if 'infrastructure' in lower_filename:
        entities['acts'].append('Infrastructure Investment and Jobs Act')
        entities['aliases']['infrastructure bill'] = 'Infrastructure Investment and Jobs Act'

    if 'chips' in lower_filename and 'science' in lower_filename:
        entities['acts'].append('CHIPS and Science Act')
        entities['aliases']['chips act'] = 'CHIPS and Science Act'

    # Scan text for canonical names and add corresponding aliases
    lower_text = text.lower()
    for alias, canonical_name in bill_aliases.items():
        # Check if the canonical name is mentioned in the text
        if canonical_name.lower() in lower_text:
            # Add the alias mapping (this ensures aliases are in metadata even if not in filename)
            # It will overwrite if filename logic already set it, which is fine.
            entities['aliases'][alias] = canonical_name
        # Also check if the alias itself is mentioned (might be useful for direct alias mentions in docs)
        # This part is nuanced; rely on filename logic and canonical name presence for now.

    # Deduplicate acts list
    entities['acts'] = list(set(entities['acts']))

    # Log aliases found for debugging
    if entities['aliases']:
        logger.info(f"Aliases found for document '{filename}': {entities['aliases']}")

    return entities
# --- END SUGGESTION 1 & 2 ---

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
                            "aliases": entities.get('aliases', {}) # Ensure aliases are in metadata
                        }
                    )
                    documents.append(doc)
                    section_count += 1
                # Start new section
                current_title = text.strip()
                current_section = f"{text}\n"
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
                table_content = f"TABLE from section: {current_title}\n{text}"
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
                current_section += f"{text}\n"

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
                        "aliases": entities.get('aliases', {}) # Ensure aliases are in metadata
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
                    "aliases": entities.get('aliases', {}) # Ensure aliases are in metadata
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
        chunk_size=200,     # Very small chunks for precise legal document retrieval
        chunk_overlap=50,   # Smaller overlap but still maintains context
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""]  # Added legal-specific separators
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️  Split into {len(chunks)} chunks")

    # Add chunk metadata and preserve entity information
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
        # Preserve entity metadata from parent document
        # --- MODIFIED: Ensure aliases are handled correctly and injected into content ---
        if 'entities' in chunk.metadata:
            # Access aliases from the chunk's metadata (which was copied from parent doc)
            aliases = chunk.metadata.get('aliases', {})
            if aliases:
                # Format aliases for inclusion in chunk text
                # This makes aliases directly searchable by the embedding model
                alias_text = "\n\n[Context Note: This document may also refer to the following terms: " + ", ".join([f"'{alias}' (meaning '{canonical}')" for alias, canonical in aliases.items()]) + "]"
                # Append alias information to the chunk content
                chunk.page_content += alias_text
        # --- END MODIFICATION ---

    if chunks:
        print(f"\n📄 First chunk preview:")
        print(f"   Content: {chunks[0].page_content[:200]}...")
        print(f"   Metadata: {chunks[0].metadata}")

    return chunks

def filter_metadata_for_chroma(metadata: dict) -> dict:
    """Filter metadata to only include types supported by ChromaDB (str, int, float, bool)"""
    filtered = {}
    
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            filtered[key] = value
        elif isinstance(value, dict):
            # Convert dict to JSON string for storage
            if value:  # Only if dict is not empty
                import json
                try:
                    filtered[f"{key}_json"] = json.dumps(value)
                except (TypeError, ValueError):
                    # Skip if can't serialize
                    pass
        elif isinstance(value, list):
            # Convert list to comma-separated string if it contains simple types
            if value and all(isinstance(item, (str, int, float, bool)) for item in value):
                filtered[f"{key}_list"] = ", ".join(str(item) for item in value[:10])  # Limit to first 10 items
        # Skip other complex types
    
    return filtered

def create_vector_database(chunks: List[Document]):
    """Create the Chroma vector database with explicit settings and error handling."""
    global CHROMA_PATH
    
    try:
        # Use same embedding model as app.py
        embedding_function = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"🔄 Creating vector database with {len(chunks)} chunks...")

        # --- ROBUST METHOD: Create client with explicit settings ---
        print("[CREATION] Creating ChromaDB client with explicit settings...")
        
        # Import required settings
        from chromadb.config import Settings
        
        # Create client with explicit settings that match what we want
        client_settings = Settings(
            persist_directory=CHROMA_PATH,
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True
        )
        
        # Create the Chroma instance with explicit client settings
        db = Chroma(
            collection_name="default",
            embedding_function=embedding_function,
            persist_directory=CHROMA_PATH,
            client_settings=client_settings
        )
        
        print("[CREATION] ChromaDB client created successfully.")
        
        # Add documents in batches to avoid memory issues
        batch_size = 100
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size
            
            print(f"[CREATION] Adding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            
            # Extract texts and filter metadatas for this batch
            texts = [chunk.page_content for chunk in batch]
            metadatas = [filter_metadata_for_chroma(chunk.metadata) for chunk in batch]
            ids = [f"chunk_{chunk.metadata.get('chunk_id', i + j)}" for j, chunk in enumerate(batch)]
            
            # Add to database
            db.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
        
        print("[CREATION] All chunks added successfully.")
        
        # Persist the database
        print("[CREATION] Persisting database...")
        # Note: Modern versions of Chroma auto-persist, but we can be explicit
        if hasattr(db, 'persist'):
            db.persist()
        
        print("💾 Database creation process completed.")
        
        # Test the database
        test_results = db.similarity_search("test", k=3)
        print(f"✅ Database created and tested successfully!")
        print(f"   Test search returned {len(test_results)} results")
        
        # Print summary
        print_summary(chunks)

    except Exception as e:
        print(f"❌ Error during database creation process: {e}")
        print(f"   Full error type: {type(e).__name__}")
        
        # If it's the specific "different settings" error, try one more approach
        if "different settings" in str(e):
            print("🔄 Attempting to resolve 'different settings' error...")
            try:
                # Force garbage collection to clear any cached clients
                import gc
                gc.collect()
                
                # Try to reset ChromaDB's internal state more aggressively
                import chromadb
                if hasattr(chromadb, '_client_cache'):
                    chromadb._client_cache.clear()
                
                # Wait a bit more
                import time
                time.sleep(2.0)
                
                # Try the creation one more time with a different collection name
                print("🔄 Retrying with fresh client...")
                db = Chroma(
                    collection_name=f"default_{int(time.time())}",  # Unique collection name
                    embedding_function=embedding_function,
                    persist_directory=CHROMA_PATH
                )
                
                # Add documents with filtered metadata
                texts = [chunk.page_content for chunk in chunks]
                metadatas = [filter_metadata_for_chroma(chunk.metadata) for chunk in chunks]
                db.add_texts(texts=texts, metadatas=metadatas)
                
                print("✅ Retry successful!")
                print_summary(chunks)
                return
                
            except Exception as retry_error:
                print(f"❌ Retry also failed: {retry_error}")
        
        # Final cleanup attempt
        if os.path.exists(CHROMA_PATH):
            try:
                shutil.rmtree(CHROMA_PATH)
                print("🧹 Final cleanup: Database path removed after creation failure.")
            except Exception as cleanup_e:
                print(f"🧹 Final cleanup failed: {cleanup_e}")
        
        raise  # Re-raise the original error

def print_summary(chunks: List[Document]):
    """Print a summary of the created database"""
    unique_sources = set(chunk.metadata.get('source', 'unknown') for chunk in chunks)
    pdf_chunks = sum(1 for chunk in chunks if chunk.metadata.get('file_type') == 'pdf')
    text_chunks = len(chunks) - pdf_chunks
    avg_chunk_size = sum(len(chunk.page_content) for chunk in chunks) / len(chunks) if chunks else 0

    # Count processing methods
    unstructured_chunks = sum(1 for chunk in chunks if chunk.metadata.get('processing_method') == 'unstructured')
    pymupdf_chunks = sum(1 for chunk in chunks if chunk.metadata.get('processing_method') == 'pymupdf')

    # Count entities (simple count of acts mentioned)
    total_acts_mentioned = sum(len(chunk.metadata.get('entities', {}).get('acts', [])) for chunk in chunks)

    # Count chunks with aliases
    chunks_with_aliases = sum(1 for chunk in chunks if chunk.metadata.get('aliases'))

    print(f"\n📊 Database Summary:")
    print(f"   • Total chunks: {len(chunks)}")
    print(f"   • PDF chunks: {pdf_chunks}")
    print(f"   • Text/MD chunks: {text_chunks}")
    print(f"   • Unique sources: {len(unique_sources)}")
    print(f"   • Average chunk size: {avg_chunk_size:.0f} characters")
    print(f"   • Enhanced processing (Unstructured): {unstructured_chunks} chunks")
    print(f"   • Standard processing (PyMuPDF): {pymupdf_chunks} chunks")
    print(f"   • Total 'Act' names mentioned in metadata: {total_acts_mentioned}")
    print(f"   • Chunks containing alias information: {chunks_with_aliases}")
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
        # --- REMINDER ---
        print("\n--- IMPORTANT ---")
        print("Please ensure your 'app.py' also uses the same CHROMA_PATH:")
        print(f"   CHROMA_PATH = '{CHROMA_PATH}'")
        print("Or, if using a relative path in app.py:")
        print(f"   CHROMA_PATH = os.path.join(os.getcwd(), 'chromadb-database')")
        print("------------------\n")
        print("You can now start your RAG application with: uvicorn app:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"\n💥 Ingestion failed: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        exit(1)

# --- End of generate_db.py ---
