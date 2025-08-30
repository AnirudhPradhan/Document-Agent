import os
import hashlib
import streamlit as st
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_K

def hash_file(filepath):
    """Generate a unique hash for a file to use as identifier."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def download_pdf_from_url(url, save_dir="uploaded_pdfs"):
    """Download a PDF from URL and save locally."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Clean filename from URL
    filename = os.path.basename(url.split("?")[0])
    if not filename.endswith('.pdf'):
        filename += '.pdf'
    
    local_filename = os.path.join(save_dir, filename)
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        r = requests.get(url, stream=True, headers=headers, timeout=30)
        r.raise_for_status()
        
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        st.sidebar.success(f"Downloaded: {filename}")
        return local_filename
        
    except Exception as e:
        raise Exception(f"Failed to download PDF: {str(e)}")

def setup_retriever(embeddings, pdf_path):
    """Set up the document retriever with proper error handling and caching."""
    
    if not os.path.exists(pdf_path):
        st.sidebar.error("PDF file not found!")
        return None, None
    
    # Create unique identifier for this document
    try:
        unique_id = hash_file(pdf_path)
    except Exception as e:
        st.sidebar.error(f"Error reading PDF file: {e}")
        return None, None
    
    vector_db_dir = os.path.join("chroma_dbs", unique_id)
    os.makedirs("chroma_dbs", exist_ok=True)

    # Try to load existing vector store
    if os.path.exists(vector_db_dir) and len(os.listdir(vector_db_dir)) > 0:
        try:
            vector_store = Chroma(
                persist_directory=vector_db_dir, 
                embedding_function=embeddings
            )
            
            # Test the vector store by checking if it has documents
            if vector_store._collection.count() > 0:
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": RETRIEVAL_K}
                )
                st.sidebar.success(f"✅ Vector DB loaded from cache ({vector_store._collection.count()} chunks)")
                return retriever, vector_store
            else:
                st.sidebar.warning("Cached DB is empty, rebuilding...")
                
        except Exception as e:
            st.sidebar.error(f"🚨 Failed to load existing DB: {e}. Rebuilding...")

    # Create new vector store
    try:
        with st.spinner("Processing PDF and creating vector store..."):
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            if not documents:
                st.sidebar.error("No content found in PDF!")
                return None, None
            
            st.sidebar.info(f"Loaded {len(documents)} pages from PDF")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            texts = text_splitter.split_documents(documents)
            
            if not texts:
                st.sidebar.error("No text chunks created from PDF!")
                return None, None
            
            st.sidebar.info(f"Created {len(texts)} text chunks")
            
            # Create vector store
            vector_store = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=vector_db_dir
            )
            
            # Create retriever with proper configuration
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": RETRIEVAL_K}
            )
            
            # Test the retriever
            test_query = "test"
            try:
                test_results = retriever.invoke(test_query)
                st.sidebar.success(f"✨ New DB created successfully with {len(texts)} chunks")
                if test_results:
                    st.sidebar.success(f"✓ Retriever test passed - found {len(test_results)} results")
                else:
                    st.sidebar.warning("⚠️ Retriever test returned no results")
            except Exception as test_e:
                st.sidebar.error(f"❌ Retriever test failed: {test_e}")
            
            return retriever, vector_store
            
    except Exception as e:
        st.sidebar.error(f"🚨 Error processing PDF: {str(e)}")
        import traceback
        st.sidebar.error(f"Detailed error: {traceback.format_exc()}")

    return None, None