# vector_store_handler.py

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import configuration from config.py
from config import PDF_PATH, VECTOR_DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP

def setup_retriever(embeddings):
    """
    Sets up the Chroma vector store and returns a retriever object.
    
    It first checks if a persisted database exists. If so, it loads it.
    If not, it processes the PDF, creates embeddings, and persists the DB.

    Args:
        embeddings: The embedding function to use.

    Returns:
        A retriever object or None if an error occurs.
    """
    if os.path.exists(VECTOR_DB_PATH):
        try:
            vector_store = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=embeddings
            )
            st.sidebar.success(f"Chroma DB loaded successfully from {VECTOR_DB_PATH}", icon="✅")
            return vector_store.as_retriever()
        except Exception as e:
            st.sidebar.error(f"Error loading persisted Chroma DB: {e}. Re-creating.", icon="🚨")
            # Fall through to re-create the DB if loading fails

    if not os.path.exists(PDF_PATH):
        st.sidebar.error(f"PDF file not found at {PDF_PATH}.", icon="🚨")
        return None
    
    try:
        with st.spinner("Processing PDF and creating new vector store..."):
            loader = PyPDFLoader(PDF_PATH)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            texts = text_splitter.split_documents(documents)
            
            vector_store = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=VECTOR_DB_PATH
            )
        st.sidebar.success(f"New Chroma DB created at {VECTOR_DB_PATH}", icon="✨")
        return vector_store.as_retriever()
    except Exception as e:
        st.sidebar.error(f"Error processing PDF: {e}", icon="🚨")
        return None