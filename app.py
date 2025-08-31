import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
import config
from vector_store_handler import setup_retriever, download_pdf_from_url
from agent_handler import create_document_retrieval_tool, get_agent_response
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

st.set_page_config(page_title="PDF Agent", page_icon="🤖", layout="wide")
st.title("🤖 Chat with Your Document")
st.markdown("This agent uses an uploaded PDF, PDF URL, or general LLM if none provided.")

# Load environment variables
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    st.error("GOOGLE_API_KEY not set in .env file.")
    st.stop()

# Sidebar for file upload/URL input
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
pdf_url = st.sidebar.text_input("...or enter an URL")
pdf_path = None

# Handle PDF input
if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name
    st.sidebar.success(f"PDF uploaded: {uploaded_pdf.name}")
elif pdf_url:
    try:
        pdf_path = download_pdf_from_url(pdf_url)
    except Exception as e:
        st.sidebar.error(f"Failed to download PDF: {e}")
        pdf_path = None

@st.cache_resource
def load_llm_and_embeddings():
    """Load and cache the LLM and embeddings models."""
    try:
        llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL_NAME, 
            temperature=0.2
        )
        embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDINGS_MODEL_NAME)
        return llm, embeddings
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

# Load models
llm, embeddings = load_llm_and_embeddings()

# Setup retriever and tools
retriever, vector_store = None, None
tools = []

if pdf_path:
    retriever, vector_store = setup_retriever(embeddings, pdf_path)
    if retriever:
        doc_retrieval_tool = create_document_retrieval_tool(retriever)
        tools = [doc_retrieval_tool]
        st.sidebar.success("✅ Document loaded and ready for queries!")
        
        # Add debug test button
        if st.sidebar.button("🔍 Test Document Retrieval"):
            with st.spinner("Testing retriever..."):
                test_query = "JTubeSpeech"
                try:
                    docs = retriever.invoke(test_query)
                    if docs:
                        st.sidebar.success(f"✅ Retriever works! Found {len(docs)} results")
                        with st.sidebar.expander("First Result Preview"):
                            st.write(docs[0].page_content[:300] + "...")
                    else:
                        st.sidebar.error("❌ Retriever returned no results")
                except Exception as e:
                    st.sidebar.error(f"❌ Retriever failed: {e}")
    else:
        st.sidebar.error("❌ Failed to setup document retriever")
else:
    st.sidebar.info("📝 Upload a PDF or provide URL to enable document chat")

# Initialize chat history
if "chat_history" not in st.session_state:
    initial_message = "Hello! " + (
        "Ask me anything about the uploaded document, or use me for general Q&A." 
        if tools else 
        "Upload a PDF to chat with your document, or ask me general questions."
    )
    st.session_state.chat_history = [
        {"role": "ai", "content": initial_message, "source": None}
    ]

# Display chat history
for message in st.session_state.chat_history:
    chat_role = "assistant" if message["role"] == "ai" else "user"
    with st.chat_message(chat_role):
        if message["role"] == "ai" and message.get("source"):
            icon = "📚" if message["source"] == "Document" else "🌍"
            st.markdown(f"{icon} **Source:** {message['source']}")
        st.write(message["content"])

# Handle user input
user_query = st.chat_input("Ask a question about the document (or anything else)")
if user_query:
    # Add user message to history
    st.session_state.chat_history.append({"role": "human", "content": user_query})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_query)
    
    # Generate and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response_content, source = get_agent_response(
                    llm, tools, st.session_state.chat_history, user_query
                )
                
                # Display source indicator
                if source:
                    icon = "📚" if source == "Document" else "🌍"
                    st.markdown(f"{icon} **Source:** {source}")
                
                # Display response
                st.write(response_content)
                
                # Add AI response to history
                st.session_state.chat_history.append({
                    "role": "ai", 
                    "content": response_content, 
                    "source": source
                })
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({
                    "role": "ai", 
                    "content": error_msg, 
                    "source": "Error"
                })

# Debug information (optional - can be removed in production)
if st.sidebar.checkbox("Show Debug Info", False):
    st.sidebar.markdown("### Debug Information")
    st.sidebar.write(f"PDF Path: {pdf_path}")
    st.sidebar.write(f"Tools Available: {len(tools)}")
    st.sidebar.write(f"Retriever Available: {retriever is not None}")
    if vector_store:
        try:
            doc_count = vector_store._collection.count()
            st.sidebar.write(f"Documents in Vector Store: {doc_count}")
        except:
            st.sidebar.write("Vector Store: Status Unknown")