# app.py

import os
import streamlit as st
from dotenv import load_dotenv
import asyncio

# Import from our custom modules
import config
from vector_store_handler import setup_retriever
from agent_handler import create_document_retrieval_tool, get_agent_response
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
# --- Fix for asyncio error in Streamlit ---
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
# -----------------------------------------

# --- 1. App Configuration and Initialization ---

st.set_page_config(page_title="Doc Agent", page_icon="🤖", layout="wide")
st.title("🤖 Chat with Your Document")
st.markdown("This agent uses a PDF document as its primary knowledge source.")

load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Please set the GOOGLE_API_KEY environment variable in your .env file.")
    st.stop()

@st.cache_resource
def load_models_and_retriever():
    """Loads LLM, embeddings, and sets up the retriever."""
    try:
        llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL_NAME, temperature=0.2)
        embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDINGS_MODEL_NAME)
        retriever = setup_retriever(embeddings)
        return llm, retriever
    except Exception as e:
        st.error(f"Error initializing models or retriever: {e}")
        return None, None

llm, retriever = load_models_and_retriever()

# --- 2. Session State Management ---

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        # Using a dictionary to store role, content, and the source of the response
        {
            "role": "ai",
            "content": "Hello! Ask me anything about the provided document.",
            "source": None # No source for the initial greeting
        }
    ]

# --- 3. Main Application Logic ---

if not llm or not retriever:
    st.warning("Application is not fully initialized. Please check your configuration and API keys.")
else:
    doc_retrieval_tool = create_document_retrieval_tool(retriever)
    tools = [doc_retrieval_tool]

    # Display previous chat messages from the dictionary-based history
    for message in st.session_state.chat_history:
        chat_role = "AI" if message["role"] == "ai" else "Human"
        with st.chat_message(chat_role):
            # For AI messages, display the source if it's available
            if message["role"] == "ai" and message.get("source"):
                source = message["source"]
                icon = "📚" if source == "Document" else "🌐"
                st.markdown(f"<small>{icon} **Source:** {source}</small>", unsafe_allow_html=True)
            
            st.write(message["content"])

    # Get user input from chat interface
    user_query = st.chat_input("Ask a question about the document...")
    if user_query:
        # Append user message to history and display it
        user_message_data = {"role": "human", "content": user_query}
        st.session_state.chat_history.append(user_message_data)
        with st.chat_message("Human"):
            st.write(user_query)

        # Generate and display AI response
        with st.chat_message("AI"):
            with st.spinner("Thinking..."):
                # The agent function now returns the response and its source
                response_content, source = get_agent_response(llm, tools, st.session_state.chat_history, user_query)
                
                # Display the source icon and text
                icon = "📚" if source == "Document" else "🌐"
                st.markdown(f"<small>{icon} **Source:** {source}</small>", unsafe_allow_html=True)
                
                # Display the actual response content
                st.write(response_content)
        
        # Append the full AI response data (content and source) to the history
        ai_message_data = {"role": "ai", "content": response_content, "source": source}
        st.session_state.chat_history.append(ai_message_data)