# agent_handler.py - Simplified version that forces document search

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage, AIMessage

# --- 1. Tool Definition ---
def create_document_retrieval_tool(retriever):
    """Factory function to create the document retrieval tool with a given retriever."""
    @tool
    def retrieve_document(query: str) -> str:
        """
        Retrieves relevant information from the loaded PDF document.
        This should be the primary tool to use for any document-related questions.
        """
        if not retriever:
            return "Document retriever is not available."
        
        try:
            # Try multiple retrieval methods
            docs = None
            
            # Method 1: Standard invoke
            try:
                docs = retriever.invoke(query)
            except:
                # Method 2: get_relevant_documents
                try:
                    docs = retriever.get_relevant_documents(query)
                except:
                    # Method 3: Direct similarity search
                    if hasattr(retriever, 'vectorstore'):
                        docs = retriever.vectorstore.similarity_search(query, k=6)
            
            if docs and len(docs) > 0:
                # Filter out empty documents
                valid_docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
                
                if valid_docs:
                    content_parts = []
                    for i, doc in enumerate(valid_docs[:6], 1):  # Limit to 6 docs
                        content_parts.append(f"Document {i}:\n{doc.page_content.strip()}")
                    
                    combined_content = "\n\n---\n\n".join(content_parts)
                    return f"Found {len(valid_docs)} relevant document sections:\n\n{combined_content}"
                else:
                    return "No relevant information was found in the document for that query."
            else:
                return "No relevant information was found in the document for that query."
                
        except Exception as e:
            return f"Error during document retrieval: {str(e)}"
            
    return retrieve_document

# --- 2. Simplified Agent Logic ---
def get_agent_response(llm, tools, chat_history, user_input):
    """
    Simplified agent logic that forces document search first when tools are available.
    """
    
    # Check if we have the document tool
    doc_tool = None
    for tool in tools:
        if tool.name == "retrieve_document":
            doc_tool = tool
            break
    
    # If we have a document tool, use it first
    if doc_tool:
        try:
            # First, search the document
            doc_result = doc_tool.invoke({"query": user_input})
            
            # If we found relevant information, use it
            if doc_result and "No relevant information was found" not in doc_result:
                # Now ask the LLM to formulate an answer based on the document
                context_prompt = f"""Based on the following information from the document, please answer the user's question: "{user_input}"

Document Information:
{doc_result}

Please provide a comprehensive answer based on this document content."""

                response = llm.invoke([HumanMessage(content=context_prompt)])
                return response.content, "Document"
            else:
                # Document didn't have relevant info, use general knowledge
                try:
                    response = llm.invoke([HumanMessage(content=user_input)])
                    return response.content, "General Knowledge"
                except Exception as e:
                    return f"I'm having trouble processing your request: {str(e)}", "Error"
                
        except Exception as e:
            # If document search fails, fall back to general knowledge
            print(f"Document search error: {e}")
            try:
                response = llm.invoke([HumanMessage(content=user_input)])
                return f"I encountered an issue accessing the document, but I can answer from general knowledge:\n\n{response.content}", "General Knowledge"
            except Exception as llm_error:
                return f"I'm experiencing technical difficulties: {str(llm_error)}", "Error"
    
    else:
        # No document tool available, use general knowledge
        try:
            response = llm.invoke([HumanMessage(content=user_input)])
            return response.content, "General Knowledge"
        except Exception as e:
            return f"I'm having trouble processing your request: {str(e)}", "Error"