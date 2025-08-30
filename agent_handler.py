# agent_handler.py - Much simpler approach

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# --- 1. Tool Definition ---
def create_document_retrieval_tool(retriever):
    """Factory function to create the document retrieval tool with a given retriever."""
    @tool
    def retrieve_document(query: str) -> str:
        """
        Retrieves relevant information from the loaded PDF document.
        """
        if not retriever:
            return ""
        
        try:
            docs = retriever.invoke(query)
            
            if docs and len(docs) > 0:
                # Simple relevance check: look for query keywords in documents
                query_words = set(query.lower().split())
                
                relevant_content = []
                for doc in docs[:5]:  # Check top 5 documents
                    if doc.page_content and doc.page_content.strip():
                        doc_lower = doc.page_content.lower()
                        
                        # Check if document contains any significant query words
                        word_matches = sum(1 for word in query_words if len(word) > 3 and word in doc_lower)
                        
                        if word_matches > 0:
                            relevant_content.append(doc.page_content.strip())
                
                if relevant_content:
                    return "\n\n---\n\n".join(relevant_content[:3])  # Return top 3 relevant sections
                else:
                    return ""  # No relevant content found
            else:
                return ""
                
        except Exception as e:
            print(f"Document retrieval error: {e}")
            return ""
            
    return retrieve_document

# --- 2. Very Simple Agent Logic ---
def get_agent_response(llm, tools, chat_history, user_input):
    """
    Very simple logic: Try document first, if nothing relevant found, use general knowledge.
    """
    
    # Check if we have document retrieval tool
    doc_tool = None
    for tool in tools:
        if tool.name == "retrieve_document":
            doc_tool = tool
            break
    
    # If we have document tool, try it first
    if doc_tool:
        try:
            doc_content = doc_tool.invoke({"query": user_input})
            
            # If we got actual content back (not None), use it
            if doc_content:
                prompt = f"""Answer this question based on the document content provided: {user_input}

Document content:
{doc_content}

Provide a comprehensive answer based on this information."""
                
                response = llm.invoke([HumanMessage(content=prompt)])
                return response.content, "Document"
            
        except Exception as e:
            print(f"Document search failed: {e}")
    
    # Either no document tool, or document had no relevant content, use general knowledge
    try:
        response = llm.invoke([HumanMessage(content=user_input)])
        return response.content, "General Knowledge"
    except Exception as e:
        return f"I'm having trouble processing your request: {str(e)}", "Error"