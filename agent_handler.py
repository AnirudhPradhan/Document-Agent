# agent_handler.py - Improved with better prompts

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
                # For summary/gist requests, return more content
                is_summary_request = any(word in query.lower() for word in 
                                       ['summary', 'summarize', 'gist', 'overview', 'about', 'describe'])
                
                if is_summary_request:
                    # Return more content for summary requests
                    content_parts = []
                    for doc in docs[:8]:  # More docs for summary
                        if doc.page_content and doc.page_content.strip():
                            content_parts.append(doc.page_content.strip())
                    
                    if content_parts:
                        return "\n\n---\n\n".join(content_parts)
                    else:
                        return ""
                
                else:
                    # For specific questions, do relevance checking
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

# --- 2. Helper Functions for Better Prompts ---
def is_summary_question(query):
    """Check if the query is asking for a summary or overview."""
    summary_keywords = [
        'summary', 'summarize', 'summarise', 'gist', 'overview', 'about',
        'describe', 'explain', 'tell me about', 'what is this document',
        'main points', 'key points', 'brief', 'outline'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in summary_keywords)

def is_specific_question(query):
    """Check if the query is asking for specific information."""
    question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which']
    query_lower = query.lower()
    return any(query_lower.startswith(word) for word in question_words) or '?' in query

def create_prompt(user_input, doc_content, query_type="specific"):
    """Create appropriate prompts based on query type."""
    
    if query_type == "summary":
        return f"""Based on the document content provided below, create a comprehensive summary for the user's request: "{user_input}"

Document content:
{doc_content}

Please provide:
1. A clear overview of what this document is about
2. The main topics or themes covered
3. Key findings, methods, or important points
4. Any significant details that would help someone understand the document

Make your response informative and well-structured."""

    elif query_type == "specific":
        return f"""Answer the specific question: "{user_input}"

Use the following document content to provide a detailed and accurate answer:

Document content:
{doc_content}

Instructions:
- Answer the question directly and specifically
- Use information from the document content provided
- If the document mentions specific details, numbers, or examples related to the question, include them
- Be precise and factual
- If the question has multiple parts, address each part"""

    else:  # general
        return f"""Based on the document content provided, please respond to: "{user_input}"

Document content:
{doc_content}

Provide a helpful and comprehensive response based on this information."""

# --- 3. Improved Agent Logic ---
def get_agent_response(llm, tools, chat_history, user_input):
    """
    Improved logic with better prompts for different types of questions.
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
            
            # If we got actual content back (not empty string), use it
            if doc_content and doc_content.strip():
                
                # Determine the type of query and create appropriate prompt
                if is_summary_question(user_input):
                    prompt = create_prompt(user_input, doc_content, "summary")
                elif is_specific_question(user_input):
                    prompt = create_prompt(user_input, doc_content, "specific")
                else:
                    prompt = create_prompt(user_input, doc_content, "general")
                
                response = llm.invoke([HumanMessage(content=prompt)])
                return response.content, "Document"
            
        except Exception as e:
            print(f"Document search failed: {e}")
    
    # Either no document tool, or document had no relevant content, use general knowledge
    try:
        # Even for general knowledge, provide better prompts
        if is_summary_question(user_input):
            general_prompt = f"""The user is asking: "{user_input}"
            
Since no relevant document is available, please explain that you would need to see the specific document they're referring to in order to provide a summary. However, if they're asking about a general topic, provide helpful information about that topic."""
        
        else:
            general_prompt = user_input
        
        response = llm.invoke([HumanMessage(content=general_prompt)])
        return response.content, "General Knowledge"
        
    except Exception as e:
        return f"I'm having trouble processing your request: {str(e)}", "Error"