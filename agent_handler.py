# agent_handler.py

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage, AIMessage

# --- 1. Tool Definition ---
def create_document_retrieval_tool(retriever):
    """Factory function to create the document retrieval tool with a given retriever."""
    @tool
    def retrieve_document(query: str) -> str:
        """
        Retrieves relevant information from the loaded PDF document about NaijaTTS.
        This should be the primary tool to use for any document-related questions.
        """
        if not retriever:
            return "Document retriever is not available."
        
        try:
            docs = retriever.invoke(query)
            if docs:
                return "\n---\n".join([doc.page_content for doc in docs])
            else:
                return "No relevant information was found in the document for that query."
        except Exception as e:
            return f"An error occurred during retrieval: {e}"
            
    return retrieve_document

# --- 2. Agent Logic ---
def get_agent_response(llm, tools, chat_history, user_input):
    """
    Runs the conversational agent logic for one turn.
    Returns a tuple of (response_content, source).
    'source' will be 'Document' or 'General Knowledge'.
    """
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {tool.name: tool for tool in tools}
    tool_used = False # Flag to track if the document retriever was used this turn

    system_message = SystemMessage(
        content="You are a helpful assistant. Your primary goal is to answer questions about the NaijaTTS document using the 'retrieve_document' tool. "
                "Always try to use this tool first. If the tool returns 'No relevant information was found', or if the user's question is clearly not about the document "
                "(e.g., 'what is the capital of France?'), then you must answer directly using your own general knowledge. "
                "Do not mention that you are using your own knowledge; just provide the answer."
    )
    
    # Convert our dict-based history to LangChain messages for the model
    messages_for_turn = []
    for msg in chat_history:
        if msg.get("role") == "human":
            messages_for_turn.append(HumanMessage(content=msg["content"]))
        elif msg.get("role") == "ai":
            # Don't include the source in the history passed to the model
            messages_for_turn.append(AIMessage(content=msg["content"]))

    # The agent loop continues until the LLM provides a direct answer
    while True:
        ai_response = llm_with_tools.invoke([system_message] + messages_for_turn)
        messages_for_turn.append(ai_response)
        
        if not ai_response.tool_calls:
            # If no tool was called, the source is general knowledge, unless a tool
            # was used in a previous step of this same turn.
            source = "Document" if tool_used else "General Knowledge"
            return ai_response.content, source

        # If there are tool calls, execute them and continue the loop
        tool_used = True # Mark that a tool has been called
        for tool_call in ai_response.tool_calls:
            tool_name = tool_call["name"]
            tool_to_call = tool_map.get(tool_name)
            
            if tool_to_call:
                tool_output = tool_to_call.invoke(tool_call["args"])
                messages_for_turn.append(
                    ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                )
            else:
                error_message = f"Error: Tool '{tool_name}' not found."
                messages_for_turn.append(
                    ToolMessage(content=error_message, tool_call_id=tool_call["id"])
                )