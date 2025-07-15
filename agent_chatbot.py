from agent_processor import AgentProcessor
import os
import json
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional, Generator

# Load environment variables once at module level
load_dotenv()


class AgentChatbot:
    """
    CrewAI-based chatbot that uses agentic RAG for intelligent question answering.
    
    Provides both regular and streaming responses using the AgentProcessor's
    CrewAI agents for sophisticated document-based question answering with
    structured outputs.
    """
    
    def __init__(self):
        """Initialize the AgentChatbot with AgentProcessor."""
        self._agent_processor = AgentProcessor(
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self._chat_histories = {}  # Store chat histories per namespace

    def start_bot_agent(self) -> Dict[str, str]:
        """
        Initialize/restart the chatbot.
        
        Returns:
            Dict containing startup status
        """
        try:
            # Clear chat histories
            self._chat_histories = {}
            return {
                "status": "success",
                "message": "Agent-based bot started successfully"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error starting bot: {str(e)}"
            }
        


    def get_chat_history(self, namespace: str) -> List[Dict[str, str]]:
        """
        Get chat history for a specific namespace.
        
        Args:
            namespace: Namespace identifier
            
        Returns:
            List of chat messages for the namespace
        """
        return self._chat_histories.get(namespace, [])

    def clear_chat_history(self, namespace: str) -> None:
        """
        Clear chat history for a specific namespace.
        
        Args:
            namespace: Namespace identifier
        """
        if namespace in self._chat_histories:
            self._chat_histories[namespace] = []

    def message_bot(self, user_input: str, namespace: str) -> Dict[str, Any]:
        """
        Send a message to the agent and get a structured response.
        
        Args:
            user_input: User's question
            namespace: Namespace to search within
            
        Returns:
            Structured response from the agent
        """
        chat_history = self.get_chat_history(namespace)
        return self._agent_processor.answer_question(user_input, namespace, chat_history)

    async def stream_message(self, user_input: str, namespace: str, history: list) -> Generator[str, Any, None]:
        """
        Stream the response from the agent.
        
        Args:
            user_input: User's question
            namespace: Namespace to search within
            history: Chat history for context
            
        Yields:
            The agent's response as a single chunk.
        """
        response = self.message_bot(user_input, namespace)
        yield response.get("answer", "No answer found.")

    def get_namespace_info(self, namespace: str) -> Dict[str, Any]:
        """
        Get information about documents in a namespace.
        
        Args:
            namespace: Namespace identifier
            
        Returns:
            Dict containing namespace information
        """
        try:
            return self._agent_processor.get_namespace_summary(namespace)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error getting namespace info: {str(e)}"
            }


# Global instance for compatibility with existing code
_global_chatbot = None


def get_bot() -> AgentChatbot:
    """
    Get or create the global chatbot instance.
    
    Returns:
        AgentChatbot instance
    """
    global _global_chatbot
    if _global_chatbot is None:
        _global_chatbot = AgentChatbot()
    return _global_chatbot



def message_bot_agent(user_input: str, namespace: str = "default", 
                          chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    New function that returns the full structured response.
    
    Args:
        user_input: User's question
        namespace: Namespace to search within
        chat_history: Chat history for context
        
    Returns:
        Full structured response from agent
    """
    try:
        chatbot = get_bot()
        
        # Set chat history for this namespace if provided
        if chat_history and namespace:
            chatbot._chat_histories[namespace] = chat_history
        
        return chatbot.message_bot(user_input, namespace)
        
    except Exception as e:
        return {
            "answer": f"Fehler: {str(e)}",
            "document_ids": [],
            "sources": [],
            "confidence_score": 0.0,
            "context_used": False,
            "additional_info": f"Exception: {type(e).__name__}",
            "pages": []
        }

