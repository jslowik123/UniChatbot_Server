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

    def start_bot(self) -> Dict[str, str]:
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
        Process a user message and return a structured response from the CrewAI agent.
        
        Args:
            user_input: The user's question or message
            namespace: Namespace to search within for relevant documents
            
        Returns:
            Structured response from the agent
        """
        try:
            # Validate inputs
            if not user_input or not isinstance(user_input, str):
                return {
                    "answer": "Fehler: Leere oder ungültige Eingabe.",
                    "document_ids": [],
                    "sources": [],
                    "confidence_score": 0.0,
                    "context_used": False,
                    "additional_info": "Ungültige Eingabe"
                }
            
            if not namespace or not isinstance(namespace, str):
                return {
                    "answer": "Fehler: Ungültiger Namespace.",
                    "document_ids": [],
                    "sources": [],
                    "confidence_score": 0.0,
                    "context_used": False,
                    "additional_info": "Ungültiger Namespace"
                }
            
            user_input = user_input.strip()
            
            # Get or initialize chat history for this namespace
            if namespace not in self._chat_histories:
                self._chat_histories[namespace] = []
            
            chat_history = self._chat_histories[namespace]
            
            # Use AgentProcessor to answer the question with chat history
            structured_response = self._agent_processor.answer_question(
                user_input, namespace, chat_history
            )
            
            # Update chat history
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": structured_response["answer"]})
            
            # Keep only last 20 messages to prevent memory overflow
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]
                self._chat_histories[namespace] = chat_history
            
            return structured_response
            
        except Exception as e:
            error_msg = f"Fehler beim Verarbeiten der Nachricht: {str(e)}"
            print(error_msg)
            return {
                "answer": error_msg,
                "document_ids": [],
                "sources": [],
                "confidence_score": 0.0,
                "context_used": False,
                "additional_info": f"Exception: {type(e).__name__}"
            }

    def message_bot_stream(self, user_input: str, namespace: str) -> Generator[str, None, None]:
        """
        Process a user message and return a streaming response.
        
        Args:
            user_input: The user's question or message
            namespace: Namespace to search within for relevant documents
            
        Yields:
            Response chunks as they are generated
        """
        try:
            # For streaming, we'll get the structured response first
            # and then stream it in a user-friendly way
            
            yield "🤔 Analysiere deine Frage..."
            
            # Get the structured response from the agent
            structured_response = self.message_bot(user_input, namespace)
            
            # Check if it's an error
            if structured_response["confidence_score"] == 0.0 and "Fehler" in structured_response["answer"]:
                yield f"\n\n❌ {structured_response['answer']}\n"
                if structured_response["additional_info"]:
                    yield f"ℹ️ {structured_response['additional_info']}"
                return
            
            yield "\n\n📚 Durchsuche Dokumente..."
            
            # Show document info if available
            if structured_response["document_ids"]:
                doc_count = len(structured_response["document_ids"])
                yield f"\n📄 {doc_count} relevante Dokument(e) gefunden"
            
            yield "\n\n💡 Formuliere Antwort...\n\n"
            
            # Stream the main answer
            answer = structured_response["answer"]
            sentences = answer.split('. ')
            
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    if i < len(sentences) - 1:
                        yield sentence + '. '
                    else:
                        yield sentence
                    
                    # Add small delay effect through yielding empty string
                    yield ""
            
            # Add metadata information at the end
            yield "\n\n"
            
            # Show sources if available
            if structured_response["sources"]:
                yield "📖 **Quellen:**\n"
                for i, source in enumerate(structured_response["sources"][:3], 1):  # Limit to 3 sources
                    # Truncate long sources
                    truncated_source = source[:200] + "..." if len(source) > 200 else source
                    yield f"{i}. {truncated_source}\n"
            
            # Show confidence and additional info
            confidence = structured_response["confidence_score"]
            confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
            yield f"\n{confidence_emoji} **Vertrauen:** {confidence:.1%}"
            
            if structured_response["context_used"]:
                yield " | 🔄 **Chat-Kontext verwendet**"
            
            if structured_response["additional_info"]:
                yield f"\nℹ️ **Hinweis:** {structured_response['additional_info']}"
            
        except Exception as e:
            error_msg = f"Fehler beim Streaming: {str(e)}"
            print(error_msg)
            yield f"\n\n❌ {error_msg}"

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


def message_bot(user_input: str, context: str, knowledge: str, database_overview: Any, 
               document_id: str, chat_history: List[Dict[str, str]], namespace: str = "default") -> str:
    """
    Legacy compatibility function for the existing API.
    
    Args:
        user_input: User's question
        context: Document context (ignored in agent-based approach)
        knowledge: Additional knowledge (ignored in agent-based approach)
        database_overview: Database overview (ignored in agent-based approach)
        document_id: Document ID (ignored in agent-based approach)
        chat_history: Chat history (used for context)
        namespace: Namespace to search within
        
    Returns:
        Agent's structured response as JSON string or plain answer
    """
    try:
        chatbot = get_bot()
        
        # Set chat history for this namespace if provided
        if chat_history and namespace:
            chatbot._chat_histories[namespace] = chat_history
        
        structured_response = chatbot.message_bot(user_input, namespace)
        
        # For legacy compatibility, return just the answer
        # But also include structured data as JSON comment if needed
        return structured_response["answer"]
        
    except Exception as e:
        return f"Fehler: {str(e)}"


def message_bot_structured(user_input: str, namespace: str = "default", 
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
            "additional_info": f"Exception: {type(e).__name__}"
        }


def message_bot_stream(user_input: str, context: str, knowledge: str, database_overview: Any, 
                      document_id: str, chat_history: List[Dict[str, str]], namespace: str = "default") -> Generator[str, None, None]:
    """
    Legacy compatibility function for streaming responses.
    
    Args:
        user_input: User's question
        context: Document context (ignored in agent-based approach)
        knowledge: Additional knowledge (ignored in agent-based approach)
        database_overview: Database overview (ignored in agent-based approach)
        document_id: Document ID (ignored in agent-based approach)
        chat_history: Chat history (used for context)
        namespace: Namespace to search within
        
    Yields:
        Response chunks
    """
    try:
        chatbot = get_bot()
        
        # Set chat history for this namespace if provided
        if chat_history and namespace:
            chatbot._chat_histories[namespace] = chat_history
        
        yield from chatbot.message_bot_stream(user_input, namespace)
    except Exception as e:
        yield f"Fehler: {str(e)}" 