import os
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
import logging
from firebase_connection import get_firebase_client
from agent_processor import AgentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentChatbot:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.firebase_db = get_firebase_client()
        self.agent_processor = AgentProcessor()

    def get_chat_history(self, namespace: str, limit: int = 10) -> List[Dict[str, str]]:
        """Retrieve chat history from Firebase for the given namespace"""
        try:
            chat_ref = self.firebase_db.collection('chat_history').document(namespace)
            doc = chat_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                messages = data.get('messages', [])
                return messages[-limit:] if len(messages) > limit else messages
            return []
        except Exception as e:
            logger.error(f"Error retrieving chat history: {str(e)}")
            return []

    def save_chat_message(self, namespace: str, role: str, content: str):
        """Save a chat message to Firebase"""
        try:
            chat_ref = self.firebase_db.collection('chat_history').document(namespace)
            doc = chat_ref.get()
            
            new_message = {
                "role": role,
                "content": content,
                "timestamp": firestore.SERVER_TIMESTAMP
            }
            
            if doc.exists:
                chat_ref.update({
                    "messages": firestore.ArrayUnion([new_message])
                })
            else:
                chat_ref.set({
                    "messages": [new_message],
                    "created_at": firestore.SERVER_TIMESTAMP
                })
        except Exception as e:
            logger.error(f"Error saving chat message: {str(e)}")

    def send_message(self, message: str, namespace: str = "default") -> str:
        """
        Send a message and get a simple text response
        
        Args:
            message: The user's message
            namespace: The namespace for document search
            
        Returns:
            Simple text response
        """
        try:
            # Get relevant documents
            relevant_docs = self.agent_processor.query_documents(message, namespace)
            
            # Get chat history for context
            chat_history = self.get_chat_history(namespace)
            
            # Generate response
            response_data = self.agent_processor.generate_response(
                message, relevant_docs, chat_history
            )
            
            # Save the conversation
            self.save_chat_message(namespace, "user", message)
            self.save_chat_message(namespace, "assistant", response_data["answer"])
            
            return response_data["answer"]
            
        except Exception as e:
            logger.error(f"Error in send_message: {str(e)}")
            return "I apologize, but I encountered an error while processing your message. Please try again."

    def send_message_structured(self, message: str, namespace: str = "default") -> Dict[str, Any]:
        """
        Send a message and get a structured response with metadata
        
        Args:
            message: The user's message
            namespace: The namespace for document search
            
        Returns:
            Structured response with answer, sources, confidence, etc.
        """
        try:
            # Get relevant documents
            relevant_docs = self.agent_processor.query_documents(message, namespace)
            
            # Get chat history for context
            chat_history = self.get_chat_history(namespace)
            
            # Generate response
            response_data = self.agent_processor.generate_response(
                message, relevant_docs, chat_history
            )
            
            # Save the conversation
            self.save_chat_message(namespace, "user", message)
            self.save_chat_message(namespace, "assistant", response_data["answer"])
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error in send_message_structured: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while processing your message. Please try again.",
                "document_ids": [],
                "sources": [],
                "confidence_score": 0.0,
                "context_used": False,
                "additional_info": "Error occurred during processing"
            }

    def send_message_stream(self, message: str, namespace: str = "default"):
        """
        Send a message and stream the response with metadata
        
        Args:
            message: The user's message
            namespace: The namespace for document search
            
        Yields:
            Streaming response chunks with metadata
        """
        try:
            # Get relevant documents
            relevant_docs = self.agent_processor.query_documents(message, namespace)
            
            # Get chat history for context
            chat_history = self.get_chat_history(namespace)
            
            # Prepare context
            context = "\n\n".join([
                f"Document: {doc['document_name']}\nContent: {doc['text']}"
                for doc in relevant_docs
            ])
            
            # Prepare chat history
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a helpful AI assistant that answers questions based on the provided documents. 
                    Use the following context to answer the user's question. If you cannot find the answer in the context, 
                    say so clearly. Be concise but comprehensive.
                    
                    Context:
                    {context}"""
                }
            ]
            
            # Add chat history if available
            if chat_history:
                messages.extend(chat_history[-5:])  # Last 5 messages for context
            
            messages.append({"role": "user", "content": message})
            
            # Stream response
            response_stream = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                stream=True
            )
            
            # First, send metadata
            confidence_score = min(0.9, sum(doc['score'] for doc in relevant_docs[:3]) / 3) if relevant_docs else 0.3
            metadata = {
                "type": "metadata",
                "document_ids": list(set(doc["document_id"] for doc in relevant_docs)),
                "sources": [doc["text"][:200] + "..." for doc in relevant_docs[:3]],
                "confidence_score": round(confidence_score, 2),
                "context_used": len(relevant_docs) > 0,
                "additional_info": f"Based on {len(relevant_docs)} relevant document sections"
            }
            yield f"data: {json.dumps(metadata)}\n\n"
            
            # Then stream the answer
            full_response = ""
            for chunk in response_stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    chunk_data = {"type": "content", "content": content}
                    yield f"data: {json.dumps(chunk_data)}\n\n"
            
            # Save the conversation
            self.save_chat_message(namespace, "user", message)
            self.save_chat_message(namespace, "assistant", full_response)
            
            # Send end marker
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
            
        except Exception as e:
            logger.error(f"Error in send_message_stream: {str(e)}")
            error_data = {
                "type": "error",
                "content": "I apologize, but I encountered an error while processing your message. Please try again."
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    def clear_chat_history(self, namespace: str = "default") -> bool:
        """Clear chat history for a namespace"""
        try:
            chat_ref = self.firebase_db.collection('chat_history').document(namespace)
            chat_ref.delete()
            return True
        except Exception as e:
            logger.error(f"Error clearing chat history: {str(e)}")
            return False

# Legacy compatibility functions
def send_message(message: str, namespace: str = "default") -> str:
    """Legacy function for backward compatibility"""
    chatbot = AgentChatbot()
    return chatbot.send_message(message, namespace)

def send_message_structured(message: str, namespace: str = "default") -> Dict[str, Any]:
    """Legacy function for backward compatibility"""
    chatbot = AgentChatbot()
    return chatbot.send_message_structured(message, namespace)

def send_message_stream(message: str, namespace: str = "default"):
    """Legacy function for backward compatibility"""
    chatbot = AgentChatbot()
    return chatbot.send_message_stream(message, namespace)

# Import firestore for timestamp
try:
    from google.cloud import firestore
except ImportError:
    # Fallback if not available
    firestore = None 