from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import json
from dotenv import load_dotenv
from pinecone_connection import PineconeCon

# Load environment variables once at module level
load_dotenv()


def _get_openai_client(streaming: bool = False) -> ChatOpenAI:
    """
    Creates and returns a configured OpenAI client.
    
    Args:
        streaming: Whether to enable streaming for real-time responses
        
    Returns:
        ChatOpenAI: Configured language model client
        
    Raises:
        ValueError: If OpenAI API key is not found
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    return ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=api_key,
        streaming=streaming
    )

def _create_context_aware_prompt(has_chat_history=False):
    """
    Creates a context-aware prompt template that emphasizes chat history when present.
    
    Args:
        has_chat_history: Whether chat history is present
        
    Returns:
        ChatPromptTemplate: Configured prompt template
    """
    base_system_prompt = """Du bist ein sachlicher, präziser und hilfreicher Assistenz-Chatbot für eine Universität."""
    
    if has_chat_history:
        chat_history_emphasis = """

🔄 CHAT HISTORY BEACHTUNG - WICHTIG:
Du siehst eine Chat History mit vorherigen Nachrichten. BERÜCKSICHTIGE diese aktiv:
- Beziehe dich auf vorherige Fragen und Antworten
- Nutze den Kontext aus früheren Nachrichten
- Wenn der Nutzer "dazu", "darüber", "das" oder ähnliche Bezugswörter verwendet, beziehe dich auf vorherige Themen
- Beantworte Rückfragen oder Nachfragen basierend auf dem bisherigen Gesprächsverlauf
- Vermeide Wiederholungen bereits gegebener Antworten, es sei denn, es wird explizit verlangt
- Erkenne den Kontext der aktuellen Frage im Zusammenhang mit der Chat History"""
    else:
        chat_history_emphasis = """

📝 NEUE UNTERHALTUNG:
Dies ist der Beginn einer neuen Unterhaltung ohne vorherige Chat History."""

    sources_section = """

DEINE ANTWORTQUELLEN:
Diese Informationen stehen dir zur Verfügung:

HOCHSCHULSPEZIFISCHE INFORMATIONEN:
{context}

ZUSÄTZLICHES WISSEN:
{knowledge}

DOKUMENTEN-ID: {document_id}

VERHALTEN:
- Stütze deine Antworten auf die bereitgestellten Quellen
- Antworte natürlich und direkt, als würdest du mit Studierenden sprechen
- Bei Widersprüchen: Bevorzuge hochschulspezifische Informationen
- Bei fehlenden Informationen: Sage es klar und biete Hilfe an
- Gib ausführliche, aber präzise Antworten
- Verwende innerhalb der "answer" kein "" sondern nur ''

ANTWORTFORMAT:
{{
  "answer": "Deine ausführliche Antwort hier",
  "document_id": "{document_id}",
  "source": "Originaltext aus dem Kontext, der die Antwort stützt"
}}"""

    full_system_prompt = base_system_prompt + chat_history_emphasis + sources_section

    return ChatPromptTemplate.from_messages(
        [
            ("system", full_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )


def get_bot():
    """
    Creates and configures the chatbot chain with prompt template.
    
    Returns:
        Chain: Configured LangChain pipeline for the university chatbot
    """
    llm = _get_openai_client()
    # Use default prompt (this will be overridden in message functions with context-aware prompt)
    prompt_template = _create_context_aware_prompt(has_chat_history=False)
    return prompt_template | llm


def _validate_inputs(user_input, context, knowledge, chat_history):
    """
    Validates and sanitizes input parameters for message processing.
    
    Args:
        user_input: User's message
        context: Document context
        knowledge: General knowledge context
        chat_history: Previous conversation history
        
    Returns:
        tuple: Validated and sanitized inputs
    """
    if not user_input or not isinstance(user_input, str):
        raise ValueError("User input must be a non-empty string")
    
    if not isinstance(chat_history, list):
        chat_history = []
        
    if not context or not isinstance(context, str):
        context = ""

    if not knowledge or not isinstance(knowledge, str):
        knowledge = ""
        
    return user_input.strip(), context, knowledge, chat_history


def _format_chat_history(chat_history):
    """
    Converts chat history to LangChain message format.
    
    Args:
        chat_history: List of chat messages with role and content
        
    Returns:
        list: Formatted chat history for LangChain
    """
    formatted_history = []
    for msg in chat_history:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            continue
            
        if msg["role"] == "user":
            formatted_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted_history.append(AIMessage(content=msg["content"]))
    
    return formatted_history


def message_bot(user_input, context, knowledge, database_overview, document_id, chat_history):
    """
    Processes a user message and returns a response from the chatbot.
    
    Args:
        user_input: The user's question or message
        context: Relevant document context from vector search
        knowledge: General knowledge context
        document_id: ID of the document being referenced
        chat_history: Previous conversation history
        
    Returns:
        str: The chatbot's response
    """
    try:
        user_input, context, knowledge, chat_history = _validate_inputs(
            user_input, context, knowledge, chat_history
        )
        
        formatted_history = _format_chat_history(chat_history)
        
        # Create context-aware chain that emphasizes chat history when present
        llm = _get_openai_client()
        prompt_template = _create_context_aware_prompt(bool(chat_history))
        chain = prompt_template | llm
        
        # Debug: Zeige welche Parameter tatsächlich an den Chain weitergegeben werden
        chain_params = {
            "input": user_input,
            "context": context,
            "knowledge": knowledge,
            "document_id": document_id,
            "chat_history": formatted_history,
        }
        
        print("CHAIN PARAMETER:")
        print(f"Input: {chain_params['input']}")
        print(f"Context (erste 300 Zeichen): {chain_params['context'][:300] if chain_params['context'] else 'LEER'}...")
        print(f"Knowledge (erste 200 Zeichen): {chain_params['knowledge'][:200] if chain_params['knowledge'] else 'LEER'}...")
        print(f"Document ID: {chain_params['document_id']}")
        print(f"Chat History Länge: {len(chain_params['chat_history'])}")
        print("\n")
        
        response = chain.invoke(chain_params)
        
        if not response or not hasattr(response, 'content'):
            return "Entschuldigung, ich konnte keine Antwort generieren."
            
        print("AI RESPONSE:", response.content)
        return response.content
        
    except ValueError as ve:
        print(f"Validation error in message_bot: {str(ve)}")
        return "Entschuldigung, ich konnte Ihre Nachricht nicht verstehen."
    except Exception as e:
        print(f"Error in message_bot: {str(e)}")
        return "Entschuldigung, es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut."


def message_bot_stream(user_input, context, knowledge, database_overview, document_id, chat_history):
    """
    Streaming version of message_bot that yields real-time response chunks.
    
    Args:
        user_input: The user's question or message
        context: Relevant document context from vector search  
        knowledge: General knowledge context
        document_id: ID of the document being referenced
        chat_history: Previous conversation history
        
    Yields:
        str: Response chunks from the language model
    """
    try:
        user_input, context, knowledge, chat_history = _validate_inputs(
            user_input, context, knowledge, chat_history
        )
        
        formatted_history = _format_chat_history(chat_history)
        
        # Create streaming LLM and chain with context-aware prompt
        llm = _get_openai_client(streaming=True)
        prompt_template = _create_context_aware_prompt(bool(chat_history))
        
        chain = prompt_template | llm
        
        # Debug: Zeige welche Parameter tatsächlich an den Chain weitergegeben werden
        chain_params = {
            "input": user_input,
            "context": context,
            "knowledge": knowledge,
            "document_id": document_id,
            "chat_history": formatted_history,
        }
        
        # Stream the response
        for chunk in chain.stream(chain_params):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
        
    except ValueError:
        yield "Entschuldigung, ich konnte Ihre Nachricht nicht verstehen."
    except Exception as e:
        print(f"Error in message_bot_stream: {str(e)}")
        yield "Entschuldigung, es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut."


