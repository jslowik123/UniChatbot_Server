import logging
logging.basicConfig(level=logging.DEBUG)

from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import uvicorn
from agent_processor import AgentProcessor
from agent_chatbot import get_bot, message_bot_agent
from firebase_connection import FirebaseConnection
from celery_app import test_task, celery
from tasks import process_document, generate_assessment
from redis import Redis
import json
import asyncio
from celery.exceptions import Ignore
from starlette.responses import JSONResponse
from openai import OpenAI

# Load environment variables
load_dotenv()

# Constants
API_VERSION = "1.0.0"
DEFAULT_DIMENSION = 1536
STREAM_DELAY = 0.01

# Initialize environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not found in environment variables")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize FastAPI app
app = FastAPI(
    title="Uni Chatbot API - Agent Edition",
    description="API for university document processing and chatbot interactions using CrewAI agents",
    version=API_VERSION
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize connections
pc = Pinecone(api_key=pinecone_api_key)
agent_processor = AgentProcessor(pinecone_api_key, openai_api_key)

class ChatState:
    """
    Manages the state of the chatbot conversation using AgentChatbot.
    
    Stores the conversation chain and chat history for maintaining
    context across multiple interactions.
    """
    
    def __init__(self):
        self.agent_chatbot = None
        self.chat_history = {}  # Chat history per namespace
    
    def reset(self):
        """Reset the chat state to initial values."""
        self.agent_chatbot = None
        self.chat_history = {}

chat_state = ChatState()

@app.get("/")
async def root():
    """
    Root endpoint providing API information and health status.
    
    Returns:
        Dict containing welcome message, status, and version information
    """
    return {
        "message": "Welcome to the Uni Chatbot API - Agent Edition", 
        "status": "online", 
        "version": API_VERSION,
        "features": ["CrewAI Agents", "Agentic RAG", "PDFPlumber", "Streaming Responses", "Structured Outputs"]
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), namespace: str = Form(...), fileID: str = Form(...), additionalInfo: str = Form(...)):
    """
    Upload and process a PDF document asynchronously using AgentProcessor.
    
    Args:
        file: PDF file to upload and process
        namespace: Namespace for organizing documents  
        fileID: Unique identifier for the document
        additionalInfo: Additional information (currently unused)
        
    Returns:
        Dict containing upload status and task information
    """
    try:
        print(f"Upload request received: filename={file.filename}, namespace={namespace}, fileID={fileID}")
        
        if not file.filename.lower().endswith('.pdf'):
            return {
                "status": "error",
                "message": "Only PDF files are supported",
                "filename": file.filename
            }
            
        content = await file.read()
        print(f"File content read: {len(content)} bytes")
        
        task = process_document.delay(content, namespace, fileID, file.filename)
        
        return {
            "status": "success",
            "message": "File upload started - will be processed with CrewAI agents",
            "task_id": task.id,
            "filename": file.filename
        }
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return {
            "status": "error", 
            "message": f"Error processing file: {str(e)}", 
            "filename": getattr(file, 'filename', 'unknown')
        }

@app.post("/delete")
async def delete_file(file_name: str = Form(...), namespace: str = Form(...), 
                     fileID: str = Form(...), just_firebase: str = Form(...)):
    """
    Delete a document from Pinecone and/or Firebase using AgentProcessor.
    
    Args:
        file_name: Name of the file to delete
        namespace: Namespace containing the document
        fileID: Document identifier
        just_firebase: If "true", delete only from Pinecone, otherwise delete from both
        
    Returns:
        Dict containing deletion status from both services
    """
    try:
        if just_firebase.lower() == "true":
            # Delete using AgentProcessor
            result = agent_processor.delete_document(namespace, fileID)
            
            # Trigger assessment update after deletion
            generate_assessment.delay(namespace)
            
            return {
                "status": result["status"], 
                "message": result["message"],
                "agent_processor_status": result["status"]
            }
        else:
            # Delete using AgentProcessor (handles both Pinecone and Firebase)
            result = agent_processor.delete_document(namespace, fileID)
            
            # Trigger assessment update after deletion
            generate_assessment.delay(namespace)
            
            return {
                "status": result["status"], 
                "message": result["message"],
                "agent_processor_status": result["status"]
            }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Error deleting file: {str(e)}"
        }

@app.post("/start_bot")
async def start_bot():
    """
    Initialize the AgentChatbot and reset conversation state.
    
    Returns:
        Dict containing initialization status
    """
    try:
        chat_state.agent_chatbot = get_bot()
        result = chat_state.agent_chatbot.start_bot_agent()
        chat_state.chat_history = {}
        return result
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Error starting agent bot: {str(e)}"
        }


@app.post("/send_message")
async def send_message(user_input: str = Form(...), namespace: str = Form(...)):
    """
    Send a message to the CrewAI agent and get a structured response with metadata.
    
    Args:
        user_input: User's question or message  
        namespace: Namespace to search for relevant documents
        
    Returns:
        Dict containing structured response with answer, sources, confidence, etc.
    """
    if not chat_state.agent_chatbot:
        return {
            "status": "error",
            "message": "Agent bot not started. Please call /start_bot first",
            "structured_response": {
                "answer": "Bot nicht gestartet",
                "document_ids": [],
                "sources": [],
                "confidence_score": 0.0,
                "context_used": False,
                "additional_info": "Bot muss zuerst gestartet werden"
            }
        }
    
    try:
        # Get or initialize chat history for this namespace
        if namespace not in chat_state.chat_history:
            chat_state.chat_history[namespace] = []
        
        history = chat_state.chat_history[namespace]
        
        # Use the structured message_bot function
        structured_response = message_bot_agent(
            user_input, namespace, history
        )
        
        # Update chat history
        chat_state.chat_history[namespace].append({"role": "user", "content": user_input})
        chat_state.chat_history[namespace].append({"role": "assistant", "content": structured_response["answer"]})
        
        # Keep only last 20 messages to prevent memory overflow
        if len(chat_state.chat_history[namespace]) > 20:
            chat_state.chat_history[namespace] = chat_state.chat_history[namespace][-20:]
        
        return structured_response
        
    except Exception as e:
        print(f"Error in send_message_structured: {str(e)}")
        return {
                "answer": f"Fehler: {str(e)}",
                "document_ids": [],
                "sources": [],
                "confidence_score": 0.0,
                "context_used": False,
                "additional_info": f"Exception: {type(e).__name__}"
            }
        

@app.post("/create_namespace")
async def create_namespace(namespace: str = Form(...),dimension: int = Form(DEFAULT_DIMENSION)):
    """
    Create a new namespace in the Pinecone index using AgentProcessor.
    
    Args:
        namespace: Name of the namespace to create
        dimension: Vector dimension (default: 1536 for OpenAI embeddings)
        
    Returns:
        Dict containing creation status
    """
    try:
        # Setup vectorstore for namespace (this creates it if it doesn't exist)
        vectorstore = agent_processor.setup_vectorstore(namespace)
        return {
            "status": "success",
            "message": f"Namespace {namespace} created/initialized successfully",
            "dimension": dimension
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Error creating namespace: {str(e)}"
        }

@app.post("/delete_namespace")
async def delete_namespace(namespace: str = Form(...)):
    """
    Delete a namespace from Pinecone index and Firebase metadata.
    Args:
        namespace: Name of the namespace to delete
    """
    try:
        # Delete namespace from Pinecone using AgentProcessor
        pinecone_result = agent_processor.delete_namespace(namespace)
        
        # Delete Firebase metadata if available
        firebase_result = {"status": "success", "message": "Firebase not available or not configured for this agent."}
        if hasattr(agent_processor, '_firebase_available') and agent_processor._firebase_available:
            firebase_result = agent_processor._firebase.delete_namespace_metadata(namespace)
        
        # Both operations should be considered successful even if namespace doesn't exist
        return JSONResponse(content={
            "status": "success",
            "message": f"Namespace {namespace} deletion process completed.",
            "pinecone_result": pinecone_result,
            "firebase_result": firebase_result
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "error": str(e),
            "message": f"Unexpected error during namespace deletion: {str(e)}"
        })

@app.get("/test_worker")
async def test_worker():
    """
    Test the Celery worker connectivity.
    
    Returns:
        Dict containing test task information
    """
    try:
        result = test_task.delay()
        return {
            "status": "success", 
            "task_id": result.id, 
            "message": "Test task sent to worker"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error testing worker: {str(e)}"
        }

def _handle_task_state(task) -> dict:
    """
    Handle different Celery task states and format response accordingly.
    
    Args:
        task: Celery AsyncResult object
        
    Returns:
        Dict containing formatted task status information
    """
    if task.state == 'PENDING':
        return {
            'state': task.state,
            'status': 'PENDING',
            'message': 'Task is waiting for execution',
            'progress': 0
        }
    elif task.state in ['STARTED', 'PROCESSING']:
        meta = task.info if isinstance(task.info, dict) else {}
        return {
            'state': task.state,
            'status': 'PROCESSING',
            'message': meta.get('status', 'Processing with agents'),
            'current': meta.get('current', 0),
            'total': meta.get('total', 100),
            'progress': meta.get('current', 0),
            'file': meta.get('file', '')
        }
    elif task.state in ['FAILURE', 'REVOKED']:
        # Handle failure states
        if isinstance(task.info, Exception):
            error_info = {
                'type': type(task.info).__name__,
                'message': str(task.info),
                'details': 'Task failed with an exception'
            }
        else:
            meta = task.info if isinstance(task.info, dict) else {}
            error_info = {
                'type': meta.get('exc_type', type(task.result).__name__ if task.result else 'Unknown'),
                'message': meta.get('exc_message', str(task.result) if task.result else 'Unknown error'),
                'details': meta.get('error', 'No additional details available')
            }
        
        raise HTTPException(
            status_code=500,
            detail={
                'state': task.state,
                'status': 'FAILURE',
                'message': 'Task processing failed',
                'error': error_info,
                'progress': 0
            }
        )
    elif task.state == 'SUCCESS':
        result = task.result if isinstance(task.result, dict) else {}
        
        if not result:
            return {
                'state': task.state,
                'status': 'SUCCESS',
                'message': 'Task completed but no result available',
                'progress': 100,
                'result': {
                    'message': 'No result data available',
                    'chunks': 0,
                    'agent_status': 'unknown',
                    'firebase_status': 'unknown',
                    'file': ''
                }
            }
        else:
            return {
                'state': task.state,
                'status': 'SUCCESS',
                'message': 'Completed successfully - Agent ready',
                'progress': 100,
                'result': {
                    'message': result.get('message', 'Task completed'),
                    'chunks': result.get('chunks', 0),
                    'pinecone_status': result.get('pinecone_result', {}).get('status', 'unknown'),
                    'firebase_status': result.get('firebase_result', {}).get('status', 'unknown'),
                    'file': result.get('file', '')
                }
            }
    else:
        return {
            'state': task.state,
            'status': 'UNKNOWN',
            'message': f'Unknown state: {task.state}',
            'info': str(task.info) if task.info else 'No info available',
            'progress': 0
        }

@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    """
    Get the status of an asynchronous task.
    
    Args:
        task_id: Unique identifier of the task to check
        
    Returns:
        Dict containing task status, progress, and results
        
    Raises:
        HTTPException: If task failed or status check encountered an error
    """
    try:
        task = celery.AsyncResult(task_id)
        print(f"Task state: {task.state}")
        print(f"Task info: {task.info}")
        print(f"Task result: {task.result}")
        
        response = _handle_task_state(task)
        print(f"Sending response: {response}")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error in task status: {str(e)}")
        error_detail = {
            'state': 'ERROR',
            'status': 'ERROR',
            'message': 'Error checking task status',
            'error': {
                'type': type(e).__name__,
                'message': str(e),
                'details': 'Error occurred while checking task status'
            },
            'progress': 0
        }
        raise HTTPException(status_code=500, detail=error_detail)

async def _stream_error_response(message: str):
    """
    Generate error response for streaming endpoints.
    
    Args:
        message: Error message to send
        
    Yields:
        Server-sent event formatted error message
    """
    yield f"data: {json.dumps({'type': 'error', 'message': message})}\n\n"

@app.post("/send_message_stream")
async def send_message_stream(user_input: str = Form(...), namespace: str = Form(...)):
    """
    Streaming version of send_message with real-time AI response chunks using CrewAI agents.
    
    Sends Server-Sent Events with incremental response chunks as the AI agent
    generates the response, providing real-time feedback to the user.
    
    Args:
        user_input: User's question or message  
        namespace: Namespace to search for relevant documents
        
    Returns:
        StreamingResponse with Server-Sent Events
    """
    if not chat_state.agent_chatbot:
        return StreamingResponse(
            _stream_error_response("Agent bot not started. Please call /start_bot first"),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            }
        )
    
    async def generate_response():
        try:
            # Get or initialize chat history for this namespace
            if namespace not in chat_state.chat_history:
                chat_state.chat_history[namespace] = []
            
            history = chat_state.chat_history[namespace]
            
            # Use the chatbot's stream_message method
            chatbot = get_bot()
            accumulated_response = ""
            
            # Send initial processing event
            yield f"data: {json.dumps({'type': 'start'})}\n\n"
            
            async for chunk in chatbot.stream_message(user_input, namespace, history):
                accumulated_response += chunk
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            
            # Update chat history
            chat_state.chat_history[namespace].append({"role": "user", "content": user_input})
            chat_state.chat_history[namespace].append({"role": "assistant", "content": accumulated_response})
            
            # Keep only last 20 messages
            if len(chat_state.chat_history[namespace]) > 20:
                chat_state.chat_history[namespace] = chat_state.chat_history[namespace][-20:]
            
            # Send completion event with structured data
            final_response = {
                "type": "complete",
                "fullResponse": accumulated_response,
                "structured_response": {
                    "answer": accumulated_response,
                    "document_ids": [],  # Placeholder
                    "sources": [],  # Placeholder
                    "confidence_score": 0.95,  # Placeholder
                    "context_used": True,  # Placeholder
                }
            }
            yield f"data: {json.dumps(final_response)}\n\n"
            
        except Exception as e:
            print(f"Error in send_message_stream: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

@app.get("/namespace_info/{namespace}")
async def get_namespace_info(namespace: str):
    """
    Get information about documents in a namespace using AgentProcessor.
    
    Args:
        namespace: Namespace identifier
        
    Returns:
        Dict containing namespace information
    """
    try:
        if not chat_state.agent_chatbot:
            chat_state.agent_chatbot = get_bot()
        
        result = chat_state.agent_chatbot.get_namespace_info(namespace)
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting namespace info: {str(e)}"
        }

@app.post("/set_project_info")
async def set_project_info(project_name: str = Form(...), info: str = Form(...)):
    """
    Setzt eine Info für ein Projekt in Firebase.
    """
    try:
        firebase_result = agent_processor._firebase.set_project_info(project_name, info)
        return firebase_result
    except Exception as e:
        return {"status": "error", "message": f"Fehler beim Speichern der Projektinfo: {str(e)}"}

@app.get("/get_project_info")
async def get_project_info(project_name: str):
    """
    Holt die Info für ein Projekt aus Firebase.
    """
    try:
        firebase_result = agent_processor._firebase.get_project_info(project_name)
        return firebase_result
    except Exception as e:
        return {"status": "error", "message": f"Fehler beim Abrufen der Projektinfo: {str(e)}"}


def get_assessment_data(namespace: str, additional_info: str):
    dokumente = agent_processor.get_documents(namespace)
    chatbot_ziel = additional_info
    print(chatbot_ziel)
    client = OpenAI()
    """
    Get assessment data for a namespace.
    """
    print(dokumente)
    system_prompt = """
    Du bist ein Experte für Bildungstechnologie und Wissensanalyse. Ein Benutzer möchte einen Chatbot für Studierende erstellen. Dazu hat er Dokumente hochgeladen, die das Fachwissen des Chatbots bilden sollen. Außerdem hat er eine Zielbeschreibung beigefügt, was der Chatbot ungefähr können oder leisten soll.

    Deine Aufgabe ist eine strukturierte Analyse mit KURZEN STICHWORTEN zu erstellen:

    WICHTIG: Antworte im JSON-Format mit exakt diesen Feldern:
    {
        "vorhandene_dokumente": ["Stichwort 1", "Stichwort 2", "Stichwort 3"],
        "fehlende_dokumente": ["Fehlendes Dokument 1", "Fehlendes Dokument 2"],
        "tipps": ["Kurzer Tipp 1", "Kurzer Tipp 2"],
        "wissensstand": "Kurze Beschreibung des Wissensstands des Chatbots, also was alles beantwortet werden kann (5-10 Sätze)",
        
        "confidence": 85
    }

    Regeln für die Stichworte:
    - "vorhanden": Kurze Namen der vorhandenen Dokumenttypen (z.B. "Modulhandbuch Bachelor", "Studienordnung")
    - "fehlt": Konkrete fehlende Dokumenttypen (z.B. "Prüfungsordnung", "Stundenplan", "Praktikumsordnung")
    - "tipps": Maximal 3-4 Wörter pro Tipp (z.B. "Aktuelle Prüfungstermine hinzufügen", "FAQ für Erstsemester")
    - "confidence": Zahl zwischen 0-100 für wie vollständing deiner meinung nach der Chatbot ist, im Hinblick auf das Ziel.

    WICHTIGE REGEL FÜR "FEHLT":
    Analysiere die Zielbeschreibung des Chatbots genau. Wenn bestimmte Dokumenttypen für das spezifische Ziel NICHT RELEVANT sind, dann markiere sie NICHT als fehlend.
    
    Beispiele:
    - Wenn der Chatbot nur für "allgemeine Studienberatung" gedacht ist → keine spezifischen Fachmodule als fehlend markieren
    - Wenn der Chatbot nur für "Erstsemester-Info" gedacht ist → keine Master-spezifischen Dokumente als fehlend markieren
    - Wenn der Chatbot nur für "einen spezifischen Studiengang" gedacht ist → keine anderen Studiengänge als fehlend markieren
    - Wenn der Chatbot für "organisatorische Fragen" gedacht ist → keine fachlichen Inhalte als fehlend markieren

    Fokussiere dich nur auf Dokumente, die für das SPEZIFISCHE ZIEL des Chatbots wirklich notwendig sind.

    Denke an typische Universitätsdokumente:
    - Modulhandbücher, Prüfungsordnungen, Studienordnungen
    - Stundenpläne, Vorlesungsverzeichnis

    Antworte NUR mit dem JSON-Objekt, keine weiteren Erklärungen.
    """

    user_prompt = f"""
    Ziel des Chatbots:
    {chatbot_ziel}

    Vorliegende Dokumente:
    {dokumente}
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # oder dein gewünschtes Modell
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.3,
        stream=False,
    )

    assessment_content = response.choices[0].message.content

    # Speichere das Assessment in Firebase unter files/{namespace}/assessment
    try:
        agent_processor._firebase._db.reference(f'files/{namespace}/assessment').set(assessment_content)
        firebase_status = "success"
    except Exception as e:
        firebase_status = f"error: {str(e)}"

    return {"status": "success", "assessment": assessment_content, "firebase_status": firebase_status}
      
@app.post("/trigger_assessment")
async def trigger_assessment(namespace: str = Form(...)):
    """
    Manually trigger assessment generation for a namespace.
    
    Args:
        namespace: Namespace to generate assessment for
        
    Returns:
        Dict containing task status
    """
    try:
        task = generate_assessment.delay(namespace)
        return {
            "status": "success",
            "message": f"Assessment generation triggered for namespace: {namespace}",
            "task_id": task.id
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error triggering assessment: {str(e)}"
        }

if __name__ == "__main__":
    # Für lokale Entwicklung mit Reload-Funktionalität
    port = int(os.environ.get("PORT", 8000))
    # reload immer True für Entwicklung
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port,
        reload=True,
        timeout_keep_alive=120,
        timeout_graceful_shutdown=120
    )