import logging
logging.basicConfig(level=logging.INFO)
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
from assessment_service import AssessmentService
from redis import Redis
import json
import asyncio
from celery.exceptions import Ignore
from starlette.responses import JSONResponse
from typing import List, Optional

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

# Initialize services
assessment_service = AssessmentService(agent_processor)

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
        "features": ["CrewAI Agents", "Agentic RAG", "PDFMiner", "Streaming Responses", "Structured Outputs"]
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), namespace: str = Form(...), fileID: str = Form(...), additionalInfo: str = Form(...), hasTablesOrGraphics: str = Form(...), numberPages: Optional[str] = Form(None)):
    """
    Upload and process a PDF document asynchronously using AgentProcessor.
    
    Args:
        file: PDF file to upload and process
        namespace: Namespace for organizing documents  
        fileID: Unique identifier for the document
        additionalInfo: Additional information (currently unused)
        hasTablesOrGraphics: Whether to use page-based chunking
        numberPages: Comma-separated list of page numbers for special processing (1-indexed)
        
    Returns:
        Dict containing upload status and task information
    """
    try:
        print(f"Upload request received: filename={file.filename}, namespace={namespace}, fileID={fileID}, numberPages={numberPages}, {type(numberPages)}")
        
        if not file.filename.lower().endswith('.pdf'):
            return {
                "status": "error",
                "message": "Only PDF files are supported",
                "filename": file.filename
            }
            
        content = await file.read()
        print(f"File content read: {len(content)} bytes")
        
        # Parse numberPages from string to list of integers if provided
        special_pages = []
        if numberPages:
            try:
                special_pages = [int(page.strip()) for page in numberPages.split(',') if page.strip()]
                print(f"Special pages to process: {special_pages}")
            except ValueError:
                return {
                    "status": "error",
                    "message": "numberPages must be a comma-separated list of integers",
                    "filename": file.filename
                }
        
        task = process_document.delay(content, namespace, fileID, file.filename, hasTablesOrGraphics, special_pages)
        
        return {
            "status": "success",
            "message": "File upload started - will be processed with CrewAI agents",
            "task_id": task.id,
            "filename": file.filename,
            "special_pages": special_pages
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
    # Input validation
    if not user_input or not isinstance(user_input, str) or not user_input.strip():
        return {
            "status": "error",
            "message": "User input must be a non-empty string",
            "structured_response": {
                "answer": "Ungültige Eingabe",
                "document_ids": [],
                "sources": [],
                "confidence_score": 0.0,
                "context_used": False,
                "additional_info": "Leere oder ungültige Benutzereingabe"
            }
        }
    
    if not namespace or not isinstance(namespace, str) or not namespace.strip():
        return {
            "status": "error",
            "message": "Namespace must be a non-empty string",
            "structured_response": {
                "answer": "Ungültiger Namespace",
                "document_ids": [],
                "sources": [],
                "confidence_score": 0.0,
                "context_used": False,
                "additional_info": "Leerer oder ungültiger Namespace"
            }
        }
    
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
        
        # Add comprehensive logging
        print(f"SEND_MESSAGE DEBUG - User input: {user_input[:100]}...")
        print(f"SEND_MESSAGE DEBUG - Namespace: {namespace}")
        print(f"SEND_MESSAGE DEBUG - History length: {len(history)}")
        
        # Use the structured message_bot function
        structured_response = message_bot_agent(
            user_input, namespace, history
        )
        
        # Validate response structure
        if not isinstance(structured_response, dict):
            print(f"WARNING: Unexpected response type: {type(structured_response)}")
            structured_response = {
                "answer": str(structured_response),
                "document_ids": [],
                "sources": [],
                "confidence_score": 0.5,
                "context_used": False,
                "additional_info": "Response was not in expected format"
            }
        
        # Update chat history
        chat_state.chat_history[namespace].append({"role": "user", "content": user_input})
        chat_state.chat_history[namespace].append({"role": "assistant", "content": structured_response.get("answer", "No answer")})
        
        # Keep only last 20 messages to prevent memory overflow
        if len(chat_state.chat_history[namespace]) > 20:
            chat_state.chat_history[namespace] = chat_state.chat_history[namespace][-20:]
        
        return structured_response
        
    except Exception as e:
        print(f"Error in send_message_structured: {str(e)}")
        import traceback
        traceback.print_exc()
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
    Get status of a specific task by its ID.
    
    Args:
        task_id: Celery task identifier
        
    Returns:
        Dict containing current task status and progress information
        
    Raises:
        HTTPException: If task fails or encounters errors
    """
    try:
        task = celery.AsyncResult(task_id)
        return _handle_task_state(task)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



async def _stream_error_response(message: str):
    """
    Generate error response for streaming endpoints.
    
    Args:
        message: Error message to send
        
    Yields:
        Server-sent event formatted error message
    """
    yield f"data: {json.dumps({'type': 'error', 'message': message})}\n\n"


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
    """
    Get assessment data for a namespace using AssessmentService.
    
    This function is called by the Celery task and maintains compatibility
    with the existing trigger_assessment endpoint.
    """
    return assessment_service.generate_assessment(namespace, additional_info)
      
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