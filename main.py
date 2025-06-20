import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
from typing import Optional
from pydantic import BaseModel
from agent_processor import AgentProcessor
from agent_chatbot import AgentChatbot
from tasks import process_file_async
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="University Chatbot API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global instances
agent_processor = AgentProcessor()
agent_chatbot = AgentChatbot()

# Request models
class ChatRequest(BaseModel):
    message: str
    namespace: Optional[str] = "default"

class FileUploadRequest(BaseModel):
    filename: str
    namespace: Optional[str] = "default"

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "message": "University Chatbot API v2.0 - Agent-based Architecture",
        "status": "running",
        "features": [
            "Document processing with PDF extraction",
            "Agent-based question answering",
            "Structured responses with confidence scores",
            "Streaming chat responses",
            "Chat history management"
        ]
    }

@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...), namespace: str = "default"):
    """Upload and process a PDF file"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process file asynchronously
        task_id = f"upload_{file.filename}_{namespace}"
        background_tasks.add_task(
            process_file_async,
            temp_file_path,
            file.filename,
            namespace,
            task_id
        )
        
        return {
            "message": f"File '{file.filename}' uploaded successfully",
            "task_id": task_id,
            "namespace": namespace,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.post("/send_message")
async def send_message(request: ChatRequest):
    """Send a message and get a simple text response"""
    try:
        response = agent_chatbot.send_message(request.message, request.namespace)
        return {"response": response}
        
    except Exception as e:
        logger.error(f"Error in send_message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.post("/send_message_structured")
async def send_message_structured(request: ChatRequest):
    """Send a message and get a structured response with metadata"""
    try:
        response = agent_chatbot.send_message_structured(request.message, request.namespace)
        return response
        
    except Exception as e:
        logger.error(f"Error in send_message_structured: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.post("/send_message_stream")
async def send_message_stream(request: ChatRequest):
    """Send a message and get a streaming response with metadata"""
    try:
        def generate_stream():
            try:
                for chunk in agent_chatbot.send_message_stream(request.message, request.namespace):
                    yield chunk
            except Exception as e:
                logger.error(f"Error in stream generation: {str(e)}")
                error_data = {
                    "type": "error",
                    "content": f"Error generating response: {str(e)}"
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
        
    except Exception as e:
        logger.error(f"Error setting up stream: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting up stream: {str(e)}")

@app.post("/clear_chat_history")
async def clear_chat_history(namespace: str = "default"):
    """Clear chat history for a namespace"""
    try:
        success = agent_chatbot.clear_chat_history(namespace)
        if success:
            return {"message": f"Chat history cleared for namespace '{namespace}'"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear chat history")
            
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing chat history: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Basic health checks
        health_status = {
            "status": "healthy",
            "api_version": "2.0.0",
            "components": {
                "agent_processor": "healthy",
                "agent_chatbot": "healthy"
            }
        }
        
        # Test environment variables
        if not os.getenv("OPENAI_API_KEY"):
            health_status["components"]["openai"] = "missing_api_key"
            health_status["status"] = "degraded"
        else:
            health_status["components"]["openai"] = "healthy"
            
        if not os.getenv("PINECONE_API_KEY"):
            health_status["components"]["pinecone"] = "missing_api_key"
            health_status["status"] = "degraded"
        else:
            health_status["components"]["pinecone"] = "healthy"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/namespaces/{namespace}/info")
async def get_namespace_info(namespace: str):
    """Get information about documents in a namespace"""
    try:
        # This would need to be implemented in the agent_processor
        # For now, return basic info
        return {
            "namespace": namespace,
            "message": "Namespace info endpoint - implementation pending"
        }
        
    except Exception as e:
        logger.error(f"Error getting namespace info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting namespace info: {str(e)}")

# Legacy endpoints for backward compatibility
@app.post("/message")
async def legacy_message(request: ChatRequest):
    """Legacy message endpoint for backward compatibility"""
    return await send_message_structured(request)

@app.post("/message_stream")
async def legacy_message_stream(request: ChatRequest):
    """Legacy streaming message endpoint for backward compatibility"""
    return await send_message_stream(request)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)