from celery_app import celery
import time
from redis import Redis
import os
from agent_processor import AgentProcessor
from dotenv import load_dotenv

load_dotenv()

# Initialize Redis and AgentProcessor
r = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
agent_processor = AgentProcessor(
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)


@celery.task(bind=True, name="tasks.process_document")
def process_document(self, file_content: bytes, namespace: str, fileID: str, filename: str, hasTablesOrGraphics: str = "false", special_pages: list = None):
    """
    Process a document asynchronously using Celery with AgentProcessor.
    
    Handles PDF parsing, text extraction, embedding generation, and storage
    in both Pinecone and Firebase with progress tracking using the new
    CrewAI-based agent architecture.
    
    Args:
        self: Celery task instance (bound task)
        file_content: Raw PDF file content as bytes
        namespace: Pinecone namespace for organization
        fileID: Unique document identifier
        filename: Original filename
        hasTablesOrGraphics: Whether to use page-based chunking ("true") or sentence-based chunking ("false")
        special_pages: List of page numbers (1-indexed) for special processing as images
        
    Returns:
        Dict containing processing results and status
        
    Raises:
        Exception: If processing fails at any stage
    """
    try:
        # Update initial status
        if agent_processor._firebase_available:
            agent_processor._firebase.update_document_status(namespace, fileID, {
                'processing': True,
                'progress': 0,
                'status': 'Starting document processing'
            })
        
        self.update_state(
            state='STARTED',
            meta={
                'status': 'Starting document processing',
                'current': 0,
                'total': 100,
                'file': filename
            }
        )
        
        # Update status: Reading document
        if agent_processor._firebase_available:
            agent_processor._firebase.update_document_status(namespace, fileID, {
                'progress': 25,
                'status': 'Reading document with pdfminer'
            })
        
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Reading document with pdfminer',
                'current': 25,
                'total': 100,
                'file': filename
            }
        )
        
        # Update status: Processing chunks and embeddings
        if agent_processor._firebase_available:
            agent_processor._firebase.update_document_status(namespace, fileID, {
                'progress': 50,
                'status': 'Creating chunks and embeddings'
            })
        
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Creating chunks and embeddings',
                'current': 50,
                'total': 100,
                'file': filename
            }
        )
        
        # Process the document using the new AgentProcessor
        result = agent_processor.process_document_full(file_content, namespace, fileID, filename, hasTablesOrGraphics, special_pages)
        
        # Update status: Finalizing processing
        if agent_processor._firebase_available:
            agent_processor._firebase.update_document_status(namespace, fileID, {
                'progress': 75,
                'status': 'Finalizing processing and setting up agent'
            })
        
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Finalizing processing and setting up agent',
                'current': 75,
                'total': 100,
                'file': filename
            }
        )
        
        if result['status'] == 'success':
            # Update final status
            if agent_processor._firebase_available:
                agent_processor._firebase.update_document_status(namespace, fileID, {
                    'processing': False,
                    'progress': 100,
                    'status': 'Complete - Agent ready'
                })
            
            # Trigger the assessment generation task
            generate_assessment.delay(namespace)
            
            return {
                'status': 'success',
                'message': result['message'],
                'chunks': result.get('chunks', 0),
                'pinecone_result': result.get('pinecone_result', {}),
                'firebase_result': result.get('firebase_result', {}),
                'file': filename,
                'current': 100,
                'total': 100
            }
        else:
            # Handle processing failure
            if agent_processor._firebase_available:
                agent_processor._firebase.update_document_status(namespace, fileID, {
                    'processing': False,
                    'progress': 0,
                    'status': f"Processing failed: {result['message']}"
                })
            raise Exception(f"Processing failed: {result['message']}")
            
    except Exception as e:
        # Update error status
        if agent_processor._firebase_available:
            agent_processor._firebase.update_document_status(namespace, fileID, {
                'processing': False,
                'progress': 0,
                'status': f"Failed: {str(e)}"
            })
        
        self.update_state(
            state='FAILURE',
            meta={
                'exc_type': type(e).__name__,
                'exc_message': str(e),
                'status': 'Failed',
                'error': f"{type(e).__name__}: {str(e)}",
                'file': filename,
                'current': 0,
                'total': 100
            }
        )
        raise e


@celery.task(name="tasks.generate_namespace_summary")
def generate_namespace_summary(namespace: str):
    """
    Generate and store a global summary for all documents in a namespace.
    
    This task is triggered after successful document processing to maintain
    an up-to-date overview of all documents in the namespace.
    
    Args:
        namespace: The namespace to generate a summary for
        
    Returns:
        Dict containing operation status and results
    """
    try:
        result = agent_processor.get_namespace_summary(namespace)
        return {
            'status': 'success',
            'message': f"Global summary generated for namespace: {namespace}",
            'summary_result': result
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Failed to generate global summary: {str(e)}"
        }


@celery.task(name="tasks.generate_assessment")
def generate_assessment(namespace: str, additional_info: str = None):
    """
    Generate assessment data for a namespace after document processing is complete.
    
    This task is triggered after successful document processing to automatically
    create an assessment of the chatbot's capabilities based on the uploaded documents.
    
    Args:
        namespace: The namespace to generate assessment for
        additional_info: Optional additional information about chatbot goals
        
    Returns:
        Dict containing assessment generation status and results
    """
    try:
        # Get the additional_info (project info) from Firebase if not provided
        if not additional_info:
            if agent_processor._firebase_available:
                project_info_result = agent_processor._firebase.get_project_info(namespace)
                if project_info_result.get("status") == "success":
                    additional_info = project_info_result.get("info", "")
                else:
                    additional_info = "Keine spezifischen Ziele definiert."
            else:
                additional_info = "Firebase nicht verfügbar - keine Projektinfo verfügbar."
        
        # Import the assessment function from main to maintain API compatibility
        from main import get_assessment_data
        
        # Generate the assessment
        assessment_result = get_assessment_data(namespace, additional_info)
        
        return {
            'status': 'success',
            'message': f"Assessment generated for namespace: {namespace}",
            'assessment_result': assessment_result
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Failed to generate assessment: {str(e)}"
        }


@celery.task(name="tasks.generate_example_questions_task")
def generate_example_questions_task(namespace: str):
    """
    Generate example questions asynchronously.
    
    Args:
        namespace: Namespace to generate questions for
        
    Returns:
        Dict containing generation results
    """
    try:
        agent_processor = AgentProcessor(
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        return agent_processor.generate_and_store_example_questions(namespace)
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Error generating example questions: {str(e)}"
        }