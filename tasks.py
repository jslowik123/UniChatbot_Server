import os
import logging
from celery_app import celery
from agent_processor import AgentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@celery.task(bind=True)
def process_file_async(self, file_path: str, filename: str, namespace: str, task_id: str):
    """Process a PDF file asynchronously using the AgentProcessor"""
    try:
        # Initialize agent processor
        agent_processor = AgentProcessor()
        
        # Update task state
        self.update_state(
            state='PROCESSING',
            meta={
                'status': f'Processing file: {filename}',
                'current': 10,
                'total': 100,
                'file': filename
            }
        )
        
        # Process the document
        result = agent_processor.process_document(file_path, filename, namespace)
        
        # Update task state
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Document processed, finalizing...',
                'current': 90,
                'total': 100,
                'file': filename
            }
        )
        
        # Clean up temporary file
        if os.path.exists(file_path):
            os.unlink(file_path)
        
        # Final state
        return {
            'status': 'success',
            'message': f'Document {filename} processed successfully',
            'document_id': result['document_id'],
            'chunks_created': result['chunks_created'],
            'file': filename,
            'namespace': namespace
        }
        
    except Exception as e:
        logger.error(f"Error processing file {filename}: {str(e)}")
        
        # Clean up temporary file on error
        if os.path.exists(file_path):
            os.unlink(file_path)
        
        # Update task state to failure
        self.update_state(
            state='FAILURE',
            meta={
                'status': f'Failed to process {filename}',
                'error': str(e),
                'file': filename
            }
        )
        
        # Re-raise the exception to mark task as failed
        raise Exception(f"Failed to process {filename}: {str(e)}")

# Backward compatibility function
def process_document(file_content: bytes, namespace: str, fileID: str, filename: str):
    """Legacy function for backward compatibility"""
    import tempfile
    
    # Save content to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name
    
    # Process using the new function
    return process_file_async.delay(temp_file_path, filename, namespace, fileID)