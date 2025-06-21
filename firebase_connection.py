import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from typing import Dict, Any, List
import json
import os
import re
from dotenv import load_dotenv
# Debug print
print("Environment variables:")
print(f"FIREBASE_DATABASE_URL: {os.getenv('FIREBASE_DATABASE_URL')}")
print(f"FIREBASE_CREDENTIALS_PATH: {os.getenv('FIREBASE_CREDENTIALS_PATH')}")
print(f"Current working directory: {os.getcwd()}")

class FirebaseConnection:
    """
    Handles connections and operations with Firebase Realtime Database.
    
    Manages document metadata storage, retrieval, and namespace operations
    for the university chatbot system.
    """
    
    def __init__(self):
        """
        Initialize Firebase connection using environment variables.
        
        Environment variables used:
        - FIREBASE_DATABASE_URL: URL of Firebase Realtime Database (required)
        - FIREBASE_CREDENTIALS_PATH: Path to credentials file (optional)
        - FIREBASE_CREDENTIALS_JSON: JSON string with credentials (optional, for Heroku)
        
        Raises:
            ValueError: If FIREBASE_DATABASE_URL is not set
        """
        load_dotenv()
        database_url = os.getenv('FIREBASE_DATABASE_URL')
        if not database_url:
            raise ValueError("FIREBASE_DATABASE_URL environment variable must be set")
        
        # Always ensure we have a Firebase app initialized
        try:
            # Check if any app exists
            if not firebase_admin._apps:
                self._initialize_firebase_app(database_url)
            else:
                # Get the default app
                app = firebase_admin.get_app()
                print(f"Using existing Firebase app: {app.name}")
        except ValueError as e:
            # If no default app exists, try to get or create one
            try:
                app = firebase_admin.get_app()
                print(f"Retrieved existing Firebase app: {app.name}")
            except ValueError:
                # No app exists, initialize new one
                print("No Firebase app found, initializing new one...")
                self._initialize_firebase_app(database_url)
        
        self._db = db

    def _initialize_firebase_app(self, database_url: str):
        """
        Initialize Firebase app with appropriate credentials.
        
        Args:
            database_url: Firebase database URL
        """
        credentials_path = os.getenv('FIREBASE_CREDENTIALS_PATH')
        credentials_json = os.getenv('FIREBASE_CREDENTIALS_JSON')
        
        try:
            if credentials_json:
                # Initialize from JSON string
                cred_dict = json.loads(credentials_json)
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred, {
                    'databaseURL': database_url
                })
                print("Firebase initialized with JSON credentials")
            elif credentials_path and os.path.exists(credentials_path):
                # Initialize from file
                cred = credentials.Certificate(credentials_path)
                firebase_admin.initialize_app(cred, {
                    'databaseURL': database_url
                })
                print("Firebase initialized with file credentials")
            else:
                # Emergency fallback without credentials (for development)
                firebase_admin.initialize_app(options={
                    'databaseURL': database_url
                })
                print("Firebase initialized without credentials (development mode)")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON credentials: {str(e)}")
            # Try file credentials as fallback
            if credentials_path and os.path.exists(credentials_path):
                cred = credentials.Certificate(credentials_path)
                firebase_admin.initialize_app(cred, {
                    'databaseURL': database_url
                })
                print("Firebase initialized with file credentials after JSON error")
            else:
                # Last resort fallback
                firebase_admin.initialize_app(options={
                    'databaseURL': database_url
                })
                print("Firebase initialized without credentials after JSON error")
        except Exception as e:
            print(f"Firebase initialization error: {str(e)}")
            # Last resort - try basic initialization
            try:
                firebase_admin.initialize_app(options={
                    'databaseURL': database_url
                })
                print("Firebase initialized with basic fallback")
            except Exception as final_e:
                print(f"Final Firebase initialization failed: {str(final_e)}")
                raise ValueError(f"Could not initialize Firebase: {str(final_e)}")

    def _fallback_initialization(self, database_url: str, credentials_path: str = None):
        """
        Fallback initialization when JSON credentials fail.
        
        Args:
            database_url: Firebase database URL
            credentials_path: Optional path to credentials file
        """
        # This method is now integrated into _initialize_firebase_app
        pass

    def append_metadata(self, namespace: str, fileID: str, chunk_count: int, 
                       keywords: List[str], summary: str) -> Dict[str, Any]:
        """
        Store or update document metadata in Firebase.
        
        Args:
            namespace: Namespace where the document is stored
            fileID: Unique document identifier
            chunk_count: Number of text chunks created
            keywords: List of extracted keywords
            summary: Document summary
            
        Returns:
            Dict containing operation status and information
        """
        try:
            # Ensure Firebase app is available
            if not firebase_admin._apps:
                return {
                    'status': 'error',
                    'message': 'Firebase app not initialized'
                }
            
            # Database path for the document
            ref = self._db.reference(f'files/{namespace}/{fileID}')
            
            # Get existing data
            existing_data = ref.get() or {}
            
            # Merge keywords (remove duplicates)
            existing_keywords = existing_data.get('keywords', [])
            combined_keywords = list(set(existing_keywords + keywords))
            
            updated_data = {
                'chunk_count': chunk_count,
                'keywords': combined_keywords,
                'summary': summary,
            }
            
            # Update only changed fields
            for key, value in updated_data.items():
                if key not in existing_data or existing_data[key] != value:
                    ref.child(key).set(value)
            
            return {
                'status': 'success',
                'message': f'Metadata for {fileID} successfully updated',
                'path': f'files/{namespace}/{fileID}'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error during Firebase upload: {str(e)}'
            }
    
    def get_document_metadata(self, namespace: str, fileID: str) -> Dict[str, Any]:
        """
        Retrieve metadata for a specific document.
        
        Args:
            namespace: Namespace containing the document
            fileID: Document identifier
            
        Returns:
            Dict containing document metadata or error information
        """
        try:
            ref = self._db.reference(f'files/{namespace}/{fileID}')
            data = ref.get()
            
            if data:
                return {
                    'status': 'success',
                    'data': data
                }
            else:
                return {
                    'status': 'error',
                    'message': f'No metadata found for {fileID}'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error retrieving metadata: {str(e)}'
            }
    
    def list_documents(self, namespace: str = None) -> Dict[str, Any]:
        """
        List all documents or documents in a specific namespace.
        
        Args:
            namespace: Optional namespace to filter by
            
        Returns:
            Dict containing list of documents
        """
        try:
            if namespace:
                ref = self._db.reference(f'files/{namespace}')
            else:
                ref = self._db.reference('files')
                
            data = ref.get()
            
            return {
                'status': 'success',
                'data': data or {}
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error listing documents: {str(e)}'
            }
            
    def delete_document_metadata(self, namespace: str, fileID: str) -> Dict[str, Any]:
        """
        Delete document metadata from Firebase.
        
        Args:
            namespace: Namespace containing the document
            fileID: Document identifier
            
        Returns:
            Dict containing operation status
        """
        try:
            ref = self._db.reference(f'files/{namespace}/{fileID}')
            
            # Check if document exists
            existing_data = ref.get()
            if not existing_data:
                return {
                    'status': 'error',
                    'message': f'No metadata found for {fileID}'
                }
                
            # Delete document
            ref.delete()
            
            return {
                'status': 'success',
                'message': f'Metadata for {fileID} successfully deleted',
                'path': f'files/{namespace}/{fileID}'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error deleting metadata: {str(e)}'
            }
            
    def delete_namespace_metadata(self, namespace: str) -> Dict[str, Any]:
        """
        Delete all metadata in a namespace from Firebase.
        
        Args:
            namespace: Namespace to delete
            
        Returns:
            Dict containing operation status
        """
        try:
            ref = self._db.reference(f'files/{namespace}')
            
            # Check if namespace exists
            existing_data = ref.get()
            if not existing_data:
                return {
                    'status': 'error',
                    'message': f'Namespace {namespace} not found'
                }
                
            # Delete namespace
            ref.delete()
            
            return {
                'status': 'success',
                'message': f'Namespace {namespace} successfully deleted',
                'path': f'files/{namespace}'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error deleting namespace: {str(e)}'
            }

    def get_namespace_data(self, namespace: str) -> Dict[str, Any]:
        """
        Retrieve all data for a specific namespace from Firebase.
        
        Args:
            namespace: Namespace whose data should be retrieved
            
        Returns:
            Dict containing namespace data or error information
        """
        try:
            ref = self._db.reference(f'files/{namespace}')
            data = ref.get()
            
            if data:
                return {
                    'status': 'success',
                    'data': data
                }
            else:
                return {
                    'status': 'error',
                    'data': {},
                    'message': f'No data found for namespace {namespace}'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error retrieving data for namespace {namespace}: {str(e)}'
            }

    def update_document_status(self, namespace: str, fileID: str, status_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the processing status of a document in Firebase.
        
        Args:
            namespace: Namespace where the document is stored
            fileID: Document identifier
            status_data: Dictionary containing status information (processing, progress, status)
            
        Returns:
            Dict containing operation status
        """
        try:
            # Ensure Firebase app is available
            if not firebase_admin._apps:
                return {
                    'status': 'error',
                    'message': 'Firebase app not initialized'
                }
            
            ref = self._db.reference(f'files/{namespace}/{fileID}')
            
            existing_data = ref.get() or {}
            
            # Update with new status data
            for key, value in status_data.items():
                existing_data[key] = value
                
            ref.update(existing_data)
            
            return {
                'status': 'success',
                'message': f'Status for {fileID} successfully updated',
                'path': f'files/{namespace}/{fileID}'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error updating document status: {str(e)}'
            }

    def update_namespace_summary(self, namespace: str, bullet_points: List[str]) -> Dict[str, Any]:
        """
        Store or update global summary bullet points for a namespace.
        
        Args:
            namespace: Namespace to store the summary for
            bullet_points: List of summary bullet points
            
        Returns:
            Dict containing operation status
        """
        try:
            # Path for the global summary of the namespace
            path = f'files/{namespace}/summary' 
            ref = self._db.reference(path)
            ref.set(bullet_points)  # Store the list of bullet points
            
            return {
                'status': 'success',
                'message': f'Global summary bullet points for namespace {namespace} updated successfully',
                'path': path
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error updating global namespace summary: {str(e)}'
            }

