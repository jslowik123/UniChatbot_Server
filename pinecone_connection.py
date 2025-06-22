import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional, Union
import numpy as np
from openai import OpenAI

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_DIMENSION = 1536
MAX_RETRIES = 10
RETRY_DELAY = 1


class PineconeCon:
    """
    Handles connections and operations with Pinecone vector database.
    
    Manages vector embeddings, uploads, queries, and namespace operations
    for document storage and retrieval.
    """
    
    def __init__(self, index_name: str):
        """
        Initialize Pinecone connection and OpenAI client.
        
        Args:
            index_name: Name of the Pinecone index to connect to
            
        Raises:
            ValueError: If required API keys are missing
            ConnectionError: If unable to connect to Pinecone index
        """
        load_dotenv(dotenv_path=".env")
        
        pinecone_key = os.getenv("PINECONE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if not pinecone_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self._pc = Pinecone(api_key=pinecone_key)
        self._openai = OpenAI(api_key=openai_key)
        self._index_name = index_name

        # Wait for index to be ready with timeout
        retries = 0
        while retries < MAX_RETRIES:
            try:
                index_status = self._pc.describe_index(index_name)
                if index_status.status['ready']:
                    break
            except Exception as e:
                print(f"Error checking index status: {e}")
            
            retries += 1
            time.sleep(RETRY_DELAY)
            
        if retries >= MAX_RETRIES:
            raise ConnectionError(f"Unable to connect to Pinecone index '{index_name}' after {MAX_RETRIES} retries")

        self._index = self._pc.Index(index_name)

    def upload(self, chunks: List[str], namespace: str, file_name: str, fileID: str) -> Dict[str, Any]:
        """
        Upload text chunks as embeddings to Pinecone.
        
        Converts text chunks to embeddings using OpenAI and stores them
        in the specified namespace with metadata.
        
        Args:
            chunks: List of text chunks to embed and upload
            namespace: Pinecone namespace for organization
            file_name: Original filename for metadata
            fileID: Unique identifier for the document
            
        Returns:
            Dict containing upload status and metadata
            
        Raises:
            Exception: If embedding generation or upload fails
        """
        if not chunks:
            return {
                "status": "error",
                "message": "No chunks provided for upload"
            }
            
        try:
            vectors_to_upload = []
            
            for i, chunk in enumerate(chunks):
                if not chunk or not chunk.strip():
                    continue
                    
                # Generate embedding for the chunk
                response = self._openai.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=chunk
                )
                embedding = response.data[0].embedding

                vector_data = {
                    "id": f"{fileID}_{i}",
                    "values": embedding,
                    "metadata": {
                        "id": fileID,
                        "text": chunk,
                        "file": file_name,
                        "chunk_number": i
                    }
                }
                vectors_to_upload.append(vector_data)
            
            if not vectors_to_upload:
                return {
                    "status": "error",
                    "message": "No valid chunks to upload"
                }
            
            # Upload all vectors at once for better performance
            self._index.upsert(
                namespace=namespace,
                vectors=vectors_to_upload
            )
            
            return {
                "status": "success",
                "message": f"File {fileID} processed successfully",
                "chunks": len(vectors_to_upload),
                "original_file": file_name
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error uploading embeddings: {str(e)}"
            }

    def delete_embeddings(self, file_name: str, namespace: str) -> Dict[str, Any]:
        """
        Delete all embeddings for a specific file from a namespace.
        
        Args:
            file_name: Name of the file whose embeddings should be deleted
            namespace: Namespace containing the embeddings
            
        Returns:
            Dict containing deletion status
        """
        try:
            self._index.delete(
                namespace=namespace,
                filter={"file": file_name}
            )
            return {
                "status": "success",
                "message": f"Embeddings for file {file_name} deleted successfully"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error deleting embeddings: {str(e)}"
            }

    def query(self, query: str, namespace: str, fileID: str, num_results: int = 3) -> Any:
        """
        Search for similar content using semantic vector search.
        
        Converts the query to an embedding and finds the most similar
        stored content within the specified document.
        
        Args:
            query: Text query to search for
            namespace: Namespace to search within  
            fileID: Specific document ID to search within
            num_results: Maximum number of results to return
            
        Returns:
            Pinecone query results with matches and metadata
            
        Raises:
            Exception: If embedding generation or query fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
            
        try:
            # Generate embedding for the query
            response = self._openai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=query
            )
            embedding = response.data[0].embedding

            # Filter to search only within the specified document
            query_filter = {"id": fileID}

            results = self._index.query(
                namespace=namespace,
                vector=embedding,
                top_k=num_results,
                include_values=False,
                include_metadata=True,
                filter=query_filter
            )
            return results
            
        except Exception as e:
            print(f"Error in vector query: {str(e)}")
            raise

    def delete_namespace(self, namespace: str) -> Dict[str, Any]:
        """
        Delete all vectors in a specific namespace.
        
        Args:
            namespace: Namespace to delete
            
        Returns:
            Dict containing deletion status
        """
        try:
            self._index.delete(namespace=namespace, delete_all=True)
            return {
                "status": "success",
                "message": f"Namespace '{namespace}' deleted successfully"
            }
        except Exception as e:
            return {
                "status": "error", 
                "message": f"Error deleting namespace: {str(e)}"
            }

    def create_namespace_with_dummy(self, namespace: str, dimension: int = DEFAULT_DIMENSION) -> Dict[str, Any]:
        """
        Create a new namespace by inserting a dummy vector.
        
        Pinecone creates namespaces implicitly when vectors are inserted.
        This method creates a placeholder vector to initialize the namespace.
        
        Args:
            namespace: Name of the namespace to create
            dimension: Vector dimension (should match embedding model)
            
        Returns:
            Dict containing creation status and details
        """
        try:
            dummy_vector = np.random.rand(dimension).tolist()
            
            vector = {
                "id": "dummy_vector_1",
                "values": dummy_vector,
                "metadata": {
                    "type": "dummy",
                    "description": "Initial dummy vector"
                }
            }
            
            self._index.upsert(
                vectors=[vector],
                namespace=namespace
            )
            
            return {
                "status": "success",
                "message": f"Namespace '{namespace}' created with dummy vector",
                "namespace": namespace,
                "vector_id": "dummy_vector_1",
                "dimension": dimension
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error creating namespace: {str(e)}"
            }
        
    def delete_all(self, namespace: str) -> Dict[str, Any]:
        """
        Delete all vectors in a namespace.
        
        Args:
            namespace: Namespace to clear
            
        Returns:
            Dict with operation status
        """
        try:
            self._index.delete(namespace=namespace, delete_all=True)
            
            return {
                "status": "success",
                "message": f"All vectors in namespace '{namespace}' deleted successfully"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error deleting all vectors: {str(e)}"
            }

    def get_index_stats(self, namespace: str = None) -> Dict[str, Any]:
        """
        Get statistics about the index or a specific namespace.
        
        Args:
            namespace: Optional namespace to get stats for
            
        Returns:
            Dict containing index statistics
        """
        try:
            stats = self._index.describe_index_stats()
            if namespace and namespace in stats.namespaces:
                return {
                    "status": "success",
                    "namespace": namespace,
                    "vector_count": stats.namespaces[namespace].vector_count
                }
            return {
                "status": "success",
                "total_vector_count": stats.total_vector_count,
                "namespaces": {ns: info.vector_count for ns, info in stats.namespaces.items()}
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error getting index stats: {str(e)}"
            }


def delete_all():
    """
    Legacy function to delete all vectors from the userfiles index.
    
    Returns:
        Dict with operation status
    """
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("PINECONE_API_KEY")
    
    if not api_key:
        return {
            "status": "error",
            "message": "PINECONE_API_KEY not found in environment variables"
        }
    
    try:
        pc = Pinecone(api_key=api_key)
        index = pc.Index("userfiles")
        index.delete(delete_all=True)
        
        return {
            "status": "success",
            "message": "All vectors in index deleted successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error deleting all vectors: {str(e)}"
        }


def test_query():
    """
    Test function for debugging query functionality.
    """
    try:
        load_dotenv(dotenv_path=".env")
        pc = PineconeCon("userfiles")
        results = pc.query(
            query="Was versteht man unter der Makroökonomie?", 
            namespace="neuertest", 
            fileID="-OPgASjQr0q6XnJWE83g", 
            num_results=3
        )
        print("Query results:", results)
        return results
    except Exception as e:
        print(f"Test query failed: {str(e)}")
        return None


if __name__ == "__main__":
    test_query()