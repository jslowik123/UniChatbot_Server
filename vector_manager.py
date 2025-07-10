import os
from typing import Dict, List, Any, Optional, Tuple
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import logging

load_dotenv()

# Constants
EMBEDDING_MODEL = "text-embedding-ada-002"

logger = logging.getLogger(__name__)


class VectorManager:
    """
    Handles all vector database operations including indexing, retrieval, and management.
    
    This class manages Pinecone vector operations, document indexing, and chunk management
    for the document processing pipeline.
    """
    
    def __init__(self, pinecone_api_key: str, openai_api_key: str, index_name: str = "pdfs-index"):
        """
        Initialize VectorManager with API keys and connections.
        
        Args:
            pinecone_api_key: API key for Pinecone vector database
            openai_api_key: API key for OpenAI embeddings
            index_name: Name of the Pinecone index
            
        Raises:
            ValueError: If required API keys are missing
        """
        if not pinecone_api_key or not openai_api_key:
            raise ValueError("Both Pinecone and OpenAI API keys are required")
            
        self._pinecone_api_key = pinecone_api_key
        self._openai_api_key = openai_api_key
        self._index_name = index_name
        
        # Initialize Pinecone
        self._pc = Pinecone(api_key=pinecone_api_key)
        
        # Initialize OpenAI embeddings
        self._embeddings = OpenAIEmbeddings(api_key=openai_api_key, model=EMBEDDING_MODEL)
        
        # Cache for vectorstores per namespace
        self._vectorstores = {}
        
        # Ensure index exists
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        """Ensure the Pinecone index exists, create if necessary."""
        try:
            # Check if index exists
            existing_indexes = [idx.name for idx in self._pc.list_indexes()]
            
            if self._index_name not in existing_indexes:
                from pinecone import ServerlessSpec
                self._pc.create_index(
                    name=self._index_name,
                    dimension=1536,  # text-embedding-ada-002 dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                # Wait for index to be ready
                import time
                time.sleep(60)  # Wait for index initialization
        except Exception as e:
            logger.error(f"Error ensuring index exists: {e}")

    def setup_vectorstore(self, namespace: str) -> PineconeVectorStore:
        """
        Set up or retrieve vectorstore for a specific namespace.
        
        Args:
            namespace: Namespace identifier
            
        Returns:
            PineconeVectorStore vectorstore instance
        """
        if namespace in self._vectorstores:
            return self._vectorstores[namespace]
        
        # Create vectorstore for this namespace
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=self._index_name, 
            embedding=self._embeddings,
            namespace=namespace
        )
        
        self._vectorstores[namespace] = vectorstore
        return vectorstore

    def index_document(self, processed_pdf: Dict[str, Any], namespace: str, fileID: str) -> Dict[str, Any]:
        """
        Index processed PDF content in Pinecone vectorstore.
        
        Args:
            processed_pdf: Dictionary containing chunks, summary, and metadata
            namespace: Namespace for document organization
            fileID: Unique document identifier
            
        Returns:
            Dict containing indexing status and results
        """
        if not processed_pdf:
            return {
                "status": "error",
                "message": "No processed PDF data provided"
            }
            
        try:
            vectorstore = self.setup_vectorstore(namespace)
            
            # Prepare texts and metadata with page information
            texts = processed_pdf["chunks"] + [processed_pdf["summary"]]
            
            # Create metadata with page information if available
            metadatas = []
            chunks_with_pages = processed_pdf.get("chunks_with_pages", [])
            
            for i in range(len(processed_pdf["chunks"])):
                metadata = {
                    "pdf_id": fileID, 
                    "document_id": fileID, 
                    "type": "chunk", 
                    "chunk_id": i
                }
                
                # Page information no longer tracked
                
                metadatas.append(metadata)
            
            # Add summary metadata
            metadatas.append({"pdf_id": fileID, "document_id": fileID, "type": "summary"})
            
            # Add texts to vectorstore with retry mechanism
            ids = [f"{fileID}_chunk_{i}" for i in range(len(processed_pdf["chunks"]))] + [f"{fileID}_summary"]
            
            print(f"📤 Indexing {len(texts)} texts for document {fileID}")
            print(f"📊 Text lengths: {[len(text) for text in texts[:5]]}...")  # Show first 5 lengths
            
            # Retry mechanism for Pinecone upload
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                    print(f"✅ Successfully indexed document {fileID} on attempt {attempt + 1}")
                    break
                except Exception as upload_error:
                    print(f"❌ Attempt {attempt + 1} failed: {str(upload_error)}")
                    if attempt == max_retries - 1:
                        raise upload_error
                    
                    # Wait before retry (exponential backoff)
                    import time
                    wait_time = 2 ** attempt
                    print(f"⏳ Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
            
            return {
                "status": "success",
                "message": f"Document {fileID} indexed successfully",
                "chunks": len(processed_pdf["chunks"]),
                "vector_ids": ids
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error indexing document: {str(e)}"
            }

    def get_adjacent_chunks(self, namespace: str, chunk_id: str) -> Dict[str, str]:
        """
        Retrieve adjacent chunks (previous and next) for a given chunk ID.
        
        Args:
            namespace: Namespace to search in
            chunk_id: ID of the main chunk (format: fileID_chunk_X)
            
        Returns:
            Dict containing previous, current, and next chunk content
        """
        try:
            index = self._pc.Index(self._index_name)
            
            # Parse chunk ID to get document ID and chunk number
            if "_chunk_" not in chunk_id:
                return {"current": None, "previous": None, "next": None}
            
            parts = chunk_id.split("_chunk_")
            if len(parts) != 2:
                return {"current": None, "previous": None, "next": None}
            
            file_id = parts[0]
            try:
                chunk_num = int(parts[1])
            except ValueError:
                return {"current": None, "previous": None, "next": None}
            
            # Construct IDs for adjacent chunks
            prev_id = f"{file_id}_chunk_{chunk_num - 1}" if chunk_num > 0 else None
            next_id = f"{file_id}_chunk_{chunk_num + 1}"
            
            # Fetch chunks from Pinecone
            result = {"current": None, "previous": None, "next": None}
            
            # Get current chunk
            try:
                current_response = index.fetch(ids=[chunk_id], namespace=namespace)
                if chunk_id in current_response.get('vectors', {}):
                    result["current"] = chunk_id
            except Exception as e:
                print(f"❌ Error fetching current chunk {chunk_id}: {e}")
            
            # Get previous chunk
            if prev_id:
                try:
                    prev_response = index.fetch(ids=[prev_id], namespace=namespace)
                    if prev_id in prev_response.get('vectors', {}):
                        result["previous"] = prev_id
                except Exception as e:
                    print(f"⚠️  Previous chunk {prev_id} not found: {e}")
            
            # Get next chunk
            try:
                next_response = index.fetch(ids=[next_id], namespace=namespace)
                if next_id in next_response.get('vectors', {}):
                    result["next"] = next_id
            except Exception as e:
                print(f"⚠️  Next chunk {next_id} not found: {e}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in get_adjacent_chunks: {e}")
            return {"current": None, "previous": None, "next": None}

    def get_chunk_content_by_id(self, namespace: str, chunk_id: str) -> Optional[str]:
        """
        Get the actual text content of a chunk by its ID.
        
        Args:
            namespace: Namespace to search in
            chunk_id: ID of the chunk
            
        Returns:
            Text content of the chunk or None if not found
        """
        try:
            vectorstore = self.setup_vectorstore(namespace)
            
            # Use similarity search with the chunk ID as metadata filter
            # This is a workaround since Pinecone doesn't store full text in fetch
            docs = vectorstore.similarity_search(
                query="",  # Empty query since we're filtering by ID
                k=1,
                filter={"chunk_id": chunk_id.split("_chunk_")[-1]} if "_chunk_" in chunk_id else {}
            )
            
            if docs and len(docs) > 0:
                return docs[0].page_content
            
            return None
            
        except Exception as e:
            print(f"❌ Error getting chunk content for {chunk_id}: {e}")
            return None

    def get_adjacent_chunks_content(self, namespace: str, doc_id: str, chunk_id: int) -> Dict[str, Optional[str]]:
        """
        Get the actual content of adjacent chunks for a given chunk.
        
        Args:
            namespace: Namespace to search in
            doc_id: Document ID
            chunk_id: Chunk number (integer)
            
        Returns:
            Dict containing previous, current, and next chunk content
        """
        try:
            vectorstore = self.setup_vectorstore(namespace)
            result = {"previous": None, "current": None, "next": None}
            
            # Get previous chunk (if chunk_id > 0)
            if chunk_id > 0:
                try:
                    prev_docs = vectorstore.similarity_search(
                        query="",
                        k=50,  # Get more results to find the specific chunk
                        filter={"document_id": doc_id, "chunk_id": chunk_id - 1}
                    )
                    if prev_docs and len(prev_docs) > 0:
                        result["previous"] = prev_docs[0].page_content
                        print(f"📄 Found previous chunk {doc_id}_chunk_{chunk_id - 1}")
                except Exception as e:
                    print(f"⚠️  Could not get previous chunk: {e}")
            
            # Get current chunk
            try:
                current_docs = vectorstore.similarity_search(
                    query="",
                    k=50,
                    filter={"document_id": doc_id, "chunk_id": chunk_id}
                )
                if current_docs and len(current_docs) > 0:
                    result["current"] = current_docs[0].page_content
                    print(f"📄 Found current chunk {doc_id}_chunk_{chunk_id}")
            except Exception as e:
                print(f"⚠️  Could not get current chunk: {e}")
            
            # Get next chunk
            try:
                next_docs = vectorstore.similarity_search(
                    query="",
                    k=50,
                    filter={"document_id": doc_id, "chunk_id": chunk_id + 1}
                )
                if next_docs and len(next_docs) > 0:
                    result["next"] = next_docs[0].page_content
                    print(f"📄 Found next chunk {doc_id}_chunk_{chunk_id + 1}")
            except Exception as e:
                print(f"⚠️  Could not get next chunk: {e}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in get_adjacent_chunks_content: {e}")
            return {"previous": None, "current": None, "next": None}

    def delete_document(self, namespace: str, fileID: str) -> Dict[str, Any]:
        """
        Delete a document from the vector database.
        
        Args:
            namespace: Namespace containing the document
            fileID: Document ID to delete
            
        Returns:
            Dict containing deletion status
        """
        try:
            index = self._pc.Index(self._index_name)
            index.delete(
                filter={
                    "pdf_id": {"$eq": fileID}
                },
                namespace=namespace
            )
            
            return {
                "status": "success",
                "message": f"Document {fileID} deleted from vector database"
            }
        except Exception as e:
            logger.error(f"Error deleting document {fileID} from namespace {namespace}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error deleting document from vector database: {str(e)}"
            }

    def delete_namespace(self, namespace: str) -> Dict[str, Any]:
        """
        Delete an entire namespace from the Pinecone index.
        
        Args:
            namespace: The name of the namespace to delete.
            
        Returns:
            A dictionary with the status of the operation.
        """
        try:
            index = self._pc.Index(self._index_name)
            
            # First check if namespace exists by trying to get stats
            try:
                stats = index.describe_index_stats()
                namespaces = stats.get('namespaces', {})
                
                if namespace not in namespaces:
                    return {
                        "status": "success",
                        "message": f"Namespace {namespace} does not exist - nothing to delete."
                    }
                
                # If namespace exists, proceed with deletion
                index.delete(namespace=namespace, delete_all=True)
                
                # Clear cached vectorstore for this namespace
                if namespace in self._vectorstores:
                    del self._vectorstores[namespace]
                
                return {
                    "status": "success",
                    "message": f"Namespace {namespace} deleted from Pinecone."
                }
                
            except Exception as check_error:
                # If we can't check namespace existence, try deletion anyway
                # This handles cases where the index might exist but be empty
                index.delete(namespace=namespace, delete_all=True)
                
                # Clear cached vectorstore for this namespace
                if namespace in self._vectorstores:
                    del self._vectorstores[namespace]
                
                return {
                    "status": "success",
                    "message": f"Namespace {namespace} deletion attempted (existence check failed but deletion succeeded)."
                }
                
        except Exception as e:
            error_message = str(e)
            # Handle 404 errors more gracefully
            if "404" in error_message or "Not Found" in error_message:
                return {
                    "status": "success",
                    "message": f"Namespace {namespace} does not exist - nothing to delete."
                }
            return {
                "status": "error",
                "message": f"Error deleting namespace from Pinecone: {error_message}"
            }

    def get_vectorstore(self, namespace: str) -> PineconeVectorStore:
        """
        Get the vectorstore for a specific namespace.
        
        Args:
            namespace: Namespace identifier
            
        Returns:
            PineconeVectorStore instance
        """
        return self.setup_vectorstore(namespace)

    def get_embeddings(self) -> OpenAIEmbeddings:
        """
        Get the OpenAI embeddings instance.
        
        Returns:
            OpenAIEmbeddings instance
        """
        return self._embeddings

    def get_index_name(self) -> str:
        """
        Get the Pinecone index name.
        
        Returns:
            Index name string
        """
        return self._index_name 