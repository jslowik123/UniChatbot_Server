import os
from typing import Dict, List, Any, Optional, Tuple
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import logging
import math

load_dotenv()

# Constants
EMBEDDING_MODEL = "text-embedding-ada-002"
DEFAULT_BATCH_SIZE = 5  # Further reduced batch size for Pinecone uploads to prevent 2MB limit

logger = logging.getLogger(__name__)


class VectorManager:
    """
    Handles all vector database operations including indexing, retrieval, and management.
    
    This class manages Pinecone vector operations, document indexing, and chunk management
    for the document processing pipeline.
    """
    
    def __init__(self, pinecone_api_key: str, openai_api_key: str, index_name: str = "pdfs-index", batch_size: int = DEFAULT_BATCH_SIZE):
        """
        Initialize VectorManager with API keys and connections.
        
        Args:
            pinecone_api_key: API key for Pinecone vector database
            openai_api_key: API key for OpenAI embeddings
            index_name: Name of the Pinecone index
            batch_size: Maximum number of vectors to upload per batch
            
        Raises:
            ValueError: If required API keys are missing
        """
        if not pinecone_api_key or not openai_api_key:
            raise ValueError("Both Pinecone and OpenAI API keys are required")
            
        self._pinecone_api_key = pinecone_api_key
        self._openai_api_key = openai_api_key
        self._index_name = index_name
        self._batch_size = batch_size
        
        # Initialize Pinecone
        self._pc = Pinecone(api_key=pinecone_api_key)
        
        # Initialize OpenAI embeddings
        self._embeddings = OpenAIEmbeddings(api_key=openai_api_key, model=EMBEDDING_MODEL)
        
        # Cache for vectorstores per namespace
        self._vectorstores = {}
        
        # Ensure index exists
        self._ensure_index_exists()
        
        logger.info(f"VectorManager initialized with batch_size: {batch_size}")

    def _ensure_index_exists(self):
        """
        Ensure the Pinecone index exists, create if it doesn't.
        """
        try:
            existing_indexes = [index.name for index in self._pc.list_indexes()]
            if self._index_name not in existing_indexes:
                self._pc.create_index(
                    name=self._index_name,
                    dimension=1536,  # OpenAI embedding dimension
                    metric='cosine',
                    spec={
                        'serverless': {
                            'cloud': 'aws',
                            'region': 'us-east-1'
                        }
                    }
                )
                logger.info(f"Created new Pinecone index: {self._index_name}")
            else:
                logger.info(f"Using existing Pinecone index: {self._index_name}")
        except Exception as e:
            logger.error(f"Error ensuring index exists: {e}")
            raise

    def setup_vectorstore(self, namespace: str) -> PineconeVectorStore:
        """
        Set up vectorstore for a specific namespace.
        
        Args:
            namespace: Namespace identifier
            
        Returns:
            PineconeVectorStore instance for the namespace
        """
        if namespace not in self._vectorstores:
            self._vectorstores[namespace] = PineconeVectorStore(
                index_name=self._index_name,
                embedding=self._embeddings,
                namespace=namespace,
                pinecone_api_key=self._pinecone_api_key
            )
        return self._vectorstores[namespace]

    def _batch_upload_texts(self, vectorstore: PineconeVectorStore, texts: List[str], metadatas: List[Dict], ids: List[str]) -> None:
        """
        Upload texts to Pinecone in batches to avoid exceeding size limits.
        
        Args:
            vectorstore: PineconeVectorStore instance
            texts: List of text chunks to upload
            metadatas: List of metadata dictionaries
            ids: List of unique IDs for each text
            
        Raises:
            Exception: If any batch upload fails
        """
        total_items = len(texts)
        num_batches = math.ceil(total_items / self._batch_size)
        
        # Calculate total estimated size
        total_size_estimate = sum(len(text) for text in texts) + sum(len(str(meta)) for meta in metadatas)
        print(f"Uploading {total_items} items in {num_batches} batches")
        
        successful_batches = 0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self._batch_size
            end_idx = min((batch_idx + 1) * self._batch_size, total_items)
            
            batch_texts = texts[start_idx:end_idx]
            batch_metadatas = metadatas[start_idx:end_idx]
            batch_ids = ids[start_idx:end_idx]
            
            # print(f"📦 Uploading batch {batch_idx + 1}/{num_batches}: items {start_idx}-{end_idx-1} ({len(batch_texts)} items)")
            
            # Calculate approximate batch size for logging
            batch_size_estimate = sum(len(text) for text in batch_texts) + sum(len(str(meta)) for meta in batch_metadatas)
            # print(f"📊 Estimated batch size: ~{batch_size_estimate/1024:.1f} KB (~{batch_size_estimate/1024/1024:.3f} MB)")
            
            # Validate batch size isn't too large
            # if batch_size_estimate > 1.5 * 1024 * 1024:  # 1.5MB warning
                # print(f"⚠️  WARNING: Batch size {batch_size_estimate/1024/1024:.2f} MB is close to 2MB limit!")
            
            max_retries = 3
            batch_success = False
            
            for attempt in range(max_retries):
                try:
                    # print(f"🔄 Attempting upload batch {batch_idx + 1}/{num_batches}, attempt {attempt + 1}/{max_retries}")
                    vectorstore.add_texts(texts=batch_texts, metadatas=batch_metadatas, ids=batch_ids)
                    # print(f"✅ Successfully uploaded batch {batch_idx + 1}/{num_batches} on attempt {attempt + 1}")
                    batch_success = True
                    successful_batches += 1
                    break
                except Exception as upload_error:
                    error_msg = str(upload_error)
                    # print(f"❌ Batch {batch_idx + 1} attempt {attempt + 1} failed: {error_msg}")
                    
                    # Check if it's a size-related error
                    # if "exceeds the maximum supported size" in error_msg:
                        # print(f"🚨 SIZE LIMIT ERROR: Batch {batch_idx + 1} is too large even with batch_size={self._batch_size}")
                        # print(f"🔧 Consider reducing batch_size further or optimizing text content")
                    
                    if attempt == max_retries - 1:
                        print(f"Upload failed: Batch {batch_idx + 1} after {max_retries} attempts")
                        raise Exception(f"Failed to upload batch {batch_idx + 1} after {max_retries} attempts: {error_msg}")
                    
                    # Wait before retry (exponential backoff)
                    import time
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
            
            if not batch_success:
                raise Exception(f"Failed to upload batch {batch_idx + 1}")
        
        print(f"Upload completed: {successful_batches}/{num_batches} batches processed")

    def index_document(self, processed_pdf: Dict[str, Any], namespace: str, fileID: str) -> Dict[str, Any]:
        """
        Index processed PDF content in Pinecone vectorstore with batch uploading.
        
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
            
            # Create metadata for chunks
            for i, chunk in enumerate(processed_pdf["chunks"]):
                chunk_metadata = {
                    "document_id": fileID,
                    "chunk_id": i,
                    "original_filename": processed_pdf.get("original_file", "unknown"),
                    "chunk_type": "content"
                }
                
                # Add page information if available
                if i < len(chunks_with_pages):
                    chunk_with_page = chunks_with_pages[i]
                    if "pages" in chunk_with_page and chunk_with_page["pages"]:
                        # Convert pages to strings for Pinecone compatibility
                        pages_as_strings = [str(int(page)) for page in chunk_with_page["pages"]]
                        chunk_metadata["pages"] = pages_as_strings
                        # Add page range for consistency
                        if len(chunk_with_page['pages']) > 1:
                            chunk_metadata["page_range"] = f"{min(chunk_with_page['pages'])}-{max(chunk_with_page['pages'])}"
                        else:
                            chunk_metadata["page_range"] = str(chunk_with_page['pages'][0])
                
                metadatas.append(chunk_metadata)
            
            # Metadata for summary
            summary_metadata = {
                "document_id": fileID,
                "chunk_id": "summary",
                "original_filename": processed_pdf.get("original_file", "unknown"),
                "chunk_type": "summary"
            }
            metadatas.append(summary_metadata)
            
            # Handle special pages processing for image-based content
            special_pages_texts = []
            special_pages_metadatas = []
            special_pages_ids = []
            
            # Process special pages if available
            if "special_pages_data" in processed_pdf:
                special_pages_data = processed_pdf["special_pages_data"]
                
                for page_data in special_pages_data:
                    page_num = page_data.get("page_number", "unknown")
                    enhanced_text = page_data.get("enhanced_text", None)
                    if not enhanced_text or not enhanced_text.strip():
                        # Fallback: versuche normalen Text aus page_data['text'] zu nehmen
                        fallback_text = page_data.get("text", None)
                        if fallback_text and fallback_text.strip():
                            enhanced_text = fallback_text
                        else:
                            continue  # Seite überspringen
                    special_pages_texts.append(enhanced_text)
                    special_pages_metadatas.append({
                        "document_id": fileID,
                        "chunk_id": f"special_page_{page_num}",
                        "original_filename": processed_pdf.get("original_file", "unknown"),
                        "chunk_type": "special_page",
                        "page_number": str(page_num),
                        "pages": [str(page_num)],  # Add pages array for consistency
                        "page_range": str(page_num),  # Add page range for consistency
                        "enhanced_content": True
                    })
                    special_pages_ids.append(f"{fileID}_special_page_{page_num}")
            
            # Combine all data
            all_texts = texts + special_pages_texts
            all_metadatas = metadatas + special_pages_metadatas
            all_ids = [f"{fileID}_chunk_{i}" for i in range(len(processed_pdf["chunks"]))] + [f"{fileID}_summary"] + special_pages_ids
            
            print(f"Indexing document {fileID}: {len(all_texts)} text vectors")
            
            # Use batch upload to prevent exceeding Pinecone's 2MB limit
            self._batch_upload_texts(vectorstore, all_texts, all_metadatas, all_ids)
            
            if special_pages_texts:
                print(f"Special pages processed: {len(special_pages_texts)} additional vectors")
            
            return {
                "status": "success",
                "message": f"Document {fileID} indexed successfully with batch upload",
                "chunks": len(processed_pdf["chunks"]),
                "special_pages": len(special_pages_texts),
                "vector_ids": all_ids,
                "batches_used": math.ceil(len(all_texts) / self._batch_size)
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
                pass
            
            # Get previous chunk
            if prev_id:
                try:
                    prev_response = index.fetch(ids=[prev_id], namespace=namespace)
                    if prev_id in prev_response.get('vectors', {}):
                        result["previous"] = prev_id
                except Exception as e:
                    pass
            
            # Get next chunk
            try:
                next_response = index.fetch(ids=[next_id], namespace=namespace)
                if next_id in next_response.get('vectors', {}):
                    result["next"] = next_id
            except Exception as e:
                pass
            
            return result
            
        except Exception as e:
            pass
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
            pass
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
                except Exception as e:
                    pass
            
            # Get current chunk
            try:
                current_docs = vectorstore.similarity_search(
                    query="",
                    k=50,
                    filter={"document_id": doc_id, "chunk_id": chunk_id}
                )
                if current_docs and len(current_docs) > 0:
                    result["current"] = current_docs[0].page_content
            except Exception as e:
                pass
            
            # Get next chunk
            try:
                next_docs = vectorstore.similarity_search(
                    query="",
                    k=50,
                    filter={"document_id": doc_id, "chunk_id": chunk_id + 1}
                )
                if next_docs and len(next_docs) > 0:
                    result["next"] = next_docs[0].page_content
            except Exception as e:
                pass
            
            return result
            
        except Exception as e:
            pass
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