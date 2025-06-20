import os
import json
from typing import List, Dict, Any, Optional
import pdfplumber
from openai import OpenAI
import logging
from firebase_connection import get_firebase_client
from pinecone import Pinecone, ServerlessSpec
import hashlib
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentProcessor:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.firebase_db = get_firebase_client()
        
        # Initialize index
        index_name = "chatbot-index"
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        self.index = self.pc.Index(index_name)
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using pdfplumber"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Find the last complete sentence or paragraph
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                cut_point = max(last_period, last_newline)
                
                if cut_point > start + chunk_size // 2:
                    chunk = chunk[:cut_point + 1]
                    end = start + len(chunk)
            
            chunks.append(chunk.strip())
            start = end - overlap
            
        return [chunk for chunk in chunks if chunk]

    def get_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for text"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise

    def process_document(self, file_path: str, document_name: str, namespace: str) -> Dict[str, Any]:
        """Process document and store in vector database"""
        try:
            logger.info(f"Processing document: {document_name}")
            
            # Extract text from PDF
            text = self.extract_text_from_pdf(file_path)
            if not text:
                raise ValueError("No text extracted from PDF")
            
            # Create chunks
            chunks = self.chunk_text(text)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Store chunks in Pinecone
            vectors = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_{i}"
                embedding = self.get_embedding(chunk)
                
                metadata = {
                    "document_id": doc_id,
                    "document_name": document_name,
                    "chunk_index": i,
                    "text": chunk,
                    "namespace": namespace
                }
                
                vectors.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": metadata
                })
            
            # Upsert to Pinecone
            self.index.upsert(vectors=vectors, namespace=namespace)
            
            # Store document metadata in Firebase
            doc_ref = self.firebase_db.collection('documents').document(doc_id)
            doc_ref.set({
                'document_id': doc_id,
                'name': document_name,
                'namespace': namespace,
                'chunk_count': len(chunks),
                'processed_at': firestore.SERVER_TIMESTAMP,
                'file_size': os.path.getsize(file_path),
                'status': 'completed'
            })
            
            logger.info(f"Successfully processed document {document_name}")
            
            return {
                "document_id": doc_id,
                "document_name": document_name,
                "chunks_created": len(chunks),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    def query_documents(self, query: str, namespace: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query the vector database for relevant documents"""
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True
            )
            
            # Format results
            relevant_docs = []
            for match in results.matches:
                relevant_docs.append({
                    "text": match.metadata.get("text", ""),
                    "document_name": match.metadata.get("document_name", ""),
                    "document_id": match.metadata.get("document_id", ""),
                    "score": match.score,
                    "chunk_index": match.metadata.get("chunk_index", 0)
                })
            
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error querying documents: {str(e)}")
            return []

    def generate_response(self, query: str, context_docs: List[Dict[str, Any]], chat_history: List[Dict] = None) -> Dict[str, Any]:
        """Generate response using OpenAI with retrieved context"""
        try:
            # Prepare context
            context = "\n\n".join([
                f"Document: {doc['document_name']}\nContent: {doc['text']}"
                for doc in context_docs
            ])
            
            # Prepare chat history
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a helpful AI assistant that answers questions based on the provided documents. 
                    Use the following context to answer the user's question. If you cannot find the answer in the context, 
                    say so clearly. Be concise but comprehensive.
                    
                    Context:
                    {context}"""
                }
            ]
            
            # Add chat history if provided
            if chat_history:
                messages.extend(chat_history[-5:])  # Last 5 messages for context
            
            messages.append({"role": "user", "content": query})
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            # Calculate confidence score based on context relevance
            confidence_score = min(0.9, sum(doc['score'] for doc in context_docs[:3]) / 3) if context_docs else 0.3
            
            return {
                "answer": answer,
                "document_ids": list(set(doc["document_id"] for doc in context_docs)),
                "sources": [doc["text"][:200] + "..." for doc in context_docs[:3]],
                "confidence_score": round(confidence_score, 2),
                "context_used": len(context_docs) > 0,
                "additional_info": f"Based on {len(context_docs)} relevant document sections"
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again.",
                "document_ids": [],
                "sources": [],
                "confidence_score": 0.0,
                "context_used": False,
                "additional_info": "Error occurred during processing"
            }

# Import firestore for timestamp
try:
    from google.cloud import firestore
except ImportError:
    # Fallback if not available
    firestore = None 