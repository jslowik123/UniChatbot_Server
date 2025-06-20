import pdfplumber
import os
import io
from typing import Dict, List, Any, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from crewai import Agent, Task, Crew
from crewai.tools import tool
import pinecone
from langchain_pinecone import Pinecone as LangchainPinecone
from firebase_connection import FirebaseConnection
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import json

load_dotenv()

# Constants
EMBEDDING_MODEL = "text-embedding-ada-002"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TEMPERATURE = 0.7
GPT_MODEL = "gpt-4"


class StructuredResponse(BaseModel):
    """Structured response model for agent outputs."""
    answer: str = Field(description="Die ausführliche Antwort auf die Frage")
    document_ids: List[str] = Field(description="Liste der verwendeten Dokument-IDs")
    sources: List[str] = Field(description="Liste der Originaltext-Quellen, die die Antwort stützen")
    confidence_score: float = Field(description="Vertrauensscore der Antwort (0.0-1.0)")
    context_used: bool = Field(description="Ob Chat-History-Kontext verwendet wurde")
    additional_info: Optional[str] = Field(description="Zusätzliche Informationen oder Hinweise", default=None)


class AgentProcessor:
    """
    Handles document processing and chatbot interactions using CrewAI agents.
    
    Manages PDF extraction, text segmentation, embedding, and agentic RAG
    for intelligent question answering with structured outputs.
    """
    
    def __init__(self, pinecone_api_key: str, openai_api_key: str, index_name: str = "pdfs-index"):
        """
        Initialize AgentProcessor with API keys and connections.
        
        Args:
            pinecone_api_key: API key for Pinecone vector database
            openai_api_key: API key for OpenAI services
            index_name: Name of the Pinecone index
            
        Raises:
            ValueError: If required API keys are missing
        """
        if not pinecone_api_key or not openai_api_key:
            raise ValueError("Both Pinecone and OpenAI API keys are required")
            
        self._openai_api_key = openai_api_key
        self._pinecone_api_key = pinecone_api_key
        self._index_name = index_name
        
        # Initialize Pinecone
        self._pc = pinecone.Pinecone(api_key=pinecone_api_key)
        
        # Initialize OpenAI components
        self._llm = ChatOpenAI(api_key=openai_api_key, model=GPT_MODEL, temperature=DEFAULT_TEMPERATURE)
        self._embeddings = OpenAIEmbeddings(api_key=openai_api_key, model=EMBEDDING_MODEL)
        
        # Initialize text splitter
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize Firebase connection
        try:
            self._firebase = FirebaseConnection()
            self._firebase_available = True
        except ValueError as e:
            print(f"Firebase nicht verfügbar: {e}")
            self._firebase_available = False
        
        # Initialize vectorstore (will be set up per namespace)
        self._vectorstores = {}
        self._agents = {}

    def extract_pdf(self, file_content: bytes) -> Optional[Dict[str, Any]]:
        """
        Extract text and metadata from PDF content using pdfplumber.
        
        Args:
            file_content: Raw PDF file content as bytes
            
        Returns:
            Dict containing extracted text and metadata, or None if extraction fails
        """
        try:
            pdf_file = io.BytesIO(file_content)
            with pdfplumber.open(pdf_file) as pdf:
                text = "".join(page.extract_text() or "" for page in pdf.pages)
                metadata = pdf.metadata if hasattr(pdf, 'metadata') else {}
            return {"text": text, "metadata": metadata}
        except Exception as e:
            print(f"Fehler beim PDF-Verarbeiten: {e}")
            return None

    def process_pdf_content(self, pdf_data: Dict[str, Any], filename: str) -> Optional[Dict[str, Any]]:
        """
        Process extracted PDF content by chunking and summarizing.
        
        Args:
            pdf_data: Dictionary containing text and metadata from PDF
            filename: Original filename for identification
            
        Returns:
            Dict containing processed chunks and summary, or None if processing fails
        """
        if not pdf_data or not pdf_data.get("text"):
            return None
        
        try:
            # Split text into chunks
            chunks = self._text_splitter.split_text(pdf_data["text"])
            
            # Generate summary
            summary_prompt = f"Fasse diesen Text zusammen (max. 1000 Zeichen): {pdf_data['text'][:10000]}"
            summary = self._llm.invoke(summary_prompt).content
            
            return {
                "pdf_id": filename,
                "chunks": chunks,
                "summary": summary,
                "metadata": pdf_data["metadata"]
            }
        except Exception as e:
            print(f"Fehler beim Verarbeiten der PDF-Inhalte: {e}")
            return None

    def setup_vectorstore(self, namespace: str) -> LangchainPinecone:
        """
        Set up or retrieve vectorstore for a specific namespace.
        
        Args:
            namespace: Namespace identifier
            
        Returns:
            LangchainPinecone vectorstore instance
        """
        if namespace in self._vectorstores:
            return self._vectorstores[namespace]
        
        # Ensure index exists
        self._ensure_index_exists()
        
        # Create vectorstore for this namespace
        vectorstore = LangchainPinecone.from_existing_index(
            index_name=self._index_name, 
            embedding=self._embeddings,
            namespace=namespace
        )
        
        self._vectorstores[namespace] = vectorstore
        return vectorstore

    def _ensure_index_exists(self):
        """Ensure the Pinecone index exists, create if necessary."""
        try:
            # Check if index exists
            existing_indexes = [idx.name for idx in self._pc.list_indexes()]
            
            if self._index_name not in existing_indexes:
                self._pc.create_index(
                    name=self._index_name,
                    dimension=1536,  # text-embedding-ada-002 dimension
                    metric="cosine",
                    spec=pinecone.ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                # Wait for index to be ready
                import time
                time.sleep(60)  # Wait for index initialization
        except Exception as e:
            print(f"Error ensuring index exists: {e}")

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
            
            # Prepare texts and metadata
            texts = processed_pdf["chunks"] + [processed_pdf["summary"]]
            metadatas = [
                {"pdf_id": fileID, "type": "chunk", "chunk_id": i} 
                for i in range(len(processed_pdf["chunks"]))
            ] + [{"pdf_id": fileID, "type": "summary"}]
            
            # Add texts to vectorstore
            vectorstore.add_texts(texts=texts, metadatas=metadatas)
            
            return {
                "status": "success",
                "message": f"Document {fileID} indexed successfully",
                "chunks": len(processed_pdf["chunks"])
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error indexing document: {str(e)}"
            }

    def setup_agent(self, namespace: str) -> Tuple[Agent, LangchainPinecone]:
        """
        Set up CrewAI agent with RAG capabilities for a specific namespace.
        
        Args:
            namespace: Namespace identifier
            
        Returns:
            Tuple containing the configured agent and vectorstore
        """
        if namespace in self._agents:
            return self._agents[namespace], self._vectorstores[namespace]
        
        vectorstore = self.setup_vectorstore(namespace)
        
        # Multi-Query-Retriever
        pdf_retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            llm=self._llm
        )
        
        # Kontextkompression
        compressor = LLMChainExtractor.from_llm(self._llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=pdf_retriever
        )
        
        # PDF Retriever Tool für CrewAI
        @tool("PDF Search Tool")
        def pdf_search_tool(query: str) -> str:
            """Durchsucht PDF-Dokumente nach relevanten Informationen basierend auf der Suchanfrage."""
            try:
                docs = compression_retriever.get_relevant_documents(query)
                results = []
                for doc in docs:
                    # Extract metadata for structured response
                    doc_id = doc.metadata.get('pdf_id', 'unknown')
                    content = doc.page_content
                    results.append(f"[DOC_ID: {doc_id}] {content}")
                return "\n\n".join(results)
            except Exception as e:
                return f"Fehler beim Durchsuchen der Dokumente: {str(e)}"
        
        # Agent definieren
        researcher = Agent(
            role="University Research Assistant",
            goal="Beantworte Fragen mit PDF- und Universitätsdaten präzise und umfassend in strukturiertem Format",
            backstory="""Du bist ein erfahrener universitärer Forschungsassistent, der PDF-Dokumente und universitäre 
            Informationen nutzt, um präzise und umfassende Antworten zu geben. Du sprichst Deutsch und hilfst 
            Studierenden bei ihren Fragen. Du gibst IMMER strukturierte Antworten im JSON-Format zurück.""",
            llm=self._llm,
            tools=[pdf_search_tool],
            verbose=True,
            allow_delegation=False
        )
        
        self._agents[namespace] = researcher
        return researcher, vectorstore

    def answer_question(self, question: str, namespace: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Answer a question using the CrewAI agent for the specified namespace with structured output.
        
        Args:
            question: User's question
            namespace: Namespace to search within
            chat_history: Previous chat history for context
            
        Returns:
            Structured response containing answer and metadata
        """
        try:
            researcher, _ = self.setup_agent(namespace)
            
            # Prepare chat history context
            chat_context = ""
            has_history = bool(chat_history and len(chat_history) > 0)
            
            if has_history:
                recent_messages = chat_history[-6:]  # Last 6 messages for context
                context_parts = []
                for msg in recent_messages:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    context_parts.append(f"{role.upper()}: {content}")
                chat_context = "\n".join(context_parts)
            
            # Create structured task description
            task_description = f"""
🔄 CHAT HISTORY BEACHTUNG - WICHTIG:
{'Du siehst eine Chat History mit vorherigen Nachrichten. BERÜCKSICHTIGE diese aktiv:' if has_history else 'Dies ist eine neue Unterhaltung ohne vorherige Chat History.'}
{'''
- Beziehe dich auf vorherige Fragen und Antworten
- Nutze den Kontext aus früheren Nachrichten  
- Wenn der Nutzer "dazu", "darüber", "das" oder ähnliche Bezugswörter verwendet, beziehe dich auf vorherige Themen
- Beantworte Rückfragen oder Nachfragen basierend auf dem bisherigen Gesprächsverlauf
- Vermeide Wiederholungen bereits gegebener Antworten, es sei denn, es wird explizit verlangt
- Erkenne den Kontext der aktuellen Frage im Zusammenhang mit der Chat History

VORHERIGE UNTERHALTUNG:
''' + chat_context if has_history else ''}

AKTUELLE FRAGE: {question}

DEINE AUFGABE:
1. Durchsuche relevante Dokumente mit dem PDF Search Tool
2. Analysiere die gefundenen Informationen
3. Beziehe Chat-History-Kontext ein, falls vorhanden
4. Erstelle eine strukturierte, präzise Antwort

VERHALTEN:
- Stütze deine Antworten auf die bereitgestellten Quellen
- Antworte natürlich und direkt, als würdest du mit Studierenden sprechen
- Bei Widersprüchen: Bevorzuge hochschulspezifische Informationen
- Bei fehlenden Informationen: Sage es klar und biete Hilfe an
- Gib ausführliche, aber präzise Antworten
- Verwende NIEMALS Anführungszeichen in der JSON-Antwort

WICHTIG: Deine Antwort MUSS in folgendem JSON-Format sein:
{{
    "answer": "Deine ausführliche Antwort hier ohne Anführungszeichen",
    "document_ids": ["doc1", "doc2"],
    "sources": ["Originaltext aus den Dokumenten, der die Antwort stützt"],
    "confidence_score": 0.9,
    "context_used": {str(has_history).lower()},
    "additional_info": "Zusätzliche Hinweise oder null"
}}
"""
            
            task = Task(
                description=task_description,
                expected_output="Eine strukturierte JSON-Antwort mit answer, document_ids, sources, confidence_score, context_used und additional_info Feldern. Die Antwort sollte präzise, gut begründet und in deutscher Sprache verfasst sein.",
                agent=researcher
            )
            
            crew = Crew(
                agents=[researcher], 
                tasks=[task],
                verbose=True
            )
            
            result = crew.kickoff()
            result_str = str(result)
            
            # Try to parse the JSON response
            try:
                # Extract JSON from the response
                json_start = result_str.find('{')
                json_end = result_str.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_str = result_str[json_start:json_end]
                    parsed_response = json.loads(json_str)
                    
                    # Validate and structure the response
                    structured_response = {
                        "answer": parsed_response.get("answer", result_str),
                        "document_ids": parsed_response.get("document_ids", []),
                        "sources": parsed_response.get("sources", []),
                        "confidence_score": float(parsed_response.get("confidence_score", 0.8)),
                        "context_used": bool(parsed_response.get("context_used", has_history)),
                        "additional_info": parsed_response.get("additional_info")
                    }
                    
                    return structured_response
                else:
                    # Fallback if JSON parsing fails
                    return {
                        "answer": result_str,
                        "document_ids": [],
                        "sources": [],
                        "confidence_score": 0.7,
                        "context_used": has_history,
                        "additional_info": "Antwort konnte nicht als strukturiertes JSON geparst werden"
                    }
                    
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "answer": result_str,
                    "document_ids": [],
                    "sources": [],
                    "confidence_score": 0.7,
                    "context_used": has_history,
                    "additional_info": "JSON-Parsing fehlgeschlagen, Rohtext-Antwort bereitgestellt"
                }
            
        except Exception as e:
            return {
                "answer": f"Fehler beim Beantworten der Frage: {str(e)}",
                "document_ids": [],
                "sources": [],
                "confidence_score": 0.0,
                "context_used": False,
                "additional_info": f"Fehler aufgetreten: {type(e).__name__}"
            }

    def process_document_full(self, file_content: bytes, namespace: str, fileID: str, filename: str) -> Dict[str, Any]:
        """
        Complete document processing pipeline from PDF to indexed content.
        
        Args:
            file_content: Raw PDF file content
            namespace: Namespace for organization
            fileID: Unique document identifier
            filename: Original filename
            
        Returns:
            Dict containing processing results and status
        """
        try:
            # Step 1: Extract PDF content
            pdf_data = self.extract_pdf(file_content)
            if not pdf_data:
                return {
                    "status": "error",
                    "message": "Failed to extract PDF content"
                }
            
            # Step 2: Process content (chunk and summarize)
            processed_pdf = self.process_pdf_content(pdf_data, filename)
            if not processed_pdf:
                return {
                    "status": "error",
                    "message": "Failed to process PDF content"
                }
            
            # Step 3: Index in Pinecone
            indexing_result = self.index_document(processed_pdf, namespace, fileID)
            if indexing_result["status"] != "success":
                return indexing_result
            
            # Step 4: Store metadata in Firebase if available
            firebase_result = {"status": "success", "message": "Firebase not available"}
            if self._firebase_available:
                firebase_result = self._firebase.append_metadata(
                    namespace=namespace,
                    fileID=fileID,
                    chunks=len(processed_pdf["chunks"]),
                    keywords=[],  # Could be extracted if needed
                    summary=processed_pdf["summary"]
                )
            
            return {
                "status": "success",
                "message": f"Document {filename} processed successfully",
                "chunks": len(processed_pdf["chunks"]),
                "pinecone_result": indexing_result,
                "firebase_result": firebase_result
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error in document processing pipeline: {str(e)}"
            }

    def delete_document(self, namespace: str, fileID: str) -> Dict[str, Any]:
        """
        Delete document from vectorstore and Firebase.
        
        Args:
            namespace: Document namespace
            fileID: Document identifier
            
        Returns:
            Dict containing deletion status
        """
        try:
            # Delete from Pinecone
            vectorstore = self.setup_vectorstore(namespace)
            # Note: We need to delete by metadata filter
            # This requires using the Pinecone client directly
            index = self._pc.Index(self._index_name)
            index.delete(
                namespace=namespace,
                filter={"pdf_id": fileID}
            )
            
            # Delete from Firebase if available
            firebase_result = {"status": "success", "message": "Firebase not available"}
            if self._firebase_available:
                firebase_result = self._firebase.delete_document_metadata(namespace, fileID)
            
            return {
                "status": "success",
                "message": f"Document {fileID} deleted successfully",
                "firebase_result": firebase_result
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error deleting document: {str(e)}"
            }

    def get_namespace_summary(self, namespace: str) -> Dict[str, Any]:
        """
        Generate a summary of all documents in a namespace.
        
        Args:
            namespace: Namespace identifier
            
        Returns:
            Dict containing namespace summary
        """
        try:
            if self._firebase_available:
                namespace_data = self._firebase.get_namespace_data(namespace)
                if namespace_data:
                    documents = []
                    for doc_id, doc_data in namespace_data.items():
                        if isinstance(doc_data, dict):
                            documents.append({
                                "id": doc_id,
                                "summary": doc_data.get("summary", ""),
                                "chunks": doc_data.get("chunks", 0)
                            })
                    
                    return {
                        "status": "success",
                        "namespace": namespace,
                        "document_count": len(documents),
                        "documents": documents
                    }
            
            return {
                "status": "success",
                "namespace": namespace,
                "document_count": 0,
                "documents": [],
                "message": "No documents found or Firebase not available"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error getting namespace summary: {str(e)}"
            } 