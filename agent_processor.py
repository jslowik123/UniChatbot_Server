import pdfplumber
import os
import io
from typing import Dict, List, Any, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from crewai import Agent, Task, Crew
from crewai.tools import tool
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from firebase_connection import FirebaseConnection
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import json
import logging
import traceback

load_dotenv()

# Constants
EMBEDDING_MODEL = "text-embedding-ada-002"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TEMPERATURE = 0.7
GPT_MODEL = "gpt-4.1-mini"

logger = logging.getLogger(__name__)

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
        self._pc = Pinecone(api_key=pinecone_api_key)
        
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
        
        # Ensure index exists
        self._ensure_index_exists()
        
        # Create vectorstore for this namespace
        vectorstore = PineconeVectorStore.from_existing_index(
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
                {"pdf_id": fileID, "document_id": fileID, "type": "chunk", "chunk_id": i} 
                for i in range(len(processed_pdf["chunks"]))
            ] + [{"pdf_id": fileID, "document_id": fileID, "type": "summary"}]
            
            # Add texts to vectorstore und speichere die IDs
            ids = [f"{fileID}_chunk_{i}" for i in range(len(processed_pdf["chunks"]))] + [f"{fileID}_summary"]
            vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            # Speichere die IDs in Firebase
            if self._firebase_available:
                self._firebase.append_metadata(
                    namespace=namespace,
                    fileID=fileID,
                    chunk_count=len(processed_pdf["chunks"]),
                    keywords=[],
                    summary=processed_pdf["summary"],
                    vector_ids=ids
                )
            
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

    def setup_agent(self, namespace: str) -> Tuple[Agent, PineconeVectorStore]:
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
        @tool("Document Overview Tool")
        def document_overview_tool() -> str:
            """Zeigt eine Übersicht aller verfügbaren Dokumente im aktuellen Namespace mit deren Zusammenfassungen und IDs."""
            try:
                summary_data = self.get_namespace_summary(namespace)  # namespace from closure
                print(f"📋 Document Overview für namespace: {namespace}")
                if summary_data["status"] != "success":
                    return f"FEHLER: {summary_data.get('message', 'Unbekannter Fehler beim Abrufen der Dokumentübersicht')}"
                
                if summary_data["document_count"] == 0:
                    return "KEINE DOKUMENTE: In diesem Namespace sind aktuell keine Dokumente verfügbar."
                
                overview_parts = [
                    f"📚 DOKUMENTÜBERSICHT ({summary_data['document_count']} Dokumente verfügbar):",
                    "=" * 60
                ]
                
                for doc in summary_data["documents"]:
                    doc_info = [
                        f"🔹 DOKUMENT-ID: {doc['id']}",
                        f"   Name: {doc.get('name', doc['id'])}",
                        f"   Status: {doc.get('status', 'Unknown')}",
                        f"   Chunks: {doc.get('chunk_count', 0)}",
                        f"   Datum: {doc.get('date', 'Unbekannt')}",
                        f"   Zusammenfassung: {doc['summary'][:200]}{'...' if len(doc['summary']) > 200 else ''}",
                    ]
                    # Add additional_info if present
                    if doc.get('additional_info'):
                        doc_info.append(f"   Zusätzliche Info: {doc['additional_info']}")
                    doc_info.append("")  # Empty line for separation
                    overview_parts.extend(doc_info)
                
                overview_parts.append("💡 Verwende das 'PDF Search Tool' mit spezifischen Dokument-IDs für detaillierte Suchen!")
                
                return "\n".join(overview_parts)
                
            except Exception as e:
                return f"FEHLER beim Abrufen der Dokumentübersicht: {str(e)}"

        @tool("PDF Search Tool")
        def pdf_search_tool(query: str, document_ids: str = "") -> str:
            """Durchsucht PDF-Dokumente nach relevanten Informationen. 
            
            Args:
                query: Die Suchanfrage
                document_ids: Optional - Komma-getrennte Liste von Dokument-IDs um die Suche zu filtern (z.B. 'doc1,doc2')
            """
            try:
                print(f"🔍 PDF Search für Query: '{query}' in namespace: {namespace}")
                
                # Parse document IDs filter if provided
                target_doc_ids = None
                if document_ids.strip():
                    target_doc_ids = [doc_id.strip() for doc_id in document_ids.split(',') if doc_id.strip()]
                    print(f"🎯 Suche beschränkt auf Dokumente: {target_doc_ids}")
                
                docs = compression_retriever.invoke(query)
                
                if not docs:
                    return "KEINE DOKUMENTE GEFUNDEN: Es wurden keine relevanten Dokumente für diese Anfrage gefunden. Der Namespace könnte leer sein oder die Anfrage passt zu keinem verfügbaren Inhalt."
                
                # Filter by document IDs if specified
                if target_doc_ids:
                    filtered_docs = []
                    for doc in docs:
                        doc_id = doc.metadata.get('document_id', doc.metadata.get('pdf_id', 'unknown'))
                        if doc_id in target_doc_ids:
                            filtered_docs.append(doc)
                    docs = filtered_docs
                    
                    if not docs:
                        return f"KEINE DOKUMENTE IN GEFILTERTEN IDs: Es wurden keine relevanten Dokumente in den spezifizierten Dokument-IDs {target_doc_ids} gefunden."
                
                results = []
                found_doc_ids = set()
                for doc in docs:
                    # Extract metadata for structured response
                    doc_id = doc.metadata.get('document_id', doc.metadata.get('pdf_id', 'unknown'))
                    found_doc_ids.add(doc_id)
                    content = doc.page_content
                    results.append(f"[DOC_ID: {doc_id}] {content}")
                
                if not results:
                    return "KEINE RELEVANTEN INHALTE: Dokumente wurden gefunden, aber sie enthalten keine relevanten Informationen für diese Anfrage."
                
                # Add document IDs info for the agent to use
                doc_ids_list = list(found_doc_ids)
                result_text = "\n\n".join(results)
                result_text += f"\n\n[SYSTEM_INFO] FOUND_DOCUMENT_IDS: {doc_ids_list}"
                
                if target_doc_ids:
                    result_text += f"\n[SYSTEM_INFO] FILTERED_BY_DOC_IDS: {target_doc_ids}"
                
                print(f"✅ PDF Search erfolgreich: {len(results)} Ergebnisse, Dokument-IDs: {doc_ids_list}")
                return result_text
                
            except Exception as e:
                error_msg = f"FEHLER BEIM DURCHSUCHEN: {str(e)}"
                print(f"❌ PDF Search Fehler: {error_msg}")
                return error_msg
        
        # Agent definieren
        researcher = Agent(
            role="Hilfsbereit Studienbuddy",
            goal="Sei ein natürlicher, freundlicher Gesprächspartner für Studierende. Führe normale Unterhaltungen und nutze Dokumente nur wenn sie wirklich relevant sind.",
            backstory="""Du bist ein entspannter, hilfsbereiter Studienbuddy der Deutsch spricht. Du kannst über alles reden - 
            Studienfragen, alltägliche Dinge, Probleme oder einfach plaudern. Wenn jemand spezifische Fragen hat, die in 
            den verfügbaren Dokumenten beantwortet werden können, dann suchst du gerne nach - aber das ist nicht dein 
            Hauptfokus. Du bist erstmal ein normaler Gesprächspartner, der hilft wo er kann. Sei locker, freundlich und 
            authentisch. Du musst nicht immer in Dokumenten suchen - manchmal reicht dein Allgemeinwissen völlig aus.
            
            Du hast zwei Tools zur Verfügung:
            1. Document Overview Tool - um zu sehen welche Dokumente verfügbar sind
            2. PDF Search Tool - um in spezifischen oder allen Dokumenten zu suchen""",
            llm=self._llm,
            tools=[document_overview_tool, pdf_search_tool],
            verbose=False,  # Disable verbose to avoid parsing issues
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
Hey! {'''Du siehst hier unsere bisherige Unterhaltung - schau sie dir an und beziehe dich darauf:

BISHERIGE UNTERHALTUNG:
''' + chat_context if has_history else 'Das ist der Anfang unserer Unterhaltung!'}

AKTUELLE NACHRICHT: {question}

WICHTIG - TOOL USAGE REGELN:
Du MUSST deine Tools verwenden für folgende Fragen:
• Jede Frage nach "welche Dokumente", "verfügbare Dokumente" oder ähnliches → Document Overview Tool
• Jede spezifische Frage zu Studiengängen, Modulen, Kursen → PDF Search Tool  
• Fragen zu Prüfungen, Regelungen, Terminen → PDF Search Tool
• Wenn nach spezifischen Informationen aus Dokumenten gefragt wird → PDF Search Tool

WIE DU ANTWORTEN SOLLST:
• Sei natürlich und gesprächig - wie ein echter Studienbuddy
• Für dokumentenbasierte Fragen: VERWENDE IMMER zuerst die entsprechenden Tools
• Bei Small Talk oder allgemeinen Fragen kannst du ohne Tools antworten
• Sei locker, freundlich und authentisch
• Beziehe dich auf vorherige Nachrichten wenn relevant

TOOL-WORKFLOW (BEFOLGE DAS GENAU):
1. WENN die Frage nach Dokumenten fragt → Document Overview Tool aufrufen
2. WENN nach spezifischen Inhalten gefragt wird → PDF Search Tool aufrufen
3. ERST DANN antworte basierend auf den Tool-Ergebnissen

ANTWORTFORMAT - WICHTIG:
Deine Antwort muss IMMER in diesem JSON-Format sein (ohne Markdown-Blöcke):
{{
    "answer": "Deine natürliche, freundliche Antwort hier",
    "document_ids": ["extrahiere diese aus [SYSTEM_INFO] FOUND_DOCUMENT_IDS wenn du das PDF Search Tool verwendet hast"],
    "sources": ["Hier übernimmst du 1zu1 die sätze aus den Quellen die du verwendet hast, in derselben Reihenfolge wie die Dokumenten IDs, die Sätze kannst du richtig formatieren."],
    "confidence_score": 0.9,
    "context_used": {str(has_history).lower()},
    "additional_info": "Zusätzliche Hinweise oder null"
}}

WICHTIG FÜR DOCUMENT_IDS:
- Wenn du das PDF Search Tool verwendest, extrahiere die document_ids aus der Zeile "[SYSTEM_INFO] FOUND_DOCUMENT_IDS: [...]" 
- Wenn du das Tool nicht verwendest, lass document_ids leer: []
- Verwende nur die echten document_ids aus den Suchergebnissen, erfinde keine!
- Gib nur die document_ids aus, die du auch wirklich verwendet hast um die Antwort zu erstellen!

WICHTIG: Verwende deine Tools aktiv! Das ist der Hauptzweck deiner Existenz.
"""
            
            task = Task(
                description=task_description,
                expected_output="Eine natürliche, freundliche Antwort im JSON-Format. Nutze Dokumente nur wenn wirklich nötig. Sei gesprächig und authentisch wie ein echter Studienbuddy.",
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
                    chunk_count=len(processed_pdf["chunks"]),
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
        pinecone_deleted = False
        firebase_deleted = False
        try:
            index = self._pc.Index(self._index_name)
            index.delete(
            filter={
                "pdf_id": {"$eq": fileID}
            },
            namespace=namespace
            )
            pinecone_deleted = True
            
            # Firebase-Löschung immer ausführen!
            firebase_result = {"status": "success", "message": "Firebase not available"}
            if self._firebase_available:
                firebase_result = self._firebase.delete_document_metadata(namespace, fileID)
                if firebase_result.get("status") == "success":
                    firebase_deleted = True
            
            return {
                "status": "success",
                "message": f"Document {fileID} deleted. Pinecone: {pinecone_deleted}, Firebase: {firebase_deleted}"
            }
        except Exception as e:
            logger.error(f"Fehler beim Löschen des Dokuments {fileID} im Namespace {namespace}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "message": f"Error deleting document: {str(e)}"
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
                return {
                    "status": "success",
                    "message": f"Namespace {namespace} deleted from Pinecone."
                }
                
            except Exception as check_error:
                # If we can't check namespace existence, try deletion anyway
                # This handles cases where the index might exist but be empty
                index.delete(namespace=namespace, delete_all=True)
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

    def get_namespace_summary(self, namespace: str):
        """
        Generate a summary of all documents in a namespace.

        Args:
            namespace: Namespace identifier

        Returns:
            Dict containing namespace summary
        """
        try:
            project_info = None
            if self._firebase_available:
                project_info_result = self._firebase.get_project_info(namespace)
                if project_info_result.get("status") == "success":
                    project_info = project_info_result.get("info")

                firebase_result = self._firebase.get_namespace_data(namespace)
                if firebase_result.get("status") == "success" and firebase_result.get("data"):
                    namespace_data = firebase_result["data"]
                    documents = []
                    for doc_id, doc_data in namespace_data.items():
                        if isinstance(doc_data, dict) and (
                            "summary" in doc_data or "status" in doc_data or "chunk_count" in doc_data
                        ):
                            document_info = {
                                "id": doc_id,
                                "name": doc_data.get("name", doc_id),
                                "summary": doc_data.get("summary", "Keine Zusammenfassung verfügbar"),
                                "chunk_count": doc_data.get("chunk_count", doc_data.get("chunks", 0)),
                                "status": doc_data.get("status", "Unknown"),
                                "date": doc_data.get("date", ""),
                                "processing": doc_data.get("processing", False),
                                "progress": doc_data.get("progress", 0),
                                "path": doc_data.get("path", ""),
                                "storageURL": doc_data.get("storageURL", "")
                            }
                            if "additional_info" in doc_data:
                                document_info["additional_info"] = doc_data["additional_info"]
                            documents.append(document_info)
                    return {
                        "status": "success",
                        "namespace": namespace,
                        "document_count": len(documents),
                        "documents": documents,
                        "project_info": project_info
                    }
            return {
                "status": "success",
                "namespace": namespace,
                "document_count": 0,
                "documents": [],
                "project_info": project_info,
                "message": "No documents found or Firebase not available"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error getting namespace summary: {str(e)}"
            }

    def get_documents(self, namespace: str) -> str:
        """
        Get a simple string representation of all documents in a namespace for assessment purposes.
        
        Args:
            namespace: Namespace identifier
            
        Returns:
            String containing document overview
        """
        try:
            summary_data = self.get_namespace_summary(namespace)
            
            if summary_data["status"] != "success" or summary_data["document_count"] == 0:
                return "Keine Dokumente im Namespace verfügbar."
            
            documents_info = []
            for doc in summary_data["documents"]:
                doc_text = f"- Dokument: {doc.get('name', doc['id'])}\n"
                doc_text += f"  ID: {doc['id']}\n"
                doc_text += f"  Status: {doc.get('status', 'Unknown')}\n"
                doc_text += f"  Chunks: {doc.get('chunk_count', 0)}\n"
                doc_text += f"  Zusammenfassung: {doc['summary']}\n"
                if doc.get('additional_info'):
                    doc_text += f"  Zusätzliche Info: {doc['additional_info']}\n"
                documents_info.append(doc_text)
            
            return f"Dokumente im Namespace '{namespace}' ({summary_data['document_count']} Dokumente):\n\n" + "\n".join(documents_info)
            
        except Exception as e:
            return f"Fehler beim Abrufen der Dokumente: {str(e)}" 
        
    