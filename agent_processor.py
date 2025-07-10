import os
from typing import Dict, List, Any, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from crewai import Agent, Task, Crew
from crewai.tools import tool
from firebase_connection import FirebaseConnection
from doc_processor import DocProcessor
from vector_manager import VectorManager
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import json
import logging
import traceback

load_dotenv()

# Constants
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 150  # Reduced to 100-200 token range (roughly 150 characters)
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
        
        # Initialize OpenAI LLM
        self._llm = ChatOpenAI(api_key=openai_api_key, model=GPT_MODEL, temperature=DEFAULT_TEMPERATURE)
        
        # Initialize VectorManager for all vector operations
        self._vector_manager = VectorManager(pinecone_api_key, openai_api_key, index_name)
        
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
        
        # Initialize DocProcessor for PDF processing
        self._doc_processor = DocProcessor(pinecone_api_key, openai_api_key)
        
        # Initialize agents cache
        self._agents = {}




    def setup_vectorstore(self, namespace: str):
        """
        Set up or retrieve vectorstore for a specific namespace.
        
        Args:
            namespace: Namespace identifier
            
        Returns:
            PineconeVectorStore vectorstore instance
        """
        return self._vector_manager.setup_vectorstore(namespace)

    def get_adjacent_chunks(self, namespace: str, chunk_id: str) -> Dict[str, str]:
        """
        Retrieve adjacent chunks (previous and next) for a given chunk ID.
        
        Args:
            namespace: Namespace to search in
            chunk_id: ID of the main chunk (format: fileID_chunk_X)
            
        Returns:
            Dict containing previous, current, and next chunk content
        """
        return self._vector_manager.get_adjacent_chunks(namespace, chunk_id)

    def get_chunk_content_by_id(self, namespace: str, chunk_id: str) -> Optional[str]:
        """
        Get the actual text content of a chunk by its ID.
        
        Args:
            namespace: Namespace to search in
            chunk_id: ID of the chunk
            
        Returns:
            Text content of the chunk or None if not found
        """
        return self._vector_manager.get_chunk_content_by_id(namespace, chunk_id)

    def _get_adjacent_chunks_content(self, namespace: str, doc_id: str, chunk_id: int) -> Dict[str, Optional[str]]:
        """
        Get the actual content of adjacent chunks for a given chunk.
        
        Args:
            namespace: Namespace to search in
            doc_id: Document ID
            chunk_id: Chunk number (integer)
            
        Returns:
            Dict containing previous, current, and next chunk content
        """
        return self._vector_manager.get_adjacent_chunks_content(namespace, doc_id, chunk_id)



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
        # Delegate to vector manager
        indexing_result = self._vector_manager.index_document(processed_pdf, namespace, fileID)
        
        # If indexing was successful, also save to Firebase
        if indexing_result["status"] == "success" and self._firebase_available:
            vector_ids = indexing_result.get("vector_ids", [])
            self._firebase.append_metadata(
                namespace=namespace,
                fileID=fileID,
                chunk_count=len(processed_pdf["chunks"]),
                keywords=[],
                summary=processed_pdf["summary"],
                vector_ids=vector_ids
            )
        
        return indexing_result

    def setup_agent(self, namespace: str) -> Tuple[Agent, Any]:
        """
        Set up CrewAI agent with RAG capabilities for a specific namespace.
        
        Args:
            namespace: Namespace identifier
            
        Returns:
            Tuple containing the configured agent and vectorstore
        """
        if namespace in self._agents:
            return self._agents[namespace], self._vector_manager.get_vectorstore(namespace)
        
        vectorstore = self.setup_vectorstore(namespace)
        
        # Get document overview for the system prompt
        documents_overview = self._get_documents_overview_for_prompt(namespace)
        
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
                
                # Input validation
                if not query or not isinstance(query, str) or not query.strip():
                    return "FEHLER: Suchanfrage ist leer oder ungültig."
                
                # Parse document IDs filter if provided
                target_doc_ids = None
                if document_ids and document_ids.strip():
                    target_doc_ids = [doc_id.strip() for doc_id in document_ids.split(',') if doc_id.strip()]
                    print(f"🎯 Suche beschränkt auf Dokumente: {target_doc_ids}")
                
                # Perform search with error handling
                docs = None
                try:
                    docs = compression_retriever.invoke(query)
                except Exception as search_error:
                    print(f"❌ Fehler beim Pinecone-Abruf: {str(search_error)}")
                    return f"FEHLER BEI PINECONE-SUCHE: {str(search_error)}"
                
                # Validate Pinecone response structure
                if docs is None:
                    return "FEHLER: Pinecone-Antwort ist None - möglicherweise Verbindungsproblem."
                
                if not isinstance(docs, list):
                    print(f"⚠️  Unerwarteter Pinecone-Response-Typ: {type(docs)}")
                    return f"FEHLER: Unerwarteter Response-Typ von Pinecone: {type(docs)}"
                
                if not docs:
                    return "KEINE DOKUMENTE GEFUNDEN: Keine relevanten Dokumente gefunden."
                
                print(f"📄 Pinecone lieferte {len(docs)} Dokumente")
                
                # Filter by document IDs if specified
                if target_doc_ids:
                    filtered_docs = []
                    for doc in docs:
                        try:
                            # Safe metadata extraction with error handling
                            if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
                                print(f"⚠️  Dokument ohne gültige Metadata: {doc}")
                                continue
                            
                            doc_id = doc.metadata.get('document_id', doc.metadata.get('pdf_id', 'unknown'))
                            if doc_id in target_doc_ids:
                                filtered_docs.append(doc)
                        except Exception as filter_error:
                            print(f"❌ Fehler beim Filtern von Dokument: {str(filter_error)}")
                            continue
                    
                    docs = filtered_docs
                    
                    if not docs:
                        return f"KEINE DOKUMENTE IN GEFILTERTEN IDs: Keine relevanten Dokumente in {target_doc_ids} gefunden."
                
                # Extract content with comprehensive error handling and adjacent chunks
                results = []
                found_doc_ids = set()
                chunk_counter = 1
                
                for i, doc in enumerate(docs):
                    try:
                        # Validate document structure
                        if not hasattr(doc, 'metadata') or not hasattr(doc, 'page_content'):
                            print(f"⚠️  Dokument {i} hat ungültige Struktur: {doc}")
                            continue
                        
                        # Safe metadata extraction
                        if not isinstance(doc.metadata, dict):
                            print(f"⚠️  Dokument {i} hat ungültige Metadata: {doc.metadata}")
                            doc_id = f"unknown_{i}"
                            chunk_id = None
                        else:
                            doc_id = doc.metadata.get('document_id', doc.metadata.get('pdf_id', f'unknown_{i}'))
                            chunk_id = doc.metadata.get('chunk_id')
                        
                        # Safe content extraction
                        content = getattr(doc, 'page_content', '')
                        if not isinstance(content, str):
                            print(f"⚠️  Dokument {i} hat ungültigen Content-Typ: {type(content)}")
                            content = str(content)
                        
                        if not content.strip():
                            print(f"⚠️  Dokument {i} hat leeren Content")
                            continue
                        
                        found_doc_ids.add(doc_id)
                        
                        # Page information no longer tracked
                        
                        # Try to get adjacent chunks if we have chunk metadata
                        if chunk_id is not None and isinstance(chunk_id, int):
                            try:
                                # Construct the full chunk ID
                                full_chunk_id = f"{doc_id}_chunk_{chunk_id}"
                                print(f"🔗 Suche Adjacent Chunks für: {full_chunk_id}")
                                
                                # Get adjacent chunks using vectorstore similarity search
                                adjacent_chunks = self._get_adjacent_chunks_content(namespace, doc_id, chunk_id)
                                
                                # Build the enhanced result with adjacent chunks
                                chunk_set = []
                                
                                # Previous chunk
                                if adjacent_chunks.get("previous"):
                                    chunk_set.append(f"--- CHUNK {chunk_counter}a (VORHERIGER) START ---")
                                    chunk_set.append(adjacent_chunks["previous"])
                                    chunk_set.append(f"--- CHUNK {chunk_counter}a (VORHERIGER) END ---")
                                
                                # Current chunk (main hit)
                                chunk_set.append(f"--- CHUNK {chunk_counter}b (HAUPTTREFFER) START ---")
                                chunk_set.append(content)
                                chunk_set.append(f"--- CHUNK {chunk_counter}b (HAUPTTREFFER) END ---")
                                
                                # Next chunk
                                if adjacent_chunks.get("next"):
                                    chunk_set.append(f"--- CHUNK {chunk_counter}c (NÄCHSTER) START ---")
                                    chunk_set.append(adjacent_chunks["next"])
                                    chunk_set.append(f"--- CHUNK {chunk_counter}c (NÄCHSTER) END ---")
                                
                                enhanced_content = "\n".join(chunk_set)
                                results.append(f"[DOC_ID: {doc_id}] {enhanced_content}")
                                
                                print(f"✅ Enhanced chunk {chunk_counter} mit Adjacent Chunks")
                                
                            except Exception as adjacent_error:
                                print(f"⚠️  Fehler bei Adjacent Chunks für {doc_id}_chunk_{chunk_id}: {adjacent_error}")
                                # Fallback to normal content
                                results.append(f"[DOC_ID: {doc_id}] {content}")
                        else:
                            # No chunk metadata, use normal content
                            results.append(f"[DOC_ID: {doc_id}] {content}")
                        
                        chunk_counter += 1
                        
                    except Exception as extract_error:
                        print(f"❌ Fehler beim Extrahieren von Dokument {i}: {str(extract_error)}")
                        continue
                
                if not results:
                    return "KEINE RELEVANTEN INHALTE: Dokumente gefunden, aber keine relevanten Informationen."
                
                # Build result with comprehensive logging
                doc_ids_list = list(found_doc_ids)
                result_text = "\n\n".join(results)
                result_text += f"\n\n[SYSTEM_INFO] FOUND_DOCUMENT_IDS: {doc_ids_list}"
                
                if target_doc_ids:
                    result_text += f"\n[SYSTEM_INFO] FILTERED_BY_DOC_IDS: {target_doc_ids}"
                
                print(f"✅ PDF Search erfolgreich: {len(results)} Ergebnisse, Dokument-IDs: {doc_ids_list}")
                print(f"📊 Context-Länge: {len(result_text)} Zeichen")
                
                return result_text
                
            except Exception as e:
                error_msg = f"FEHLER BEIM DURCHSUCHEN: {str(e)}"
                print(f"❌ PDF Search Fehler: {error_msg}")
                import traceback
                traceback.print_exc()
                return error_msg
        
        # Agent definieren
        researcher = Agent(
            role="Studienberater",
            goal="Hilf Studierenden bei ihren Fragen zu den verfügbaren Dokumenten. Sage direkt wenn du keine Informationen zu einem Thema hast.",
            backstory=f"""Du bist ein freundlicher Studienberater mit Zugang zu spezifischen Dokumenten. Du bist ehrlich über deine Grenzen.

            {documents_overview}

            WICHTIGE VERHALTENSREGELN:
            - Du kennst NUR die oben aufgelisteten Dokumente
            - Sage SOFORT wenn du zu einem Thema keine Informationen hast
            - Beispiel: "Zu diesem Thema habe ich leider keine Informationen in meinen verfügbaren Dokumenten."
            - Erfinde NIEMALS Informationen
            - Verwende deine Tools aktiv um in den Dokumenten zu suchen
            - Bei unklaren Fragen: Stelle eine kurze Rückfrage ("Welcher Studiengang?" oder "Welches Semester?")

            DEINE TOOLS:
            1. Document Overview Tool - zeigt aktuelle Dokumentenliste (nutze das wenn nach "verfügbaren Dokumenten" gefragt wird)
            2. PDF Search Tool - durchsucht die Dokumente nach spezifischen Informationen

            WICHTIG: Wenn eine Frage außerhalb deiner verfügbaren Dokumente liegt, sage das ehrlich und direkt!""",
            llm=self._llm,
            tools=[document_overview_tool, pdf_search_tool],
            verbose=False,  # Disable verbose to avoid parsing issues
            allow_delegation=False
        )
        
        self._agents[namespace] = researcher
        return researcher, vectorstore

    def _get_documents_overview_for_prompt(self, namespace: str) -> str:
        """
        Get a compact overview of available documents for the system prompt.
        
        Args:
            namespace: Namespace identifier
            
        Returns:
            String containing document overview for system prompt
        """
        try:
            summary_data = self.get_namespace_summary(namespace)
            
            if summary_data["status"] != "success" or summary_data["document_count"] == 0:
                return "KEINE DOKUMENTE VERFÜGBAR - Du hast aktuell keinen Zugang zu Dokumenten in diesem Namespace."
            
            overview_parts = [
                f"📚 VERFÜGBARE DOKUMENTE IN DIESEM NAMESPACE ({summary_data['document_count']} Dokumente):",
            ]
            
            for doc in summary_data["documents"]:
                doc_summary = doc['summary'][:150] + ("..." if len(doc['summary']) > 150 else "")
                overview_parts.append(f"• ID: {doc['id']} | Name: {doc.get('name', doc['id'])} | Thema: {doc_summary}")
            
            overview_parts.extend([
                "",
                "🎯 DEINE AUFGABEN:",
                "- Beantworte Fragen NUR basierend auf diesen verfügbaren Dokumenten",
                "- Sage DIREKT wenn du zu einem Thema keine Informationen hast",
                "- Nutze das Document Overview Tool für allgemeine Übersichten",
                "- Nutze das PDF Search Tool für spezifische Suchen",
                "- Sei ehrlich über deine Grenzen - erfinde keine Informationen"
            ])
            
            return "\n".join(overview_parts)
            
        except Exception as e:
            return f"FEHLER beim Laden der Dokumentübersicht: {str(e)}"

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
            # Input validation
            if not question or not isinstance(question, str) or not question.strip():
                return {
                    "answer": "Ungültige Frage",
                    "document_ids": [],
                    "sources": [],
                    "confidence_score": 0.0,
                    "context_used": False,
                    "additional_info": "Leere oder ungültige Frage"
                }
            
            if not namespace or not isinstance(namespace, str) or not namespace.strip():
                return {
                    "answer": "Ungültiger Namespace",
                    "document_ids": [],
                    "sources": [],
                    "confidence_score": 0.0,
                    "context_used": False,
                    "additional_info": "Leerer oder ungültiger Namespace"
                }
            
            print(f"🤖 AGENT PROCESSOR - Question: {question[:100]}...")
            print(f"🤖 AGENT PROCESSOR - Namespace: {namespace}")
            
            researcher, _ = self.setup_agent(namespace)
            
            # Prepare chat history context with validation
            chat_context = ""
            has_history = bool(chat_history and len(chat_history) > 0)
            
            if has_history:
                print(f"🤖 AGENT PROCESSOR - Processing {len(chat_history)} chat history messages")
                recent_messages = chat_history[-6:]  # Last 6 messages for context
                context_parts = []
                for i, msg in enumerate(recent_messages):
                    try:
                        if not isinstance(msg, dict):
                            print(f"⚠️  Chat message {i} ist kein Dict: {type(msg)}")
                            continue
                        
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        
                        if not isinstance(role, str) or not isinstance(content, str):
                            print(f"⚠️  Chat message {i} hat ungültigen Typ: role={type(role)}, content={type(content)}")
                            continue
                        
                        context_parts.append(f"{role.upper()}: {content}")
                    except Exception as msg_error:
                        print(f"❌ Fehler beim Verarbeiten von Chat-Message {i}: {str(msg_error)}")
                        continue
                
                chat_context = "\n".join(context_parts)
                print(f"🤖 AGENT PROCESSOR - Chat context length: {len(chat_context)} chars")
            
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

WENN ETWAS UNKLAR IST:
• Frage kurz nach: "Welcher Studiengang?" oder "Welches Semester?"
• Arbeite mit dem was du hast, auch wenn es nicht perfekt ist
• Nur bei wirklich nötigen Informationen nachfragen

WIE DU ANTWORTEN SOLLST:
• Sei natürlich und hilfsbereit - wie ein erfahrener Studienberater
• Für dokumentenbasierte Fragen: VERWENDE IMMER zuerst die entsprechenden Tools
• Bei unklaren Ergebnissen: Stelle SOFORT präzise Rückfragen
• NIEMALS über Kontext oder Suche sprechen
• Beziehe dich auf vorherige Nachrichten wenn relevant

TOOL-WORKFLOW (BEFOLGE DAS GENAU):
1. WENN die Frage nach Dokumenten fragt → Document Overview Tool aufrufen
2. WENN nach spezifischen Inhalten gefragt wird → PDF Search Tool aufrufen
3. WENN Ergebnisse unklar/unvollständig sind → Präzise Rückfragen stellen
4. ERST DANN antworte basierend auf den Tool-Ergebnissen

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
            
            # Try to parse the JSON response with comprehensive error handling
            try:
                print(f"🤖 AGENT PROCESSOR - Raw result length: {len(result_str)} chars")
                print(f"🤖 AGENT PROCESSOR - Raw result preview: {result_str[:200]}...")
                
                # Extract JSON from the response
                json_start = result_str.find('{')
                json_end = result_str.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_str = result_str[json_start:json_end]
                    print(f"🤖 AGENT PROCESSOR - Extracted JSON: {json_str[:300]}...")
                    
                    try:
                        parsed_response = json.loads(json_str)
                        print(f"🤖 AGENT PROCESSOR - JSON parsing successful")
                        
                        # Validate parsed response structure
                        if not isinstance(parsed_response, dict):
                            print(f"⚠️  Parsed response is not a dict: {type(parsed_response)}")
                            raise ValueError("Parsed response is not a dictionary")
                        
                        # Validate and structure the response with safe extraction
                        structured_response = {
                            "answer": str(parsed_response.get("answer", result_str)),
                            "document_ids": parsed_response.get("document_ids", []) if isinstance(parsed_response.get("document_ids"), list) else [],
                            "sources": parsed_response.get("sources", []) if isinstance(parsed_response.get("sources"), list) else [],
                            "confidence_score": float(parsed_response.get("confidence_score", 0.8)) if isinstance(parsed_response.get("confidence_score"), (int, float)) else 0.8,
                            "context_used": bool(parsed_response.get("context_used", has_history)),
                            "additional_info": parsed_response.get("additional_info")
                        }
                        
                        print(f"🤖 AGENT PROCESSOR - Structured response created successfully")
                        return structured_response
                        
                    except json.JSONDecodeError as json_error:
                        print(f"❌ JSON parsing error: {str(json_error)}")
                        print(f"❌ Problematic JSON: {json_str}")
                        raise json_error
                        
                else:
                    print(f"⚠️  No JSON found in response")
                    # Fallback if JSON parsing fails
                    return {
                        "answer": result_str,
                        "document_ids": [],
                        "sources": [],
                        "confidence_score": 0.7,
                        "context_used": has_history,
                        "additional_info": "Antwort konnte nicht als strukturiertes JSON geparst werden - keine JSON-Struktur gefunden"
                    }
                    
            except json.JSONDecodeError as json_error:
                print(f"❌ JSON parsing completely failed: {str(json_error)}")
                # Fallback if JSON parsing fails
                return {
                    "answer": result_str,
                    "document_ids": [],
                    "sources": [],
                    "confidence_score": 0.7,
                    "context_used": has_history,
                    "additional_info": f"JSON-Parsing fehlgeschlagen: {str(json_error)}"
                }
            except Exception as parse_error:
                print(f"❌ Unexpected parsing error: {str(parse_error)}")
                return {
                    "answer": result_str,
                    "document_ids": [],
                    "sources": [],
                    "confidence_score": 0.7,
                    "context_used": has_history,
                    "additional_info": f"Unerwarteter Parsing-Fehler: {str(parse_error)}"
                }
            
        except Exception as e:
            print(f"❌ AGENT PROCESSOR - Unerwarteter Fehler: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"Fehler beim Beantworten der Frage: {str(e)}",
                "document_ids": [],
                "sources": [],
                "confidence_score": 0.0,
                "context_used": False,
                "additional_info": f"Fehler aufgetreten: {type(e).__name__}"
            }

    def process_document_full(self, file_content: bytes, namespace: str, fileID: str, filename: str, hasTablesOrGraphics: str = "false", special_pages: list = None) -> Dict[str, Any]:
        """
        Complete document processing pipeline from PDF to indexed content.
        
        Args:
            file_content: Raw PDF file content
            namespace: Namespace for organization
            fileID: Unique document identifier
            filename: Original filename
            hasTablesOrGraphics: Whether PDF has tables/graphics requiring page-based chunking
            special_pages: List of page numbers (1-indexed) for special processing as images
            
        Returns:
            Dict containing processing results and status
        """
        try:
            # Step 1: Extract PDF content using DocProcessor
            pdf_data = self._doc_processor.extract_pdf(file_content, hasTablesOrGraphics, special_pages)
            if not pdf_data:
                return {
                    "status": "error",
                    "message": "Failed to extract PDF content"
                }
            
            # Step 2: Process content (chunk and summarize) using DocProcessor
            processed_pdf = self._doc_processor.process_pdf_content(pdf_data, filename, hasTablesOrGraphics, special_pages)
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
        """
        Delete a document from both vector database and Firebase.
        
        Args:
            namespace: Namespace containing the document
            fileID: Document ID to delete
            
        Returns:
            Dict containing deletion status
        """
        # Delete from vector database
        vector_result = self._vector_manager.delete_document(namespace, fileID)
        pinecone_deleted = vector_result.get("status") == "success"
        
        # Delete from Firebase
        firebase_deleted = False
        firebase_result = {"status": "success", "message": "Firebase not available"}
        if self._firebase_available:
            firebase_result = self._firebase.delete_document_metadata(namespace, fileID)
            if firebase_result.get("status") == "success":
                firebase_deleted = True
        
        return {
            "status": "success" if pinecone_deleted else "error",
            "message": f"Document {fileID} deletion - Pinecone: {pinecone_deleted}, Firebase: {firebase_deleted}",
            "vector_result": vector_result,
            "firebase_result": firebase_result
        }

    def delete_namespace(self, namespace: str) -> Dict[str, Any]:
        """
        Delete an entire namespace from the Pinecone index.
        
        Args:
            namespace: The name of the namespace to delete.
            
        Returns:
            A dictionary with the status of the operation.
        """
        return self._vector_manager.delete_namespace(namespace)

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
        
    