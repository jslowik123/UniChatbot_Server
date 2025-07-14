from openai import OpenAI
import pymupdf  # PyMuPDF
from typing import Dict, Any, List, Tuple, Optional
from pinecone import Pinecone
import json
import os
import unicodedata
import re
import io
import base64
from firebase_connection import FirebaseConnection

# Constants
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_MODEL = "gpt-4.1-nano"
DEFAULT_TEMPERATURE = 0.3
EMBEDDING_MODEL = "text-embedding-3-small"


class DocProcessor:
    """
    Handles document processing, including PDF extraction, text cleaning, 
    chunking, and storage in vector database with metadata.
    """
    
    def __init__(self, pinecone_api_key: str, openai_api_key: str):
        """
        Initialize DocProcessor with API keys and connections.
        
        Args:
            pinecone_api_key: API key for Pinecone vector database
            openai_api_key: API key for OpenAI services
            
        Note:
            Firebase connection is configured via environment variables:
            - FIREBASE_DATABASE_URL: URL of Firebase Realtime Database
            - FIREBASE_CREDENTIALS_PATH: Path to credentials file (optional)
            - FIREBASE_CREDENTIALS_JSON: JSON string with credentials (optional, for Heroku)
            
        Raises:
            ValueError: If required API keys are missing
        """
        if not pinecone_api_key or not openai_api_key:
            raise ValueError("Both Pinecone and OpenAI API keys are required")
            
        self._openai = OpenAI(api_key=openai_api_key)
        self._pinecone = Pinecone(api_key=pinecone_api_key)
        
        try:
            self._firebase = FirebaseConnection()
            self._firebase_available = True
        except ValueError as e:
            self._firebase_available = False

    def _extract_page_text(self, page, page_num: int) -> Dict[str, Any]:
        """
        Extract text from a single PDF page.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
            
        Returns:
            Dict containing page number and text
        """
        page_text = page.get_text()
        
        page_data = {
            "page_number": page_num + 1,  # 1-indexed for user-friendly display
            "text": page_text
        }
        
        return page_data
    
    def _extract_page_as_image(self, page, page_num: int) -> Optional[Dict[str, Any]]:
        """
        Extract a page as a high-quality image and use OpenAI Vision API for precise text extraction.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
            
        Returns:
            Dict containing page image as base64, OpenAI-extracted text, and metadata
        """
        try:
            # Render page as image with high DPI for quality
            matrix = pymupdf.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=matrix)
            
            # Convert to PNG bytes
            img_bytes = pix.tobytes("png")
            
            # Convert to base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # Use OpenAI Vision API for precise text extraction
            try:
                vision_response = self._openai.chat.completions.create(
                    model="gpt-4o",  # Use gpt-4o for vision capabilities
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Bitte extrahiere den gesamten Text aus diesem Bild vollständig und genau. Achte auf alle Details, Tabellen, Formeln, Fußnoten und jeden sichtbaren Text. Gib nur den reinen Text zurück, ohne zusätzliche Kommentare oder Formatierungen. Einzelne Sachen kannst du auch richtig formattieren, sodass infroamtioenn korrekt extrahiert werden."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_base64}",
                                        "detail": "high"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=4000,
                    temperature=0.0  # Deterministic output for consistency
                )
                openai_extracted_text = vision_response.choices[0].message.content
                print(f"🤖 OpenAI Vision raw response for page {page_num + 1}: {repr(openai_extracted_text)[:300]}")
                print(f"🤖 OpenAI Vision extracted {len(openai_extracted_text)} chars from page {page_num + 1}")
                if not openai_extracted_text or not openai_extracted_text.strip():
                    print(f"⚠️ OpenAI Vision returned empty text for page {page_num + 1}, using PyMuPDF fallback.")
                    openai_extracted_text = page.get_text()
                    print(f"📄 PyMuPDF fallback extracted {len(openai_extracted_text)} chars from page {page_num + 1}")
                return {
                    "page_number": page_num + 1,
                    "enhanced_text": openai_extracted_text,
                    "image_base64": img_base64,
                    "meta": {"page_number": str(page_num + 1)}
                }
            except Exception as e:
                print(f"Fehler bei OpenAI Vision: {e}")
                return None
            
        except Exception as e:
            print(f"Error extracting page {page_num + 1} as image: {e}")
            return None

    def _split_text_by_pages(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split text into page-based chunks (one chunk per page) for documents with tables/graphics.
        
        Each page becomes a separate chunk.
        
        Args:
            pages_data: List of dictionaries containing page_number and text
            
        Returns:
            List of dictionaries containing text chunks with page information
        """
        if not pages_data:
            return []
            
        chunks = []
        
        for page_data in pages_data:
            page_num = page_data["page_number"]
            page_text = page_data["text"]
            
            if not page_text or not page_text.strip():
                continue
                
            # Clean the page text
            cleaned_text = self._clean_extracted_text(page_text)
            
            if not cleaned_text.strip():
                continue
            
            chunk_data = {
                "text": cleaned_text.strip(),
                "pages": [page_num],
                "page_range": str(page_num)
            }
            
            chunks.append(chunk_data)
            
        return chunks

    def extract_pdf(self, file_content: bytes, hasTablesOrGraphics: str = "false", special_pages: list = None) -> Optional[Dict[str, Any]]:
        """
        Extract text and metadata from PDF content using PyMuPDF.
        
        Args:
            file_content: Raw PDF file content as bytes
            hasTablesOrGraphics: Parameter for page-based chunking (for future use)
            special_pages: List of page numbers (1-indexed) for special processing as images
            
        Returns:
            Dict containing extracted text with page information and metadata, or None if extraction fails
        """
        try:
            # Create a BytesIO object from the file content
            pdf_file = io.BytesIO(file_content)
            
            # Open PDF with PyMuPDF
            doc = pymupdf.open(stream=pdf_file, filetype="pdf")
            
            # Extract text from all pages with page information
            pages_data = []
            full_text = ""
            special_pages_data = []
            metadata = {
                "page_count": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", "")
            }
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract page data (text only) - ALL pages get normal processing
                page_data = self._extract_page_text(page, page_num)
                pages_data.append(page_data)
                
                full_text += page_data["text"] + "\n"
            
            # Process special pages ADDITIONALLY (after normal processing)
            if special_pages:
                print(f"📸 Processing {len(special_pages)} special pages as additional high-quality vectors")
                for page_num_1indexed in special_pages:
                    page_num_0indexed = page_num_1indexed - 1  # Convert to 0-indexed
                    if 0 <= page_num_0indexed < len(doc):
                        page = doc.load_page(page_num_0indexed)
                        print(f"📸 Processing special page {page_num_1indexed} as additional image vector")
                        special_page_data = self._extract_page_as_image(page, page_num_0indexed)
                        if special_page_data:
                            special_pages_data.append(special_page_data)
            
            doc.close()
            
            # Clean up the extracted text
            full_text = self._clean_extracted_text(full_text)
            
            print(f"📄 PyMuPDF extracted {len(full_text)} characters from {metadata['page_count']} pages (all pages processed normally)")
            if special_pages_data:
                print(f"📸 Additionally extracted {len(special_pages_data)} special pages as high-quality image vectors")
            
            return {
                "text": full_text or "", 
                "pages_data": pages_data,
                "metadata": metadata,
                "special_pages_data": special_pages_data
            }
        except Exception as e:
            print(f"Fehler beim PDF-Verarbeiten mit PyMuPDF: {e}")
            return None

    def process_pdf_content(self, pdf_data: Dict[str, Any], filename: str, hasTablesOrGraphics: str = "false", special_pages: list = None) -> Optional[Dict[str, Any]]:
        """
        Process extracted PDF content by chunking and summarizing.
        
        Args:
            pdf_data: Dictionary containing text, pages_data and metadata from PDF
            filename: Original filename for identification
            hasTablesOrGraphics: Whether to use page-based chunking ("true") or sentence-based chunking ("false")
            special_pages: List of page numbers (1-indexed) for special processing as images
            
        Returns:
            Dict containing processed chunks with page info and summary, or None if processing fails
        """
        if not pdf_data or not pdf_data.get("text"):
            return None
        
        try:
            # Use page-based chunking when hasTablesOrGraphics is "true"
            if hasTablesOrGraphics.lower() == "true" and pdf_data.get("pages_data"):
                # Create one chunk per page when handling tables/graphics
                chunks_with_pages = self._split_text_by_pages(pdf_data["pages_data"])
                chunks = [chunk["text"] for chunk in chunks_with_pages]
            elif pdf_data.get("pages_data"):
                # Use normal sentence-based chunking with page tracking
                chunks_with_pages = self._split_text_with_page_tracking(pdf_data["pages_data"])
                chunks = [chunk["text"] for chunk in chunks_with_pages]
            else:
                # Fallback to regular chunking
                chunks = self._split_text(pdf_data["text"])
                chunks_with_pages = None
            
            # Generate summary
            summary_prompt = f"Fasse diesen Text zusammen (max. 1000 Zeichen): {pdf_data['text'][:10000]}"
            summary_response = self._openai.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=DEFAULT_TEMPERATURE
            )
            summary = summary_response.choices[0].message.content
            
            result = {
                "pdf_id": filename,
                "chunks": chunks,
                "summary": summary,
                "metadata": pdf_data["metadata"]
            }
            
            # Add page information if available
            if chunks_with_pages:
                result["chunks_with_pages"] = chunks_with_pages
            
            # Add special pages data if available (these are ADDITIONAL to normal chunks)
            if pdf_data.get("special_pages_data"):
                result["special_pages_data"] = pdf_data["special_pages_data"]
                print(f"📸 Added {len(pdf_data['special_pages_data'])} additional special page vectors to result")
            
            return result
        except Exception as e:
            print(f"Fehler beim Verarbeiten der PDF-Inhalte: {e}")
            return None

    def process_pdf(self, file_path: str, namespace: str, fileID: str) -> Dict[str, Any]:
        """
        Process a PDF file from disk and store its content in vector database.
        
        Args:
            file_path: Path to the PDF file on disk
            namespace: Pinecone namespace for organizing documents
            fileID: Unique identifier for the document
            
        Returns:
            Dict containing processing results with status, message, and metadata
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            Exception: For various processing errors
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        try:
            # Use PyMuPDF to extract text directly from file path
            doc = pymupdf.open(file_path)
            
            # Extract text from all pages with page information
            pages_data = []
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                # Store page data for later processing
                pages_data.append({
                    "page_number": page_num + 1,  # 1-indexed for user-friendly display
                    "text": page_text
                })
                
                full_text += page_text + "\n"
            
            doc.close()
            
            # Clean up the text
            full_text = self._clean_extracted_text(full_text)

            file_name = os.path.basename(file_path)
            return self._process_extracted_text_with_pages(full_text, pages_data, namespace, fileID, file_name)
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing PDF file: {str(e)}"
            }
    
    def process_pdf_bytes(self, pdf_file, namespace: str, fileID: str, file_name: str) -> Dict[str, Any]:
        """
        Process a PDF file from bytes (e.g., uploaded file) and store content in vector database.
        
        Args:
            pdf_file: PDF file as bytes or file-like object
            namespace: Pinecone namespace for organizing documents
            fileID: Unique identifier for the document
            file_name: Original filename of the document
            
        Returns:
            Dict containing processing results with status, message, and metadata
        """
        try:
            # Convert to BytesIO if it's raw bytes
            if isinstance(pdf_file, bytes):
                pdf_file = io.BytesIO(pdf_file)
            
            # Use PyMuPDF to extract text from file-like object
            doc = pymupdf.open(stream=pdf_file, filetype="pdf")
            
            # Extract text from all pages with page information
            pages_data = []
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                # Store page data for later processing
                pages_data.append({
                    "page_number": page_num + 1,  # 1-indexed for user-friendly display
                    "text": page_text
                })
                
                full_text += page_text + "\n"
            
            doc.close()
            
            # Clean up the text
            full_text = self._clean_extracted_text(full_text)
            
            return self._process_extracted_text_with_pages(full_text, pages_data, namespace, fileID, file_name)
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing PDF bytes: {str(e)}"
            }
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean and normalize extracted text from PDF.
        
        Args:
            text: Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Replace multiple whitespaces with single space but preserve paragraphs
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs -> single space
        text = re.sub(r'\n[ \t]*\n', '\n\n', text)  # Preserve paragraph breaks
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines -> single newline
        text = text.strip()
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\w\s\n.,!?;:()\-\[\]{}"\'/]', '', text, flags=re.UNICODE)
        
        # Additional cleanup for common HTML-like artifacts
        replacements = {
            "<br>": "",
            "<p>": "",
            "</p>": "",
            "|": "",
            "•": "",
            "_": "",
            "..": "",
            ". .": "",
            "...": "",
            "\r\n": "",
            "*": "",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text
    
    def _process_extracted_text(self, text: str, namespace: str, fileID: str, file_name: str) -> Dict[str, Any]:
        """
        Process extracted text by cleaning, chunking, and extracting metadata.
        
        Note: This method only processes the text and extracts metadata.
        Vector storage is handled by AgentProcessor.
        
        Args:
            text: Raw extracted text
            namespace: Pinecone namespace
            fileID: Document identifier
            file_name: Original filename
            
        Returns:
            Dict containing processing results including chunks and metadata
        """
        try:
            keywords, summary = self._extract_keywords_and_summary(text)
            chunks = self._split_text(text)
            
            # Store metadata in Firebase if available
            firebase_result = self._store_metadata(namespace, fileID, len(chunks), keywords, summary)
            
            return {
                "status": "success",
                "message": f"File {file_name} processed successfully",
                "chunks": len(chunks),
                "text_chunks": chunks,
                "summary": summary,
                "keywords": keywords,
                "firebase_result": firebase_result,
                "original_file": file_name
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error in text processing pipeline: {str(e)}"
            }
    
    def _process_extracted_text_with_pages(self, text: str, pages_data: List[Dict[str, Any]], namespace: str, fileID: str, file_name: str) -> Dict[str, Any]:
        """
        Process extracted text by cleaning, chunking, and extracting metadata.
        
        Note: This method only processes the text and extracts metadata.
        Vector storage is handled by AgentProcessor.
        
        Args:
            text: Raw extracted text
            pages_data: List of dictionaries containing page number and text
            namespace: Pinecone namespace
            fileID: Document identifier
            file_name: Original filename
            
        Returns:
            Dict containing processing results including chunks and metadata
        """
        try:
            keywords, summary = self._extract_keywords_and_summary(text)
            
            # Use page-aware chunking for better text processing
            chunks_with_pages = self._split_text_with_page_tracking(pages_data)
            
            # Extract just the text for backward compatibility
            text_chunks = [chunk["text"] for chunk in chunks_with_pages]
            
            # Store metadata in Firebase if available
            firebase_result = self._store_metadata(namespace, fileID, len(text_chunks), keywords, summary)
            
            return {
                "status": "success",
                "message": f"File {file_name} processed successfully",
                "chunks": len(text_chunks),
                "text_chunks": text_chunks,
                "chunks_with_pages": chunks_with_pages,  # Text chunks with additional structure
                "summary": summary,
                "keywords": keywords,
                "firebase_result": firebase_result,
                "original_file": file_name,
                "pages_data": pages_data
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error in text processing pipeline: {str(e)}"
            }
    
    def _store_metadata(self, namespace: str, fileID: str, chunk_count: int, 
                       keywords: List[str], summary: str) -> Dict[str, Any]:
        """
        Store document metadata in Firebase if available.
        
        Args:
            namespace: Document namespace
            fileID: Document identifier  
            chunk_count: Number of text chunks created
            keywords: Extracted keywords
            summary: Document summary
            
        Returns:
            Dict with storage result status
        """
        if self._firebase_available:
            return self._firebase.append_metadata(
                namespace=namespace,
                fileID=fileID,
                chunk_count=chunk_count,
                keywords=keywords,
                summary=summary
            )
        else:
            return {
                'status': 'error',
                'message': 'Firebase nicht verfügbar'
            }
    
    def _extract_keywords_and_summary(self, text: str) -> Tuple[List[str], str]:
        """
        Extract keywords and summary using OpenAI.
        
        This method uses AI to:
        1. Extract relevant keywords and topics (max 3 words each)
        2. Generate a concise summary (3-5 sentences)
        
        Args:
            text: Raw text extracted from PDF
            
        Returns:
            Tuple containing (keywords_list, summary)
            
        Raises:
            Exception: If OpenAI API call fails or returns invalid JSON
        """
        if not text or not text.strip():
            return [], ""
            
        prompt = {
            "role": "system", 
            "content": """Du bist ein Assistent zur Textbereinigung und Inhaltsanalyse. Du erhältst einen Text aus einem PDF und sollst zwei Aufgaben erfüllen:
            1) Die wichtigsten Schlagwörter und Themen aus dem Text extrahieren. Die Keywords sollen nicht mehr als 3 Wörter lang sein.
            2) Eine kurze Zusammenfassung des Dokuments in 3-5 Sätzen erstellen.

            Gib das Ergebnis als JSON mit den Feldern 'keywords' und 'summary' zurück."""
        }
        
        user_message = {
            "role": "user",
            "content": f"Hier ist der Text aus einem PDF. Bitte extrahiere die Schlagwörter/Themen und erstelle eine kurze Zusammenfassung:\n\n{text}"
        }
        
        try:
            response = self._openai.chat.completions.create(
                model=DEFAULT_MODEL,
                response_format={"type": "json_object"},
                messages=[prompt, user_message],
                temperature=DEFAULT_TEMPERATURE,
            )
            
            result = json.loads(response.choices[0].message.content)
            return (
                result.get("keywords", []), 
                result.get("summary", "")
            )
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error in keywords extraction: {str(e)}")
            # Fallback: return empty metadata
            return [], ""

    def _split_text(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
        """
        Split text into manageable chunks for vector embedding.
        
        Splits text by sentences to maintain semantic coherence within chunks.
        Each chunk aims for approximately chunk_size characters.
        
        Args:
            text: Text to split into chunks
            chunk_size: Target size for each chunk in characters
            
        Returns:
            List of text chunks, each roughly chunk_size characters
        """
        if not text or not text.strip():
            return []
            
        chunks = []
        current_chunk = ""
        
        # Split by sentences to maintain semantic boundaries
        sentences = text.replace('\n', ' ').split('. ')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def _split_text_with_page_tracking(self, pages_data: List[Dict[str, Any]], chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[Dict[str, Any]]:
        """
        Split text from pages into manageable chunks with page tracking.
        
        Splits text by sentences to maintain semantic coherence within chunks.
        Each chunk aims for approximately chunk_size characters and tracks
        which pages contributed to the chunk.
        
        Args:
            pages_data: List of dictionaries containing page_number and text
            chunk_size: Target size for each chunk in characters
            
        Returns:
            List of dictionaries containing text chunks with page information
        """
        if not pages_data:
            return []
            
        chunks = []
        current_chunk = ""
        current_pages = set()
        
        for page_data in pages_data:
            page_num = page_data["page_number"]
            page_text = page_data["text"]
            
            if not page_text or not page_text.strip():
                continue
                
            # Clean the page text
            page_text = self._clean_extracted_text(page_text)
            
            # Split by sentences to maintain semantic boundaries
            sentences = page_text.replace('\n', ' ').split('. ')
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # Check if adding this sentence would exceed chunk size
                if len(current_chunk) + len(sentence) < chunk_size:
                    current_chunk += sentence + ". "
                    current_pages.add(page_num)
                else:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "pages": sorted(list(current_pages)),
                            "page_range": f"{min(current_pages)}-{max(current_pages)}" if len(current_pages) > 1 else str(min(current_pages))
                        })
                    current_chunk = sentence + ". "
                    current_pages = {page_num}
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "pages": sorted(list(current_pages)),
                "page_range": f"{min(current_pages)}-{max(current_pages)}" if len(current_pages) > 1 else str(min(current_pages))
            })
            
        return chunks
    
    def get_namespace_data(self, namespace: str) -> List[Dict[str, Any]]:
        """
        Retrieve document metadata for all documents in a namespace.
        
        Args:
            namespace: The namespace to query
            
        Returns:
            List of dictionaries containing document metadata (id, name, keywords, summary, chunk_count)
        """
        if not self._firebase_available:
            print("Firebase not available for namespace data retrieval")
            return []
            
        try:
            namespace_data = self._firebase.get_namespace_data(namespace)
            
            extracted_data = []
            if namespace_data.get('status') == 'success' and 'data' in namespace_data:
                for doc_id, doc_data in namespace_data['data'].items():
                    # Skip non-document entries
                    if not isinstance(doc_data, dict) or 'keywords' not in doc_data:
                        continue
                    
                    doc_info = {
                        'id': doc_id,
                        'name': doc_data.get('name', 'Unknown'),
                        'keywords': doc_data.get('keywords', []),
                        'summary': doc_data.get('summary', ''),
                        'additional_info': doc_data.get('additional_info', '')
                    }
                    extracted_data.append(doc_info)
                    
            return extracted_data
        except Exception as e:
            print(f"Error retrieving namespace data: {str(e)}")
            return []

    def appropriate_document_search(self, namespace: str, extracted_data: List[Dict[str, Any]], user_query: str, history: list) -> Dict[str, Any]:
        """
        Find the most appropriate document to answer a user's query.
        
        Uses AI to analyze document metadata (keywords, summaries) and match
        them against the user's question to find the most relevant document.
        
        Args:
            namespace: The namespace being searched
            extracted_data: List of document metadata dictionaries
            user_query: User's question or search query
            
        Returns:
            Dict containing the ID of the most appropriate document
        """
        if not extracted_data:
            return {"id": "default"}
            
        if len(extracted_data) == 1:
            return {"id": extracted_data[0]["id"]}

        try:
            prompt = {
                "role": "system", 
                "content": """Du bist ein Assistent, der verschiedene Informationen über Dokumente bekommt. Du sollst entscheiden welches Dokument am besten passt um eine Frage des Nutzers zu beantworten. 
                Antworte im JSON-Format mit genau diesem Schema: {"id": "document_id"}. 
                Verwende keine anderen Felder und füge keine Erklärungen hinzu.
                Analysiere die Keywords und Zusammenfassungen der Dokumente und wähle das relevanteste für die Nutzeranfrage aus. 
                Beachte dabei die vom Nutzer zu den jewiligen Dokumenten hinzugefügten Infos.
                Bachte die beigefügte Chat History des Nutzers, wenn deiner Meinung nach keine weiteren Informationen benötigt werden aus den Dokumente, sondenr einfahc nur weiterführende Fragen gestellt wurden,
                dann antworte mit {"id": "no_document_found"}.
                Wenn du kein passendes Dokument findest, antworte mit {"id": "no_document_found"}."""
            }
                
            user_message = {
                "role": "user",
                "content": f"Hier sind die verfügbaren Dokumente:\n\n{json.dumps(extracted_data, indent=2, ensure_ascii=False)}\n\nDie Frage des Users lautet: {user_query}\n\nDie Chat History des Users lautet: {history}\n\nWelches Dokument ist am besten geeignet? \n\n"
            }
                
            response = self._openai.chat.completions.create(
                model=DEFAULT_MODEL,
                response_format={"type": "json_object"},
                messages=[prompt, user_message],
                temperature=0.1,  # Low temperature for consistent selection
            )
                
            response_content = response.choices[0].message.content
            return json.loads(response_content)
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error in document selection: {str(e)}")
            print(f"Response content: {response_content if 'response_content' in locals() else 'No response'}")
            # Fallback: return first document
            return {"id": extracted_data[0]["id"]}

    def generate_global_summary(self, namespace: str) -> Dict[str, Any]:
        """
        Generate a global summary for all documents in a namespace.
        
        Args:
            namespace: The namespace to summarize
            
        Returns:
            Dict containing the operation status and generated summary
        """
        try:
            namespace_data = self.get_namespace_data(namespace)
            if not namespace_data:
                return {
                    "status": "error",
                    "message": f"No documents found in namespace {namespace}"
                }
            
            # Extract all summaries and keywords
            all_summaries = []
            all_keywords = []
            
            for doc in namespace_data:
                if doc.get('summary'):
                    all_summaries.append(f"Document {doc['name']}: {doc['summary']}")
                if doc.get('keywords'):
                    all_keywords.extend(doc['keywords'])
            
            if not all_summaries:
                return {
                    "status": "error", 
                    "message": "No document summaries available for global summary generation"
                }
            
            # Generate global summary using AI
            prompt = {
                "role": "system",
                "content": """Du erstellst eine globale Zusammenfassung für eine Sammlung von Dokumenten. 
                Erstelle eine kohärente Übersicht über alle Dokumente und extrahiere die wichtigsten Themenbereiche 
                als Bullet Points. Antworte im JSON-Format mit den Feldern 'global_summary' und 'main_topics'."""
            }
            
            user_message = {
                "role": "user", 
                "content": f"Erstelle eine globale Zusammenfassung für diese Dokumente:\n\n{chr(10).join(all_summaries)}\n\nWichtige Keywords: {', '.join(set(all_keywords))}"
            }
            
            response = self._openai.chat.completions.create(
                model=DEFAULT_MODEL,
                response_format={"type": "json_object"},
                messages=[prompt, user_message],
                temperature=DEFAULT_TEMPERATURE,
            )
            
            result = json.loads(response.choices[0].message.content)
            global_summary = result.get("global_summary", "")
            main_topics = result.get("main_topics", [])
            
            # Store in Firebase if available
            if self._firebase_available:
                self._firebase.update_namespace_summary(namespace, main_topics)
            
            return {
                "status": "success",
                "global_summary": global_summary,
                "main_topics": main_topics,
                "message": f"Global summary generated for namespace {namespace}"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error generating global summary: {str(e)}"
            }
        
    