# Uni Chatbot API - Agent Edition

Ein moderner FastAPI-basierter Chatbot für Universitätsdokumente, der CrewAI-Agenten, pdfplumber und agentic RAG verwendet.

## 🚀 Features

- **CrewAI Agents**: Intelligente Agenten für sophistizierte Dokumentenanalyse
- **Agentic RAG**: Erweiterte Retrieval-Augmented Generation mit Multi-Query und Kontextkompression
- **PDFPlumber**: Verbesserte PDF-Extraktion für bessere Textqualität
- **Streaming Responses**: Echtzeit-Antworten für bessere Benutzererfahrung
- **Structured Outputs**: Strukturierte JSON-Antworten mit Metadaten, Quellen und Vertrauenswerten
- **Namespace-basierte Organisation**: Separate Dokumentensammlungen pro Namespace
- **Asynchrone Verarbeitung**: Celery-basierte Hintergrundverarbeitung
- **Firebase Integration**: Optional für Metadaten-Speicherung

## 🏗️ Architektur

### Komponenten

1. **AgentProcessor** (`agent_processor.py`): Kernkomponente für Dokumentenverarbeitung und Agent-Setup
2. **AgentChatbot** (`agent_chatbot.py`): CrewAI-basierter Chatbot mit Streaming-Unterstützung
3. **FastAPI Main** (`main.py`): API-Endpunkte und Server-Konfiguration
4. **Celery Tasks** (`tasks.py`): Asynchrone Dokumentenverarbeitung
5. **Firebase Connection** (`firebase_connection.py`): Optional für Metadaten

### Technologien

- **FastAPI**: Web-Framework
- **CrewAI**: Agent-Framework
- **LangChain**: RAG-Pipeline
- **Pinecone**: Vektordatenbank
- **PDFPlumber**: PDF-Verarbeitung
- **Celery**: Asynchrone Tasks
- **Redis**: Message Broker
- **Firebase**: Metadaten-Speicherung (optional)

## 📋 Anforderungen

- Python 3.8+
- Redis Server
- Pinecone API Key
- OpenAI API Key
- Firebase Credentials (optional)

## 🛠️ Installation

1. **Dependencies installieren:**
```bash
pip install -r requirements.txt
```

2. **Environment Variables setzen:**
```bash
# .env Datei erstellen
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
REDIS_URL=redis://localhost:6379
# Optional für Firebase
FIREBASE_DATABASE_URL=your_firebase_url
FIREBASE_CREDENTIALS_PATH=path/to/credentials.json
```

3. **Redis Server starten:**
```bash
redis-server
```

4. **Celery Worker starten:**
```bash
celery -A celery_app worker --loglevel=info
```

5. **FastAPI Server starten:**
```bash
python main.py
```

## 🔗 API Endpunkte

### Grundlegende Endpunkte

- **GET `/`**: API-Informationen und Gesundheitsstatus
- **POST `/start_bot`**: Agent-Chatbot initialisieren

### Dokumentenmanagement

- **POST `/upload`**: PDF-Dokument hochladen und verarbeiten
  - `file`: PDF-Datei
  - `namespace`: Namespace für Organisation
  - `fileID`: Eindeutige Dokument-ID
  - `additionalInfo`: Zusätzliche Informationen

- **POST `/delete`**: Dokument löschen
  - `file_name`: Dateiname
  - `namespace`: Namespace
  - `fileID`: Dokument-ID
  - `just_firebase`: "true" für Pinecone-only Löschung

### Namespace-Management

- **POST `/create_namespace`**: Neuen Namespace erstellen
  - `namespace`: Name des Namespace
  - `dimension`: Vektordimension (default: 1536)

- **POST `/delete_namespace`**: Namespace löschen
  - `namespace`: Name des Namespace

- **GET `/namespace_info/{namespace}`**: Namespace-Informationen abrufen

### Chat-Funktionen

- **POST `/send_message`**: Einfache Chat-Nachricht (nur Antwort-Text)
  - `user_input`: Benutzerfrage
  - `namespace`: Namespace für Dokumentensuche

- **POST `/send_message_structured`**: Strukturierte Chat-Nachricht mit Metadaten
  - `user_input`: Benutzerfrage
  - `namespace`: Namespace für Dokumentensuche

- **POST `/send_message_stream`**: Streaming-Chat mit Agent
  - `user_input`: Benutzerfrage
  - `namespace`: Namespace für Dokumentensuche

### Task-Management

- **GET `/task_status/{task_id}`**: Status einer asynchronen Aufgabe
- **GET `/test_worker`**: Celery Worker testen

## 🔄 Strukturierte Ausgaben

Der Agent gibt jetzt strukturierte JSON-Antworten zurück:

```json
{
  "answer": "Die ausführliche Antwort auf die Frage",
  "document_ids": ["doc1", "doc2", "doc3"],
  "sources": [
    "Originaltext aus Dokument 1, der die Antwort stützt",
    "Relevanter Text aus Dokument 2"
  ],
  "confidence_score": 0.9,
  "context_used": true,
  "additional_info": "Zusätzliche Hinweise oder Informationen"
}
```

### Strukturierte Felder erklärt:

- **`answer`**: Die hauptsächliche Antwort auf die Frage
- **`document_ids`**: Liste der Dokument-IDs, die für die Antwort verwendet wurden
- **`sources`**: Originaltexte aus den Dokumenten, die die Antwort stützen
- **`confidence_score`**: Vertrauenswert der Antwort (0.0 - 1.0)
- **`context_used`**: Ob Chat-History-Kontext verwendet wurde
- **`additional_info`**: Zusätzliche Informationen oder Hinweise

## 🤖 Agent-Workflow

1. **Dokumenten-Upload**: PDF wird mit pdfplumber extrahiert
2. **Text-Segmentierung**: RecursiveCharacterTextSplitter erstellt Chunks
3. **Embedding-Generierung**: OpenAI text-embedding-ada-002
4. **Vektorspeicherung**: Pinecone mit Namespace-Organisation
5. **Agent-Setup**: CrewAI Agent mit Multi-Query Retriever
6. **Kontextkompression**: LLMChainExtractor für relevante Inhalte
7. **Strukturierte Antwortgenerierung**: GPT-4 mit agentic RAG und JSON-Output

## 📝 Beispielverwendung

### Python Client (Strukturierte Antworten)

```python
import requests
import json

# Bot starten
response = requests.post("http://localhost:8000/start_bot")
print(response.json())

# Strukturierte Chat-Nachricht
data = {
    "user_input": "Was ist das Modul Software Engineering?",
    "namespace": "university"
}
response = requests.post("http://localhost:8000/send_message_structured", data=data)
result = response.json()

# Strukturierte Antwort verarbeiten
structured = result["structured_response"]
print("Antwort:", structured["answer"])
print("Vertrauen:", structured["confidence_score"])
print("Quellen:", structured["sources"])
print("Dokumente:", structured["document_ids"])
```

### Python Client (Einfache Antworten)

```python
# Einfache Chat-Nachricht (nur Text)
data = {
    "user_input": "Was ist das Modul Software Engineering?",
    "namespace": "university"
}
response = requests.post("http://localhost:8000/send_message", data=data)
result = response.json()
print(result["response"])  # Nur die Antwort
```

### JavaScript Client (Strukturiert)

```javascript
// Strukturierte Antwort
const formData = new FormData();
formData.append('user_input', 'Welche Module kann man wählen?');
formData.append('namespace', 'university');

fetch('/send_message_structured', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    const structured = data.structured_response;
    console.log('Antwort:', structured.answer);
    console.log('Vertrauen:', structured.confidence_score);
    console.log('Quellen:', structured.sources);
    console.log('Chat-Kontext verwendet:', structured.context_used);
});
```

### JavaScript Client (Streaming mit Metadaten)

```javascript
// Streaming-Chat mit erweiterten Metadaten
const formData = new FormData();
formData.append('user_input', 'Erkläre mir die Prüfungsordnung');
formData.append('namespace', 'university');

const eventSource = new EventSource('/send_message_stream');
eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'chunk') {
        // Chunks können jetzt auch Emojis und Metadaten enthalten
        console.log(data.content);
    } else if (data.type === 'complete') {
        console.log('Vollständige Antwort:', data.fullResponse);
        eventSource.close();
    }
};
```

## 🔧 Konfiguration

### Agent-Einstellungen

```python
# In agent_processor.py
EMBEDDING_MODEL = "text-embedding-ada-002"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
GPT_MODEL = "gpt-4"
```

### Strukturierte Ausgabe-Konfiguration

Der Agent wird explizit angewiesen, strukturierte JSON-Ausgaben zu produzieren:

```python
# Beispiel der strukturierten Task-Beschreibung
task_description = """
🔄 CHAT HISTORY BEACHTUNG - WICHTIG:
Du siehst eine Chat History mit vorherigen Nachrichten. BERÜCKSICHTIGE diese aktiv:
- Beziehe dich auf vorherige Fragen und Antworten
- Nutze den Kontext aus früheren Nachrichten
- Wenn der Nutzer "dazu", "darüber", "das" oder ähnliche Bezugswörter verwendet, beziehe dich auf vorherige Themen

WICHTIG: Deine Antwort MUSS in folgendem JSON-Format sein:
{
    "answer": "Deine ausführliche Antwort hier ohne Anführungszeichen",
    "document_ids": ["doc1", "doc2"],
    "sources": ["Originaltext aus den Dokumenten"],
    "confidence_score": 0.9,
    "context_used": true,
    "additional_info": "Zusätzliche Hinweise oder null"
}
"""
```

## 📊 Vertrauenswerte und Qualität

### Confidence Score Interpretation:
- **🟢 0.8-1.0**: Hohe Vertrauenswürdigkeit - Antwort basiert auf klaren, relevanten Quellen
- **🟡 0.6-0.8**: Mittlere Vertrauenswürdigkeit - Antwort ist wahrscheinlich korrekt, aber weniger eindeutig
- **🔴 0.0-0.6**: Niedrige Vertrauenswürdigkeit - Antwort könnte ungenau sein oder auf unzureichenden Informationen basieren

### Qualitätsindikatoren:
- **Anzahl Quellen**: Mehr Quellen = höhere Verlässlichkeit
- **Chat-Kontext verwendet**: Zeigt an, ob die Antwort im Kontext der Unterhaltung steht
- **Dokument-IDs**: Identifiziert die genauen Dokumente, die für die Antwort verwendet wurden

## 🔄 Migration von alter Version

Die neue Agent-Edition ersetzt folgende Komponenten:

- `doc_processor.py` → `agent_processor.py`
- `chatbot.py` → `agent_chatbot.py`
- `pinecone_connection.py` → Direkte Integration in `agent_processor.py`
- PyPDF2 → PDFPlumber
- Einfache RAG → Agentic RAG mit CrewAI
- **Neu**: Strukturierte JSON-Ausgaben mit Metadaten

## 🐛 Debugging

### Logs aktivieren

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Strukturierte Ausgaben debuggen

```python
# Die strukturierte Antwort analysieren
response = message_bot_structured("Test Frage", "namespace")
print(f"Confidence: {response['confidence_score']}")
print(f"Sources: {len(response['sources'])}")
print(f"Context used: {response['context_used']}")
```

### Celery Worker Debugging

```bash
celery -A celery_app worker --loglevel=debug
```

### Agent-Ausgaben aktivieren

```python
# In agent_processor.py
researcher = Agent(
    # ...
    verbose=True  # Bereits aktiviert
)
```

## 📈 Performance

- **Parallele Verarbeitung**: Celery für asynchrone Dokumentenverarbeitung
- **Effiziente Vektorsuche**: Pinecone mit Namespace-Isolation
- **Streaming**: Echtzeitantworten für bessere UX mit Metadaten
- **Kontextkompression**: Reduzierte Token-Nutzung
- **Chat-History**: Pro-Namespace Speicherung mit Context-Awareness
- **Strukturierte Ausgaben**: Optimiert für maschinelle Weiterverarbeitung

## 🔒 Sicherheit

- **API-Key-Validierung**: Alle externen APIs erfordern gültige Keys
- **Namespace-Isolation**: Dokumente sind per Namespace getrennt
- **Input-Validierung**: Robuste Eingabevalidierung
- **Error-Handling**: Umfassendes Fehlerbehandlung mit strukturierten Fehlermeldungen
- **Vertrauenswerte**: Transparente Qualitätsbewertung der Antworten

## 📚 Weitere Dokumentation

- [CrewAI Documentation](https://docs.crewai.com/)
- [LangChain Documentation](https://docs.langchain.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/) 