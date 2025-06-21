# Test Scripts für server_uni

## 📋 Übersicht

Zwei umfassende Testskripte zum Validieren aller Komponenten der `server_uni` Anwendung:

1. **`test_all.py`** - Umfassende Systemtests (Dependencies, Verbindungen, Funktionalität)
2. **`test_api.py`** - API-Endpunkt Tests (FastAPI, HTTP-Requests)

## 🔧 Setup

### Voraussetzungen:
```bash
# 1. Dependencies installieren
./update_dependencies.sh

# 2. Environment Variables setzen (.env Datei)
PINECONE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
REDIS_URL=redis://localhost:6379  # optional

# 3. Zusätzlich für API-Tests: requests library
pip install requests
```

## 🧪 Test Scripts

### 1. Vollständige Systemtests (`test_all.py`)

**Was wird getestet:**
- ✅ **Import-Tests**: Alle Dependencies (langchain, crewai, pinecone, etc.)
- ✅ **Environment Variables**: API Keys und Konfiguration
- ✅ **Externe Verbindungen**: Pinecone, OpenAI, Firebase, Redis
- ✅ **Core Funktionalität**: AgentProcessor, PDF-Verarbeitung, Text-Splitting
- ✅ **Framework Setup**: FastAPI, Celery, CrewAI Agents

**Ausführung:**
```bash
# Alle Tests
python test_all.py

# Hilfe anzeigen
python test_all.py --help
```

**Beispiel-Ausgabe:**
```
🚀 STARTING COMPREHENSIVE TESTS
============================================================

🔍 TESTING IMPORTS
==================================================
[14:23:01] ✅ Import fastapi: PASS
    📝 FastAPI framework
[14:23:01] ✅ Import crewai: PASS
    📝 CrewAI framework
...

📊 FINAL TEST REPORT
============================================================
⏱️  Total Time: 45.2s
✅ Passed: 28
❌ Failed: 0
📈 Success Rate: 100.0%

🎉 ALL TESTS PASSED! Your application is ready to use.
```

### 2. API-Endpunkt Tests (`test_api.py`)

**Was wird getestet:**
- ✅ **Server Status**: Ist FastAPI Server erreichbar?
- ✅ **Bot Initialization**: `/start_bot` Endpunkt
- ✅ **Message Endpoints**: `/send_message`, `/send_message_structured`
- ✅ **Namespace Operations**: `/create_namespace`, `/namespace_info`
- ✅ **Worker Status**: Celery Tasks und `/test_worker`

**Voraussetzung:**
```bash
# Server muss laufen
uvicorn main:app --reload
```

**Ausführung:**
```bash
# Lokaler Test (localhost:8000)
python test_api.py

# Custom URL
python test_api.py http://localhost:5000

# Hilfe
python test_api.py --help
```

**Beispiel-Ausgabe:**
```
Testing API at: http://localhost:8000
🚀 STARTING API TESTS
==================================================

✅ Server Status: PASS
    📝 Server running - Welcome to the Uni Chatbot API - Agent Edition

🔍 TESTING API ENDPOINTS
==================================================
✅ Bot Start: PASS
    📝 Bot initialized successfully
✅ Send Message: PASS
    📝 Response received: Hallo! Ich bin Ihr universitärer Forschungsassistent...

📊 API TEST REPORT
==================================================
✅ Passed: 8
❌ Failed: 0
📈 Success Rate: 100.0%

🎉 ALL API TESTS PASSED!
```

## 📊 Test-Kategorien

### System Tests (`test_all.py`)

| Kategorie | Tests | Kritisch |
|-----------|--------|----------|
| **Dependencies** | langchain, crewai, pinecone, fastapi | ✅ |
| **Environment** | API Keys, Redis URL | ✅ |
| **Externe Services** | Pinecone, OpenAI, Firebase | ✅ |
| **Core Logic** | AgentProcessor, PDF Processing | ✅ |
| **Frameworks** | FastAPI, Celery, CrewAI | ⚠️ |

### API Tests (`test_api.py`)

| Endpunkt | Test | Erwartung |
|----------|------|-----------|
| `GET /` | Server Status | 200 + JSON response |
| `POST /start_bot` | Bot Init | Chat state initialisiert |
| `POST /send_message` | Message | Agent response |
| `POST /send_message_structured` | Structured | JSON mit confidence_score |
| `POST /create_namespace` | Namespace | Pinecone namespace erstellt |
| `GET /namespace_info/{ns}` | Info | Namespace-Details |
| `GET /test_worker` | Worker | Celery Task erstellt |

## 🔍 Troubleshooting

### Häufige Probleme:

**1. Import-Fehler:**
```bash
# Lösung: Dependencies neu installieren
./update_dependencies.sh
```

**2. API Key Fehler:**
```bash
# Prüfen: .env Datei vorhanden?
cat .env

# Setzen:
echo "PINECONE_API_KEY=your_key" >> .env
echo "OPENAI_API_KEY=your_key" >> .env
```

**3. Server nicht erreichbar:**
```bash
# Server starten:
uvicorn main:app --reload

# Port prüfen:
netstat -tulpn | grep :8000
```

**4. Redis-Verbindung:**
```bash
# Redis starten (macOS):
brew services start redis

# Redis starten (Docker):
docker run -d -p 6379:6379 redis:alpine

# Test:
redis-cli ping
```

**5. Pinecone-Verbindung:**
```bash
# Index prüfen:
python -c "from pinecone import Pinecone; pc = Pinecone(api_key='your_key'); print(list(pc.list_indexes()))"
```

## 🚀 Automatisierung

### CI/CD Integration:
```bash
# In GitHub Actions oder Jenkins:
python test_all.py
exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "All tests passed"
else
    echo "Tests failed"
    exit 1
fi
```

### Pre-Deployment Check:
```bash
#!/bin/bash
echo "Running pre-deployment tests..."
python test_all.py && python test_api.py
if [ $? -eq 0 ]; then
    echo "✅ Ready for deployment"
else
    echo "❌ Fix issues before deployment"
    exit 1
fi
```

## 📈 Erweiterte Tests

Zukünftige Erweiterungen:

- **Load Tests**: Stress-Tests für API-Endpunkte
- **PDF Upload Tests**: Echte PDF-Dateien testen
- **Streaming Tests**: `/send_message_stream` Endpunkt
- **Security Tests**: Input-Validation, Rate-Limiting
- **Performance Tests**: Response-Zeiten messen

## 🏁 Zusammenfassung

- **`test_all.py`**: Vollständige System-Validierung (vor erstem Start)
- **`test_api.py`**: API-Funktionalität (nach Server-Start)
- **Exit Codes**: 0 = Erfolg, 1 = Fehler (für Automatisierung)
- **Logs**: Detaillierte Fehlerausgaben für Debugging
- **Kategorisiert**: Critical vs. Optional Tests 