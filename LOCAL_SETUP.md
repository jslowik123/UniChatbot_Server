# 🏠 Lokales Setup - Uni Chatbot API

Du hast jetzt alles was du brauchst, um deinen FastAPI Server mit Celery-Workern lokal zu starten! Es gibt zwei Optionen:

## 🚀 Option 1: Direkte Installation (Empfohlen)

### Voraussetzungen
- Python 3.8+
- Redis Server
- Pinecone API Key
- OpenAI API Key

### Setup-Schritte

1. **Environment-Variablen einrichten:**
```bash
# Kopiere die Beispiel-Datei
cp .env.example .env

# Bearbeite .env mit deinen API Keys
nano .env
```

2. **Redis installieren (falls nicht vorhanden):**
```bash
# macOS
brew install redis

# Ubuntu/Debian
sudo apt-get install redis-server

# Oder mit Docker
docker run --name redis -p 6379:6379 -d redis:7-alpine
```

3. **Dependencies installieren:**
```bash
# Virtual Environment (empfohlen)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder: venv\Scripts\activate  # Windows

# Packages installieren
pip install -r requirements.txt
```

4. **Alles auf einmal starten:**
```bash
./run_local.sh
```

Das Script startet automatisch:
- ✅ Redis Server
- ✅ Celery Worker
- ✅ FastAPI Server mit Hot-Reload

### Manueller Start (falls gewünscht)
```bash
# Terminal 1: Redis
redis-server

# Terminal 2: Celery Worker
celery -A celery_app worker --loglevel=info

# Terminal 3: FastAPI Server
python main.py
```

## 🐳 Option 2: Docker Compose

### Voraussetzungen
- Docker
- Docker Compose

### Setup-Schritte

1. **Environment-Variablen einrichten:**
```bash
cp .env.example .env
# Bearbeite .env mit deinen API Keys
```

2. **Mit Docker starten:**
```bash
./run_docker.sh
```

Oder manuell:
```bash
docker-compose up --build
```

## 📊 Zugriff auf die Services

Nach dem Start sind folgende Services verfügbar:

- **FastAPI Server**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Interactive API**: http://localhost:8000/redoc
- **Redis**: localhost:6379

## 🔧 Environment-Variablen

Fülle deine `.env` Datei mit folgenden Werten:

```env
# Erforderlich
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...

# Standard für lokale Entwicklung
REDIS_URL=redis://localhost:6379
ENVIRONMENT=development

# Optional (Firebase)
FIREBASE_DATABASE_URL=https://...
FIREBASE_CREDENTIALS_PATH=path/to/credentials.json
```

## 🧪 Testen des Setups

1. **Health Check:**
```bash
curl http://localhost:8000/
```

2. **Worker Test:**
```bash
curl http://localhost:8000/test_worker
```

3. **Bot starten:**
```bash
curl -X POST http://localhost:8000/start_bot
```

## 🛠️ Entwicklung

### Hot Reload
Der FastAPI Server startet automatisch mit Hot-Reload wenn `ENVIRONMENT=development` gesetzt ist.

### Logs anschauen
```bash
# Celery Worker Logs
celery -A celery_app worker --loglevel=debug

# Redis Logs
redis-cli monitor
```

### Debugging
- FastAPI läuft auf Port 8000 mit Debug-Modus
- Celery Worker läuft mit INFO Log-Level
- Redis läuft auf Standard-Port 6379

## 🔄 Was ist anders zu Heroku?

### Vorher (Heroku):
- Procfile definiert web und worker Prozesse
- Redis über Heroku Add-on
- Umgebungsvariablen über Heroku Dashboard
- SSL für Redis automatisch

### Jetzt (Lokal):
- Direkte Prozess-Verwaltung über Scripts
- Lokaler Redis Server oder Docker
- .env Datei für Variablen
- Kein SSL nötig für lokalen Redis

## ❓ Troubleshooting

### Redis Verbindungsfehler
```bash
# Prüfe ob Redis läuft
redis-cli ping
# Sollte "PONG" zurückgeben
```

### Celery Worker startet nicht
```bash
# Prüfe ob alle Dependencies installiert sind
pip install -r requirements.txt

# Starte Worker im Debug-Modus
celery -A celery_app worker --loglevel=debug
```

### API Keys funktionieren nicht
```bash
# Prüfe .env Datei
cat .env

# Teste direkt in Python
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('OPENAI_API_KEY'))"
```

## 🎉 Du bist fertig!

Dein FastAPI Server läuft jetzt lokal mit allen Features:
- ✅ Asynchrone PDF-Verarbeitung
- ✅ CrewAI Agents  
- ✅ Streaming Chat
- ✅ Hot-Reload für Entwicklung
- ✅ Separate Worker-Prozesse

Viel Spaß beim Entwickeln! 🚀 