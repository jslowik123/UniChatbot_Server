#!/bin/bash

# Lokales Entwicklungssetup für Uni Chatbot API
echo "🚀 Starte lokales Entwicklungssetup..."

# Prüfe ob .env Datei existiert
if [ ! -f ".env" ]; then
    echo "❌ .env Datei nicht gefunden!"
    echo "Bitte erstelle eine .env Datei mit folgenden Variablen:"
    echo "PINECONE_API_KEY=your_pinecone_api_key"
    echo "OPENAI_API_KEY=your_openai_api_key"
    echo "FIREBASE_DATABASE_URL=your_firebase_url (optional)"
    echo "FIREBASE_CREDENTIALS_PATH=path/to/credentials.json (optional)"
    exit 1
fi

# Prüfe ob Redis installiert ist
if ! command -v redis-server &> /dev/null; then
    echo "❌ Redis nicht installiert!"
    echo "Installiere Redis:"
    echo "  macOS: brew install redis"
    echo "  Ubuntu: apt-get install redis-server"
    echo "  Oder verwende Docker: docker run --name redis -p 6379:6379 -d redis:7-alpine"
    exit 1
fi

# Function um Prozesse zu beenden
cleanup() {
    echo ""
    echo "🛑 Beende alle Prozesse..."
    kill $REDIS_PID $WORKER_PID $SERVER_PID 2>/dev/null
    wait $REDIS_PID $WORKER_PID $SERVER_PID 2>/dev/null
    echo "✅ Alle Prozesse beendet"
    exit 0
}

# Signal handler für sauberes Beenden
trap cleanup SIGINT SIGTERM

# Starte Redis Server
echo "📦 Starte Redis Server..."
redis-server --daemonize yes
REDIS_PID=$!

# Warte kurz bis Redis bereit ist
sleep 2

# Prüfe Redis Verbindung
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis Server nicht erreichbar!"
    exit 1
fi

echo "✅ Redis Server läuft"

# Starte Celery Worker
echo "👷 Starte Celery Worker..."
venv/bin/celery -A celery_app worker --loglevel=info &
WORKER_PID=$!

# Warte kurz bis Worker startet
sleep 3

# Starte FastAPI Server
echo "🌐 Starte FastAPI Server..."
venv/bin/python3 main.py &
SERVER_PID=$!

echo ""
echo "🎉 Alle Services gestartet!"
echo "📊 FastAPI Server: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "🔄 Redis: localhost:6379"
echo ""
echo "Drücke Ctrl+C zum Beenden..."

# Warte auf Beenden-Signal
wait 