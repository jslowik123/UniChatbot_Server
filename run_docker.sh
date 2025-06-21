#!/bin/bash

# Docker-basiertes lokales Setup für Uni Chatbot API
echo "🐳 Starte Docker-basiertes Setup..."

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

# Prüfe ob Docker installiert ist
if ! command -v docker &> /dev/null; then
    echo "❌ Docker nicht installiert!"
    echo "Installiere Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose nicht installiert!"
    echo "Installiere Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Function um Container zu stoppen
cleanup() {
    echo ""
    echo "🛑 Stoppe Docker Container..."
    docker-compose down
    echo "✅ Container gestoppt"
    exit 0
}

# Signal handler für sauberes Beenden
trap cleanup SIGINT SIGTERM

# Baue und starte Container
echo "🔨 Baue Docker Images..."
docker-compose build

echo "🚀 Starte Services..."
docker-compose up

echo ""
echo "🎉 Alle Services gestartet!"
echo "📊 FastAPI Server: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "🔄 Redis: localhost:6379"
echo ""
echo "Drücke Ctrl+C zum Beenden..."

# Warte auf Beenden-Signal
wait 