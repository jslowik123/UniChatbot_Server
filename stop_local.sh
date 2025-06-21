#!/bin/bash

# Stoppt alle lokalen Entwicklungsservices für Uni Chatbot API
echo "🛑 Beende alle lokalen Services..."

# Beende FastAPI Server (Python main.py)
echo "🌐 Beende FastAPI Server..."
pkill -f "python main.py"

# Beende Celery Worker
echo "👷 Beende Celery Worker..."
pkill -f "celery -A celery_app worker"

# Beende Redis Server
echo "📦 Beende Redis Server..."
redis-cli shutdown 2>/dev/null || echo "Redis war bereits beendet"

# Warte kurz
sleep 2

# Prüfe ob alle Prozesse beendet wurden
echo ""
echo "🔍 Prüfe verbleibende Prozesse..."

# Prüfe FastAPI
if pgrep -f "python main.py" > /dev/null; then
    echo "⚠️  FastAPI Server läuft noch"
    echo "   Verwende: pkill -f 'python main.py'"
else
    echo "✅ FastAPI Server beendet"
fi

# Prüfe Celery
if pgrep -f "celery -A celery_app worker" > /dev/null; then
    echo "⚠️  Celery Worker läuft noch"
    echo "   Verwende: pkill -f 'celery -A celery_app worker'"
else
    echo "✅ Celery Worker beendet"
fi

# Prüfe Redis
if redis-cli ping > /dev/null 2>&1; then
    echo "⚠️  Redis Server läuft noch"
    echo "   Verwende: redis-cli shutdown"
else
    echo "✅ Redis Server beendet"
fi

echo ""
echo "🎉 Alle Services beendet!" 