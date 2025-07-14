from tasks import generate_assessment

# Namespace, den du testen möchtest (z.B. "Informatik")
namespace = "Wirtschaftsinformatik"
# Starte die Generierung (asynchron, gibt ein AsyncResult-Objekt zurück)
result = generate_assessment.delay(namespace)

print(f"Assessment-Task gestartet! Task-ID: {result.id}")

# Optional: Warte auf das Ergebnis und gib es aus (nur für Testzwecke, blockiert bis fertig)
from celery.result import AsyncResult
import time

print("Warte auf das Ergebnis ...")
while not result.ready():
    print("... läuft noch ...")
    time.sleep(2)

print("Fertig!")
print("Ergebnis:")
print(result.get())