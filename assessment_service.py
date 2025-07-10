import os
import json
from typing import Dict, Any
from openai import OpenAI
from agent_processor import AgentProcessor
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class AssessmentService:
    """
    Service for generating chatbot assessments based on uploaded documents and goals.
    
    This class analyzes uploaded documents against chatbot objectives and provides
    structured feedback on completeness, missing documents, and recommendations.
    """
    
    def __init__(self, agent_processor: AgentProcessor):
        """
        Initialize AssessmentService with required dependencies.
        
        Args:
            agent_processor: AgentProcessor instance for document access
        """
        self._agent_processor = agent_processor
        self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def generate_assessment(self, namespace: str, chatbot_goal: str) -> Dict[str, Any]:
        """
        Generate a comprehensive assessment of the chatbot's knowledge base.
        
        Args:
            namespace: Namespace containing the documents to assess
            chatbot_goal: Description of what the chatbot should achieve
            
        Returns:
            Dict containing assessment results and Firebase storage status
        """
        try:
            # Get documents from the namespace
            documents = self._agent_processor.get_documents(namespace)
            logger.info(f"Generating assessment for namespace: {namespace}")
            logger.info(f"Chatbot goal: {chatbot_goal}")
            logger.info(f"Documents found: {len(documents) if documents else 0}")
            
            # Generate assessment using OpenAI
            assessment_content = self._create_assessment_with_openai(documents, chatbot_goal)
            
            # Save assessment to Firebase
            firebase_status = self._save_assessment_to_firebase(namespace, assessment_content)
            
            return {
                "status": "success",
                "assessment": assessment_content,
                "firebase_status": firebase_status
            }
            
        except Exception as e:
            logger.error(f"Error generating assessment: {str(e)}")
            return {
                "status": "error",
                "message": f"Error generating assessment: {str(e)}"
            }
    
    def _create_assessment_with_openai(self, documents: str, chatbot_goal: str) -> str:
        """
        Use OpenAI to create the assessment based on documents and goals.
        
        Args:
            documents: String representation of available documents
            chatbot_goal: The intended purpose/goal of the chatbot
            
        Returns:
            JSON string containing the assessment
        """
        system_prompt = """
        Du bist ein Experte für Bildungstechnologie und Wissensanalyse. Ein Benutzer möchte einen Chatbot für Studierende erstellen. Dazu hat er Dokumente hochgeladen, die das Fachwissen des Chatbots bilden sollen. Außerdem hat er eine Zielbeschreibung beigefügt, was der Chatbot ungefähr können oder leisten soll.

        Deine Aufgabe ist eine strukturierte Analyse mit KURZEN STICHWORTEN zu erstellen:

        WICHTIG: Antworte im JSON-Format mit exakt diesen Feldern:
        {
            "vorhandene_dokumente": ["Stichwort 1", "Stichwort 2", "Stichwort 3"],
            "fehlende_dokumente": ["Fehlendes Dokument 1", "Fehlendes Dokument 2"],
            "tipps": ["Kurzer Tipp 1", "Kurzer Tipp 2"],
            "wissensstand": "Kurze Beschreibung des Wissensstands des Chatbots, also was alles beantwortet werden kann (5-10 Sätze)",
            "confidence": 85
        }

        Regeln für die Stichworte:
        - "vorhanden": Kurze Namen der vorhandenen Dokumenttypen (z.B. "Modulhandbuch Bachelor", "Studienordnung")
        - "fehlt": Konkrete fehlende Dokumenttypen (z.B. "Prüfungsordnung", "Stundenplan", "Praktikumsordnung")
        - "tipps": Maximal 3-4 Wörter pro Tipp (z.B. "Aktuelle Prüfungstermine hinzufügen", "FAQ für Erstsemester")
        - "confidence": Zahl zwischen 0-100 für wie vollständig deiner Meinung nach der Chatbot ist, im Hinblick auf das Ziel.

        WICHTIGE REGEL FÜR "FEHLT":
        Analysiere die Zielbeschreibung des Chatbots genau. Wenn bestimmte Dokumenttypen für das spezifische Ziel NICHT RELEVANT sind, dann markiere sie NICHT als fehlend.
        
        Beispiele:
        - Wenn der Chatbot nur für "allgemeine Studienberatung" gedacht ist → keine spezifischen Fachmodule als fehlend markieren
        - Wenn der Chatbot nur für "Erstsemester-Info" gedacht ist → keine Master-spezifischen Dokumente als fehlend markieren
        - Wenn der Chatbot nur für "einen spezifischen Studiengang" gedacht ist → keine anderen Studiengänge als fehlend markieren
        - Wenn der Chatbot für "organisatorische Fragen" gedacht ist → keine fachlichen Inhalte als fehlend markieren

        Fokussiere dich nur auf Dokumente, die für das SPEZIFISCHE ZIEL des Chatbots wirklich notwendig sind.

        Denke an typische Universitätsdokumente:
        - Modulhandbücher, Prüfungsordnungen, Studienordnungen
        - Stundenpläne, Vorlesungsverzeichnis
        - Wenn nur ein Dokument hochgeladen wurde, und noch 4 fehlen, dann sind so circa 20% erreicht.
        - Also versuch ein deterministisches Vorgehen zu finden.

        Antworte NUR mit dem JSON-Objekt, keine weiteren Erklärungen.
        """

        user_prompt = f"""
        Ziel des Chatbots:
        {chatbot_goal}

        Vorliegende Dokumente:
        {documents}
        """

        try:
            response = self._openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_prompt.strip()},
                ],
                temperature=0.3,
                stream=False,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            # Return a fallback assessment
            fallback_assessment = {
                "vorhandene_dokumente": ["Unbekannte Dokumente"],
                "fehlende_dokumente": ["Analyse fehlgeschlagen"],
                "tipps": ["OpenAI API Fehler beheben"],
                "wissensstand": "Die Analyse konnte aufgrund eines technischen Fehlers nicht durchgeführt werden.",
                "confidence": 0
            }
            return json.dumps(fallback_assessment)
    
    def _save_assessment_to_firebase(self, namespace: str, assessment_content: str) -> str:
        """
        Save the assessment to Firebase under the namespace.
        
        Args:
            namespace: Namespace identifier
            assessment_content: JSON string of the assessment
            
        Returns:
            Status string indicating success or error
        """
        try:
            if not self._agent_processor._firebase_available:
                return "Firebase not available"
            
            self._agent_processor._firebase._db.reference(
                f'files/{namespace}/assessment'
            ).set(assessment_content)
            
            logger.info(f"Assessment saved to Firebase for namespace: {namespace}")
            return "success"
            
        except Exception as e:
            logger.error(f"Error saving assessment to Firebase: {str(e)}")
            return f"error: {str(e)}"
    
    def get_assessment(self, namespace: str) -> Dict[str, Any]:
        """
        Retrieve an existing assessment from Firebase.
        
        Args:
            namespace: Namespace identifier
            
        Returns:
            Dict containing the assessment or error information
        """
        try:
            if not self._agent_processor._firebase_available:
                return {
                    "status": "error",
                    "message": "Firebase not available"
                }
            
            assessment_ref = self._agent_processor._firebase._db.reference(
                f'files/{namespace}/assessment'
            )
            assessment_data = assessment_ref.get()
            
            if assessment_data:
                return {
                    "status": "success",
                    "assessment": assessment_data
                }
            else:
                return {
                    "status": "error",
                    "message": "No assessment found for this namespace"
                }
                
        except Exception as e:
            logger.error(f"Error retrieving assessment: {str(e)}")
            return {
                "status": "error",
                "message": f"Error retrieving assessment: {str(e)}"
            } 