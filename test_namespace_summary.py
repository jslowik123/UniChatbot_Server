#!/usr/bin/env python3
"""
Kurzes Testskript für die get_namespace_summary Methode
"""

import os
import json
from dotenv import load_dotenv
from agent_processor import AgentProcessor

def test_namespace_summary():
    """Test der get_namespace_summary Methode"""
    
    # Load environment variables
    load_dotenv()
    
    # Get API keys from environment
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not pinecone_api_key or not openai_api_key:
        print("❌ FEHLER: PINECONE_API_KEY und OPENAI_API_KEY müssen in .env gesetzt sein")
        return
    
    try:
        # Initialize AgentProcessor
        print("🔧 Initialisiere AgentProcessor...")
        processor = AgentProcessor(
            pinecone_api_key=pinecone_api_key,
            openai_api_key=openai_api_key
        )
        
        # Test different namespaces
        test_namespaces = ["WiWi"]
        
        for namespace in test_namespaces:
            print(f"\n📊 Testing namespace: '{namespace}'")
            print("=" * 50)
            
            # Get namespace summary
            result = processor.get_namespace_summary(namespace)
            
            # Pretty print the result
            print(f"Status: {result['status']}")
            
            if result['status'] == 'success':
                print(f"Namespace: {result['namespace']}")
                print(f"Document Count: {result['document_count']}")
                
                if result['document_count'] > 0:
                    print("\n📚 Verfügbare Dokumente:")
                    for i, doc in enumerate(result['documents'], 1):
                        print(f"  {i}. ID: {doc['id']}")
                        print(f"     Name: {doc.get('name', 'Unknown')}")
                        print(f"     Status: {doc.get('status', 'Unknown')}")
                        print(f"     Chunks: {doc.get('chunk_count', 0)}")
                        print(f"     Date: {doc.get('date', 'Unknown')}")
                        print(f"     Summary: {doc['summary'][:100]}...")
                        if doc.get('additional_info'):
                            print(f"     Additional Info: {doc['additional_info']}")
                        print()
                else:
                    print("   ℹ️  Keine Dokumente in diesem Namespace gefunden")
                    
            else:
                print(f"❌ Fehler: {result.get('message', 'Unbekannter Fehler')}")
            
            print("-" * 50)
    
    except Exception as e:
        print(f"❌ Fehler beim Testen: {str(e)}")

def test_with_specific_namespace():
    """Test mit einem spezifischen Namespace (interaktiv)"""
    
    namespace = input("\n🔍 Gib einen Namespace zum Testen ein (oder Enter für 'test-namespace'): ").strip()
    if not namespace:
        namespace = "test-namespace"
    
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not pinecone_api_key or not openai_api_key:
        print("❌ FEHLER: API Keys nicht gefunden")
        return
    
    try:
        processor = AgentProcessor(pinecone_api_key, openai_api_key)
        result = processor.get_namespace_summary(namespace)
        
        print(f"\n📋 Ergebnis für Namespace '{namespace}':")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"❌ Fehler: {str(e)}")

if __name__ == "__main__":
    print("🧪 Testing get_namespace_summary Methode")
    print("=" * 60)
    
    # Automatische Tests
    test_namespace_summary()
    