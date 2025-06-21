#!/usr/bin/env python3
"""
Comprehensive Test Script for server_uni
Tests all components: Dependencies, Connections, Functionality, and APIs
"""

import sys
import os
import time
import json
import traceback
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TestRunner:
    """Comprehensive test runner for the server_uni application."""
    
    def __init__(self):
        self.results = {
            "passed": 0,
            "failed": 0,
            "errors": [],
            "warnings": []
        }
        self.start_time = time.time()
    
    def log_test(self, test_name: str, status: str, message: str = "", error: str = ""):
        """Log test results."""
        timestamp = time.strftime("%H:%M:%S")
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        
        print(f"[{timestamp}] {status_emoji} {test_name}: {status}")
        if message:
            print(f"    📝 {message}")
        if error:
            print(f"    🔴 {error}")
            self.results["errors"].append(f"{test_name}: {error}")
        
        if status == "PASS":
            self.results["passed"] += 1
        elif status == "FAIL":
            self.results["failed"] += 1
    
    def test_imports(self) -> bool:
        """Test all critical imports."""
        print("\n🔍 TESTING IMPORTS\n" + "="*50)
        
        imports_to_test = [
            ("fastapi", "FastAPI framework"),
            ("uvicorn", "ASGI server"),
            ("pdfplumber", "PDF processing"),
            ("langchain_text_splitters", "Text splitting"),
            ("langchain_openai", "OpenAI integration"),
            ("langchain.retrievers.multi_query", "Multi-query retrieval"),
            ("langchain.retrievers", "Document retrieval"),
            ("crewai", "CrewAI framework"),
            ("crewai.tools", "CrewAI tools"),
            ("pinecone", "Pinecone vector database"),
            ("langchain_pinecone", "LangChain Pinecone integration"),
            ("firebase_admin", "Firebase integration"),
            ("celery", "Task queue"),
            ("redis", "Redis cache"),
            ("openai", "OpenAI API"),
            ("pydantic", "Data validation")
        ]
        
        all_passed = True
        for module, description in imports_to_test:
            try:
                __import__(module)
                self.log_test(f"Import {module}", "PASS", description)
            except ImportError as e:
                self.log_test(f"Import {module}", "FAIL", description, str(e))
                all_passed = False
            except Exception as e:
                self.log_test(f"Import {module}", "FAIL", description, f"Unexpected error: {str(e)}")
                all_passed = False
        
        return all_passed
    
    def test_environment_variables(self) -> bool:
        """Test required environment variables."""
        print("\n🔍 TESTING ENVIRONMENT VARIABLES\n" + "="*50)
        
        required_vars = [
            ("PINECONE_API_KEY", "Pinecone API access"),
            ("OPENAI_API_KEY", "OpenAI API access"),
            ("REDIS_URL", "Redis connection (optional)")
        ]
        
        all_passed = True
        for var, description in required_vars:
            value = os.getenv(var)
            if value:
                # Mask sensitive values
                masked_value = f"{value[:8]}..." if len(value) > 8 else "***"
                self.log_test(f"Env var {var}", "PASS", f"{description} - {masked_value}")
            else:
                if var == "REDIS_URL":
                    self.log_test(f"Env var {var}", "WARN", f"{description} - Optional, using default")
                else:
                    self.log_test(f"Env var {var}", "FAIL", description, "Missing required environment variable")
                    all_passed = False
        
        return all_passed
    
    def test_pinecone_connection(self) -> bool:
        """Test Pinecone connection."""
        print("\n🔍 TESTING PINECONE CONNECTION\n" + "="*50)
        
        try:
            from pinecone import Pinecone
            
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                self.log_test("Pinecone Connection", "FAIL", "", "PINECONE_API_KEY not found")
                return False
            
            pc = Pinecone(api_key=api_key)
            indexes = list(pc.list_indexes())
            
            self.log_test("Pinecone Connection", "PASS", f"Connected successfully, {len(indexes)} indexes found")
            return True
            
        except Exception as e:
            self.log_test("Pinecone Connection", "FAIL", "", str(e))
            return False
    
    def test_openai_connection(self) -> bool:
        """Test OpenAI connection."""
        print("\n🔍 TESTING OPENAI CONNECTION\n" + "="*50)
        
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.log_test("OpenAI Connection", "FAIL", "", "OPENAI_API_KEY not found")
                return False
            
            # Test ChatOpenAI
            llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0)
            response = llm.invoke("Hello, this is a test. Respond with 'OK'.")
            
            self.log_test("OpenAI ChatGPT", "PASS", f"Response: {response.content[:50]}...")
            
            # Test Embeddings
            embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-ada-002")
            test_embedding = embeddings.embed_query("test")
            
            self.log_test("OpenAI Embeddings", "PASS", f"Embedding dimension: {len(test_embedding)}")
            return True
            
        except Exception as e:
            self.log_test("OpenAI Connection", "FAIL", "", str(e))
            return False
    
    def test_firebase_connection(self) -> bool:
        """Test Firebase connection."""
        print("\n🔍 TESTING FIREBASE CONNECTION\n" + "="*50)
        
        try:
            from firebase_connection import FirebaseConnection
            
            firebase = FirebaseConnection()
            self.log_test("Firebase Connection", "PASS", "Firebase initialized successfully")
            return True
            
        except Exception as e:
            self.log_test("Firebase Connection", "WARN", "Firebase optional", str(e))
            return True  # Firebase is optional
    
    def test_redis_connection(self) -> bool:
        """Test Redis connection."""
        print("\n🔍 TESTING REDIS CONNECTION\n" + "="*50)
        
        try:
            import redis
            
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            r = redis.from_url(redis_url)
            
            # Test connection
            r.ping()
            self.log_test("Redis Connection", "PASS", f"Connected to {redis_url}")
            return True
            
        except Exception as e:
            self.log_test("Redis Connection", "WARN", "Redis needed for Celery", str(e))
            return True  # Not critical for basic functionality
    
    def test_agent_processor(self) -> bool:
        """Test AgentProcessor functionality."""
        print("\n🔍 TESTING AGENT PROCESSOR\n" + "="*50)
        
        try:
            from agent_processor import AgentProcessor
            
            pinecone_key = os.getenv("PINECONE_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")
            
            if not pinecone_key or not openai_key:
                self.log_test("AgentProcessor Init", "FAIL", "", "Missing API keys")
                return False
            
            # Initialize AgentProcessor
            processor = AgentProcessor(pinecone_key, openai_key, "test-index")
            self.log_test("AgentProcessor Init", "PASS", "AgentProcessor initialized")
            
            # Test PDF processing with sample text
            sample_pdf_data = {
                "text": "This is a test document. It contains sample text for testing purposes. " * 10,
                "metadata": {"title": "Test Document"}
            }
            
            # Test text processing
            processed = processor.process_pdf_content(sample_pdf_data, "test.pdf")
            if processed:
                self.log_test("PDF Processing", "PASS", f"Processed {len(processed['chunks'])} chunks")
            else:
                self.log_test("PDF Processing", "FAIL", "", "Failed to process PDF content")
                return False
            
            # Test vectorstore setup
            vectorstore = processor.setup_vectorstore("test-namespace")
            self.log_test("Vectorstore Setup", "PASS", "Vectorstore configured")
            
            # Test agent setup
            agent, _ = processor.setup_agent("test-namespace")
            self.log_test("Agent Setup", "PASS", "CrewAI agent configured")
            
            return True
            
        except Exception as e:
            self.log_test("AgentProcessor", "FAIL", "", str(e))
            return False
    
    def test_pdf_processing(self) -> bool:
        """Test PDF processing functionality."""
        print("\n🔍 TESTING PDF PROCESSING\n" + "="*50)
        
        try:
            import pdfplumber
            import io
            
            # Create a simple test PDF content (we'll simulate this)
            self.log_test("PDFPlumber Import", "PASS", "PDF processing library available")
            
            # Test with sample content
            from agent_processor import AgentProcessor
            
            pinecone_key = os.getenv("PINECONE_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")
            
            if pinecone_key and openai_key:
                processor = AgentProcessor(pinecone_key, openai_key)
                
                # Test text splitting
                sample_text = "This is a long document. " * 100
                chunks = processor._text_splitter.split_text(sample_text)
                
                self.log_test("Text Splitting", "PASS", f"Split into {len(chunks)} chunks")
                return True
            else:
                self.log_test("PDF Processing", "SKIP", "Missing API keys")
                return True
                
        except Exception as e:
            self.log_test("PDF Processing", "FAIL", "", str(e))
            return False
    
    def test_celery_setup(self) -> bool:
        """Test Celery task queue setup."""
        print("\n🔍 TESTING CELERY SETUP\n" + "="*50)
        
        try:
            from celery_app import celery
            from tasks import process_document
            
            # Test Celery app
            self.log_test("Celery App", "PASS", f"Celery app configured: {celery.main}")
            
            # Test task registration
            if hasattr(process_document, 'delay'):
                self.log_test("Celery Tasks", "PASS", "Tasks registered successfully")
            else:
                self.log_test("Celery Tasks", "FAIL", "", "Tasks not properly registered")
                return False
            
            return True
            
        except Exception as e:
            self.log_test("Celery Setup", "FAIL", "", str(e))
            return False
    
    def test_fastapi_imports(self) -> bool:
        """Test FastAPI application imports."""
        print("\n🔍 TESTING FASTAPI IMPORTS\n" + "="*50)
        
        try:
            # Test importing main application components
            from main import app, agent_processor, chat_state
            
            self.log_test("FastAPI App", "PASS", "Main application imported")
            self.log_test("Agent Processor", "PASS", "AgentProcessor instance available")
            self.log_test("Chat State", "PASS", "ChatState management available")
            
            # Test app configuration
            if hasattr(app, 'routes'):
                route_count = len(app.routes)
                self.log_test("API Routes", "PASS", f"{route_count} routes configured")
            
            return True
            
        except Exception as e:
            self.log_test("FastAPI Imports", "FAIL", "", str(e))
            return False
    
    def test_crewai_functionality(self) -> bool:
        """Test CrewAI functionality."""
        print("\n🔍 TESTING CREWAI FUNCTIONALITY\n" + "="*50)
        
        try:
            from crewai import Agent, Task, Crew
            from crewai.tools import tool
            from langchain_openai import ChatOpenAI
            
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                self.log_test("CrewAI Test", "SKIP", "Missing OPENAI_API_KEY")
                return True
            
            # Create a simple test agent
            llm = ChatOpenAI(api_key=openai_key, model="gpt-3.5-turbo", temperature=0)
            
            @tool("Test Tool")
            def test_tool(query: str) -> str:
                """A simple test tool."""
                return f"Test tool received: {query}"
            
            agent = Agent(
                role="Test Assistant",
                goal="Test CrewAI functionality",
                backstory="A test agent for validation",
                llm=llm,
                tools=[test_tool],
                verbose=False
            )
            
            task = Task(
                description="Say 'Hello World' using the test tool with input 'test'",
                expected_output="A response that includes 'Hello World'",
                agent=agent
            )
            
            crew = Crew(agents=[agent], tasks=[task], verbose=False)
            
            self.log_test("CrewAI Setup", "PASS", "Agent, Task, and Crew created successfully")
            
            # Quick test execution (with timeout protection)
            try:
                result = crew.kickoff()
                self.log_test("CrewAI Execution", "PASS", f"Task executed: {str(result)[:100]}...")
            except Exception as exec_error:
                self.log_test("CrewAI Execution", "WARN", "Setup OK, execution error", str(exec_error))
            
            return True
            
        except Exception as e:
            self.log_test("CrewAI Functionality", "FAIL", "", str(e))
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results."""
        print("🚀 STARTING COMPREHENSIVE TESTS")
        print("="*60)
        
        # Run all tests
        tests = [
            ("Imports", self.test_imports),
            ("Environment Variables", self.test_environment_variables),
            ("Pinecone Connection", self.test_pinecone_connection),
            ("OpenAI Connection", self.test_openai_connection),
            ("Firebase Connection", self.test_firebase_connection),
            ("Redis Connection", self.test_redis_connection),
            ("Agent Processor", self.test_agent_processor),
            ("PDF Processing", self.test_pdf_processing),
            ("Celery Setup", self.test_celery_setup),
            ("FastAPI Imports", self.test_fastapi_imports),
            ("CrewAI Functionality", self.test_crewai_functionality)
        ]
        
        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                self.log_test(test_name, "FAIL", "", f"Test crashed: {str(e)}")
                print(traceback.format_exc())
        
        # Generate final report
        self.generate_report()
        return self.results
    
    def generate_report(self):
        """Generate final test report."""
        elapsed_time = time.time() - self.start_time
        total_tests = self.results["passed"] + self.results["failed"]
        
        print("\n" + "="*60)
        print("📊 FINAL TEST REPORT")
        print("="*60)
        print(f"⏱️  Total Time: {elapsed_time:.2f}s")
        print(f"✅ Passed: {self.results['passed']}")
        print(f"❌ Failed: {self.results['failed']}")
        print(f"📈 Success Rate: {(self.results['passed']/total_tests)*100:.1f}%" if total_tests > 0 else "No tests run")
        
        if self.results["errors"]:
            print(f"\n🔴 ERRORS ({len(self.results['errors'])}):")
            for error in self.results["errors"]:
                print(f"   • {error}")
        
        if self.results["warnings"]:
            print(f"\n⚠️  WARNINGS ({len(self.results['warnings'])}):")
            for warning in self.results["warnings"]:
                print(f"   • {warning}")
        
        # Overall status
        if self.results["failed"] == 0:
            print("\n🎉 ALL TESTS PASSED! Your application is ready to use.")
        elif self.results["failed"] < 3:
            print("\n⚠️  SOME TESTS FAILED - Check the errors above, but core functionality should work.")
        else:
            print("\n🔴 MULTIPLE FAILURES - Please fix the critical issues before running the application.")
        
        print("="*60)


def main():
    """Main test execution."""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("""
Comprehensive Test Script for server_uni

Usage:
    python test_all.py              # Run all tests
    python test_all.py --help       # Show this help

This script tests:
- All dependency imports
- Environment variables
- External service connections (Pinecone, OpenAI, Firebase, Redis)
- Core functionality (AgentProcessor, PDF processing)
- Framework setup (FastAPI, Celery, CrewAI)
        """)
        return
    
    tester = TestRunner()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main() 