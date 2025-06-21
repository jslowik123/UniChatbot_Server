#!/usr/bin/env python3
"""
API Test Script for server_uni
Quick tests for FastAPI endpoints and API functionality
"""

import requests
import json
import time
import sys
from typing import Dict, Any

class APITester:
    """Simple API tester for FastAPI endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = {"passed": 0, "failed": 0, "errors": []}
    
    def log_test(self, test_name: str, status: str, message: str = "", error: str = ""):
        """Log test results."""
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {status}")
        
        if message:
            print(f"    📝 {message}")
        if error:
            print(f"    🔴 {error}")
            self.results["errors"].append(f"{test_name}: {error}")
        
        if status == "PASS":
            self.results["passed"] += 1
        elif status == "FAIL":
            self.results["failed"] += 1
    
    def test_server_running(self) -> bool:
        """Test if the server is running."""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.log_test("Server Status", "PASS", f"Server running - {data.get('message', 'OK')}")
                return True
            else:
                self.log_test("Server Status", "FAIL", "", f"HTTP {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            self.log_test("Server Status", "FAIL", "", "Server not running - start with 'uvicorn main:app --reload'")
            return False
        except Exception as e:
            self.log_test("Server Status", "FAIL", "", str(e))
            return False
    
    def test_start_bot(self) -> bool:
        """Test bot initialization endpoint."""
        try:
            response = self.session.post(f"{self.base_url}/start_bot", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.log_test("Bot Start", "PASS", data.get("message", "Bot started"))
                return True
            else:
                self.log_test("Bot Start", "FAIL", "", f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.log_test("Bot Start", "FAIL", "", str(e))
            return False
    
    def test_send_message(self) -> bool:
        """Test message sending endpoint."""
        try:
            data = {
                "user_input": "Hello, this is a test message",
                "namespace": "test-namespace"
            }
            response = self.session.post(f"{self.base_url}/send_message", data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                self.log_test("Send Message", "PASS", f"Response received: {result.get('response', '')[:100]}...")
                return True
            else:
                self.log_test("Send Message", "FAIL", "", f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.log_test("Send Message", "FAIL", "", str(e))
            return False
    
    def test_structured_message(self) -> bool:
        """Test structured message endpoint."""
        try:
            data = {
                "user_input": "test test test",
                "namespace": "test-namespace"
            }
            response = self.session.post(f"{self.base_url}/send_message_structured", data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if "answer" in result and "confidence_score" in result:
                    self.log_test("Structured Message", "PASS", f"Structured response with confidence: {result.get('confidence_score', 0)}")
                    return True
                else:
                    self.log_test("Structured Message", "FAIL", "", "Response missing required fields")
                    return False
            else:
                self.log_test("Structured Message", "FAIL", "", f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.log_test("Structured Message", "FAIL", "", str(e))
            return False
    
    def test_namespace_operations(self) -> bool:
        """Test namespace creation and info endpoints."""
        test_namespace = f"test-{int(time.time())}"
        
        try:
            # Test namespace creation
            data = {"namespace": test_namespace, "dimension": 1536}
            response = self.session.post(f"{self.base_url}/create_namespace", data=data, timeout=15)
            
            if response.status_code == 200:
                self.log_test("Create Namespace", "PASS", f"Namespace {test_namespace} created")
            else:
                self.log_test("Create Namespace", "FAIL", "", f"HTTP {response.status_code}")
                return False
            
            # Test namespace info
            response = self.session.get(f"{self.base_url}/namespace_info/{test_namespace}", timeout=10)
            
            if response.status_code == 200:
                info = response.json()
                self.log_test("Namespace Info", "PASS", f"Info retrieved: {info.get('message', 'OK')}")
                return True
            else:
                self.log_test("Namespace Info", "FAIL", "", f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Namespace Operations", "FAIL", "", str(e))
            return False
    
    def test_worker_status(self) -> bool:
        """Test worker/celery status."""
        try:
            response = self.session.get(f"{self.base_url}/test_worker", timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                task_id = result.get("task_id")
                if task_id:
                    self.log_test("Worker Test", "PASS", f"Task created: {task_id}")
                    
                    # Check task status
                    time.sleep(2)  # Wait a bit
                    status_response = self.session.get(f"{self.base_url}/task_status/{task_id}", timeout=5)
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        self.log_test("Task Status", "PASS", f"Status: {status_data.get('state', 'unknown')}")
                    else:
                        self.log_test("Task Status", "WARN", "Could not check task status")
                    
                    return True
                else:
                    self.log_test("Worker Test", "FAIL", "", "No task_id returned")
                    return False
            else:
                self.log_test("Worker Test", "FAIL", "", f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Worker Test", "WARN", "Worker may not be running", str(e))
            return True  # Not critical
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all API tests."""
        print("🚀 STARTING API TESTS")
        print("="*50)
        
        # Check if server is running first
        if not self.test_server_running():
            print("\n❌ Server not running! Start it with:")
            print("   uvicorn main:app --reload")
            return self.results
        
        print("\n🔍 TESTING API ENDPOINTS\n" + "="*50)
        
        # Run API tests
        tests = [
            ("Bot Initialization", self.test_start_bot),
            ("Send Message", self.test_send_message),
            ("Structured Message", self.test_structured_message),
            ("Namespace Operations", self.test_namespace_operations),
            ("Worker Status", self.test_worker_status)
        ]
        
        for test_name, test_func in tests:
            try:
                test_func()
                time.sleep(1)  # Small delay between tests
            except Exception as e:
                self.log_test(test_name, "FAIL", "", f"Test crashed: {str(e)}")
        
        # Generate report
        self.generate_report()
        return self.results
    
    def generate_report(self):
        """Generate final test report."""
        total_tests = self.results["passed"] + self.results["failed"]
        
        print("\n" + "="*50)
        print("📊 API TEST REPORT")
        print("="*50)
        print(f"✅ Passed: {self.results['passed']}")
        print(f"❌ Failed: {self.results['failed']}")
        print(f"📈 Success Rate: {(self.results['passed']/total_tests)*100:.1f}%" if total_tests > 0 else "No tests run")
        
        if self.results["errors"]:
            print(f"\n🔴 ERRORS ({len(self.results['errors'])}):")
            for error in self.results["errors"]:
                print(f"   • {error}")
        
        if self.results["failed"] == 0:
            print("\n🎉 ALL API TESTS PASSED!")
        else:
            print(f"\n⚠️  {self.results['failed']} API tests failed")
        
        print("="*50)


def main():
    """Main API test execution."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("""
API Test Script for server_uni

Usage:
    python test_api.py                      # Test localhost:8000
    python test_api.py http://localhost:5000 # Test custom URL
    python test_api.py --help               # Show this help

Prerequisites:
    1. Server must be running: uvicorn main:app --reload
    2. Redis should be running for worker tests
    3. Environment variables should be set (.env file)
            """)
            return
        else:
            # Custom URL provided
            base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8000"
    
    print(f"Testing API at: {base_url}")
    
    tester = APITester(base_url)
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main() 