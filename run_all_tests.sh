#!/bin/bash

echo "🧪 COMPREHENSIVE TEST SUITE FOR SERVER_UNI"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    case $2 in
        "SUCCESS") echo -e "${GREEN}✅ $1${NC}" ;;
        "ERROR")   echo -e "${RED}❌ $1${NC}" ;;
        "WARNING") echo -e "${YELLOW}⚠️  $1${NC}" ;;
        *)         echo -e "$1" ;;
    esac
}

# Check if virtual environment is activated
check_venv() {
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        print_status "Virtual environment active: $(basename $VIRTUAL_ENV)" "SUCCESS"
        return 0
    else
        print_status "No virtual environment detected" "WARNING"
        echo "Attempting to activate venv..."
        if [ -d "venv" ]; then
            source venv/bin/activate
            print_status "Virtual environment activated" "SUCCESS"
            return 0
        else
            print_status "No venv directory found. Please run ./update_dependencies.sh first" "ERROR"
            return 1
        fi
    fi
}

# Install requests if needed for API tests
install_requests() {
    python -c "import requests" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Installing requests for API tests..."
        pip install requests
    fi
}

# Check if .env file exists
check_env_file() {
    if [ -f ".env" ]; then
        print_status ".env file found" "SUCCESS"
        
        # Check for required keys (without revealing values)
        if grep -q "PINECONE_API_KEY" .env && grep -q "OPENAI_API_KEY" .env; then
            print_status "Required API keys found in .env" "SUCCESS"
        else
            print_status "Missing API keys in .env file" "WARNING"
            echo "Make sure PINECONE_API_KEY and OPENAI_API_KEY are set"
        fi
    else
        print_status ".env file not found" "WARNING"
        echo "Create .env file with your API keys for full testing"
    fi
}

# Main execution
main() {
    echo
    echo "🔧 Pre-flight checks..."
    echo "----------------------"
    
    # Check virtual environment
    if ! check_venv; then
        exit 1
    fi
    
    # Install requests
    install_requests
    
    # Check environment file
    check_env_file
    
    echo
    echo "🚀 Starting comprehensive tests..."
    echo "--------------------------------"
    
    # Run system tests
    echo
    print_status "PHASE 1: System Tests (Dependencies, Connections, Core Logic)" "WARNING"
    echo "=============================================================="
    python test_all.py
    system_exit_code=$?
    
    if [ $system_exit_code -eq 0 ]; then
        print_status "System tests PASSED" "SUCCESS"
    else
        print_status "System tests FAILED" "ERROR"
        echo
        echo "❌ System tests failed. Fix critical issues before proceeding."
        echo "   Check dependency installation and API keys."
        exit 1
    fi
    
    echo
    echo "⏸️  Pausing before API tests..."
    echo "   If you want to run API tests, start the server in another terminal:"
    echo "   uvicorn main:app --reload"
    echo
    read -p "Press Enter to continue with API tests, or Ctrl+C to exit..."
    
    # Run API tests
    echo
    print_status "PHASE 2: API Tests (FastAPI Endpoints)" "WARNING"
    echo "======================================="
    python test_api.py
    api_exit_code=$?
    
    if [ $api_exit_code -eq 0 ]; then
        print_status "API tests PASSED" "SUCCESS"
    else
        print_status "API tests FAILED (Server may not be running)" "WARNING"
        echo "   Start server with: uvicorn main:app --reload"
    fi
    
    # Final report
    echo
    echo "📊 FINAL TEST SUMMARY"
    echo "===================="
    
    if [ $system_exit_code -eq 0 ] && [ $api_exit_code -eq 0 ]; then
        print_status "🎉 ALL TESTS PASSED! Application is ready for use." "SUCCESS"
        exit 0
    elif [ $system_exit_code -eq 0 ]; then
        print_status "✅ System tests passed, API tests need server running" "WARNING"
        exit 0
    else
        print_status "❌ Critical system issues found. Fix before proceeding." "ERROR"
        exit 1
    fi
}

# Help function
show_help() {
    echo "Comprehensive Test Suite for server_uni"
    echo
    echo "Usage:"
    echo "  ./run_all_tests.sh         # Run all tests"
    echo "  ./run_all_tests.sh --help  # Show this help"
    echo
    echo "This script will:"
    echo "1. Check virtual environment and dependencies"
    echo "2. Run system tests (dependencies, connections, core logic)"
    echo "3. Run API tests (requires server to be running)"
    echo
    echo "Prerequisites:"
    echo "- Virtual environment with dependencies installed"
    echo "- .env file with API keys"
    echo "- For API tests: uvicorn main:app --reload"
}

# Check for help argument
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

# Run main function
main 