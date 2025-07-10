"""
Setup script for Dynamic Conversation Evaluator testing
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        "openai",
        "asyncio",
        "json",
        "logging"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - MISSING")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_api_key():
    """Check if OpenAI API key is set"""
    print("\nChecking OpenAI API key...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("✓ OPENAI_API_KEY is set")
        return True
    else:
        print("✗ OPENAI_API_KEY is not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return False

def check_files():
    """Check if required files exist"""
    print("\nChecking required files...")
    
    required_files = [
        "conversation_data.json",
        "conversation_evaluator.py",
        "test_conversation_evaluator.py"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            missing_files.append(file)
            print(f"✗ {file} - MISSING")
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        return False
    
    return True

def check_backend_modules():
    """Check if backend modules are accessible"""
    print("\nChecking backend modules...")
    
    # Add backend directory to path
    backend_path = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(backend_path)
    
    required_modules = [
        "intelligence_layer.engineer_chat_agent",
        "intelligence_layer.network_knowledge_agent",
        "knowledge_layer",
        "network_layer.simulation_engine",
        "settings"
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            missing_modules.append(module)
            print(f"✗ {module} - MISSING: {e}")
    
    if missing_modules:
        print(f"\nMissing modules: {missing_modules}")
        print("Make sure you're running from the correct directory and the backend is properly set up.")
        return False
    
    return True

def create_test_directory():
    """Create test output directory"""
    print("\nCreating test directory...")
    
    test_dir = "my_results"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"✓ Created {test_dir} directory")
    else:
        print(f"✓ {test_dir} directory already exists")
    
    return True

def run_quick_test():
    """Run a quick test to verify everything works"""
    print("\nRunning quick test...")
    
    try:
        # Import the evaluator
        from conversation_evaluator import DynamicConversationEvaluator
        
        # Create evaluator instance
        evaluator = DynamicConversationEvaluator()
        
        # Test loading conversation data
        data = evaluator.load_conversation_data()
        if data.get("questions"):
            print(f"✓ Loaded {len(data['questions'])} questions from conversation_data.json")
        else:
            print("✗ No questions found in conversation_data.json")
            return False
        
        print("✓ Quick test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("Dynamic Conversation Evaluator Setup")
    print("=" * 40)
    
    # Check all requirements
    checks = [
        ("Dependencies", check_dependencies),
        ("API Key", check_api_key),
        ("Files", check_files),
        ("Backend Modules", check_backend_modules),
        ("Test Directory", create_test_directory),
        ("Quick Test", run_quick_test)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All checks passed! You're ready to run the evaluator.")
        print("\nTo run the evaluator:")
        print("python test_conversation_evaluator.py")
        print("\nOr run individual tests:")
        print("python conversation_evaluator.py --output-dir my_results")
    else:
        print("✗ Some checks failed. Please fix the issues above before running the evaluator.")
    
    return all_passed

if __name__ == "__main__":
    main() 