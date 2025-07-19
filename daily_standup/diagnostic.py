#!/usr/bin/env python3
"""
Quick diagnostic script to test the modular Daily Standup system
Run this to verify all modules load correctly and identify any issues
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all modules can be imported successfully"""
    print("🧪 Testing module imports...")
    
    try:
        # Add current directory to Python path for imports
        sys.path.insert(0, str(Path.cwd()))
        
        # Test core module imports
        print("  📦 Testing core imports...")
        from core import config, DatabaseManager, SessionData, SessionStage
        print("    ✅ Core basic imports successful")
        
        from core import SummaryManager, OptimizedAudioProcessor, UltraFastTTSProcessor
        print("    ✅ Core AI service imports successful")
        
        from core import shared_clients, prompts
        print("    ✅ Core shared components successful")
        
        # Test SessionData creation
        print("  🏗️ Testing SessionData creation...")
        session_data = SessionData(
            session_id="test_123",
            test_id="test_session",
            student_id=999,
            student_name="Test User",
            session_key="TEST_KEY",
            created_at=1234567890.0,
            last_activity=1234567890.0,
            current_stage=SessionStage.GREETING,
            websocket=None  # This should work now
        )
        print("    ✅ SessionData creation successful")
        
        # Test environment loading
        print("  🌍 Testing environment variables...")
        print(f"    USE_DUMMY_DATA: {config.USE_DUMMY_DATA}")
        print(f"    DEBUG_MODE: {config.DEBUG_MODE}")
        print(f"    OpenAI Model: {config.OPENAI_MODEL}")
        print("    ✅ Config loading successful")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Import test failed: {e}")
        print(f"    Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_directory_structure():
    """Test if directory structure is correct"""
    print("📁 Testing directory structure...")
    
    current_dir = Path.cwd()
    expected_files = [
        "main.py",
        ".env",
        "core/__init__.py", 
        "core/config.py",
        "core/database.py",
        "core/ai_services.py",
        "core/prompts.py"
    ]
    
    missing_files = []
    for file_path in expected_files:
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ Missing files: {missing_files}")
        return False
    
    print("  ✅ All required files present")
    return True

def test_fastapi_startup():
    """Test if FastAPI app can start"""
    print("🚀 Testing FastAPI app creation...")
    
    try:
        # Add current directory to Python path for imports
        sys.path.insert(0, str(Path.cwd()))
        
        # This should work if all imports are correct
        import main
        app = main.app
        print("  ✅ FastAPI app import successful")
        
        # Test basic endpoints exist
        routes = [route.path for route in app.routes]
        expected_routes = ["/start_test", "/health", "/test"]
        
        for route in expected_routes:
            if route in routes:
                print(f"  ✅ Route {route} exists")
            else:
                print(f"  ❌ Route {route} missing")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FastAPI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests"""
    print("🔍 Daily Standup System Diagnostic")
    print("=" * 50)
    
    # Change to script directory to ensure relative imports work
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"📍 Working directory: {Path.cwd()}")
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Module Imports", test_imports),
        ("FastAPI Startup", test_fastapi_startup)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Score: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your modular setup is working correctly.")
        print("✅ You can now start your FastAPI server with confidence.")
    else:
        print("⚠️ Some tests failed. Please fix the issues above before starting the server.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())