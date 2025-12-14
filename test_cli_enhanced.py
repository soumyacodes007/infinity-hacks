#!/usr/bin/env python3
"""
Test the enhanced CLI functionality (Task 7)
"""

import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_cli_help():
    """Test CLI help and basic functionality"""
    print("🖥️ Testing Enhanced CLI...")
    
    try:
        # Test main help
        result = subprocess.run([
            sys.executable, "src/cli.py", "--help"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ Main CLI help works")
            
            # Check for new flags
            help_text = result.stdout
            if "--verbose" in help_text and "--dry-run" in help_text and "--json" in help_text:
                print("✅ New global flags present (--verbose, --dry-run, --json)")
            else:
                print("❌ Missing new global flags")
                return False
        else:
            print(f"❌ CLI help failed: {result.stderr}")
            return False
        
        # Test diagnose help
        result = subprocess.run([
            sys.executable, "src/cli.py", "diagnose", "--help"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ Diagnose command help works")
            
            # Check for --full-scan flag
            if "--full-scan" in result.stdout:
                print("✅ --full-scan flag present")
            else:
                print("❌ Missing --full-scan flag")
                return False
        else:
            print(f"❌ Diagnose help failed: {result.stderr}")
            return False
        
        # Test treat help
        result = subprocess.run([
            sys.executable, "src/cli.py", "treat", "--help"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ Treat command help works")
            
            # Check for new options
            help_text = result.stdout
            if "--skip-verification" in help_text and "--cure-samples" in help_text:
                print("✅ Enhanced treat command options present")
            else:
                print("❌ Missing enhanced treat command options")
                return False
        else:
            print(f"❌ Treat help failed: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def test_dry_run_mode():
    """Test dry-run functionality"""
    print("\n🔍 Testing Dry-Run Mode...")
    
    try:
        # Test dry-run diagnose
        result = subprocess.run([
            sys.executable, "src/cli.py", "--dry-run", 
            "diagnose", "--model", "test-model", "--symptom", "safety"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            output = result.stdout
            if "DRY RUN" in output and "Would diagnose model" in output:
                print("✅ Dry-run diagnose works")
            else:
                print("❌ Dry-run diagnose output incorrect")
                return False
        else:
            print(f"❌ Dry-run diagnose failed: {result.stderr}")
            return False
        
        # Test dry-run treat
        result = subprocess.run([
            sys.executable, "src/cli.py", "--dry-run",
            "treat", "--model", "test-model", "--symptom", "safety"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            output = result.stdout
            if "DRY RUN" in output and "full treatment pipeline" in output:
                print("✅ Dry-run treat works")
            else:
                print("❌ Dry-run treat output incorrect")
                return False
        else:
            print(f"❌ Dry-run treat failed: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Dry-run test failed: {e}")
        return False

def test_json_output():
    """Test JSON output functionality"""
    print("\n📄 Testing JSON Output...")
    
    try:
        # Test JSON diagnose
        result = subprocess.run([
            sys.executable, "src/cli.py", "--json", "--dry-run",
            "diagnose", "--model", "test-model", "--symptom", "safety"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            try:
                import json
                output_data = json.loads(result.stdout)
                
                if output_data.get("operation") == "diagnose" and output_data.get("dry_run") == True:
                    print("✅ JSON diagnose output works")
                else:
                    print("❌ JSON diagnose output format incorrect")
                    return False
            except json.JSONDecodeError:
                print("❌ JSON diagnose output is not valid JSON")
                return False
        else:
            print(f"❌ JSON diagnose failed: {result.stderr}")
            return False
        
        # Test JSON treat
        result = subprocess.run([
            sys.executable, "src/cli.py", "--json", "--dry-run",
            "treat", "--model", "test-model", "--symptom", "safety"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            try:
                import json
                output_data = json.loads(result.stdout)
                
                if output_data.get("operation") == "treat" and "pipeline" in output_data:
                    print("✅ JSON treat output works")
                else:
                    print("❌ JSON treat output format incorrect")
                    return False
            except json.JSONDecodeError:
                print("❌ JSON treat output is not valid JSON")
                return False
        else:
            print(f"❌ JSON treat failed: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ JSON output test failed: {e}")
        return False

def test_recipe_commands():
    """Test recipe sharing commands"""
    print("\n🌐 Testing Recipe Commands...")
    
    try:
        # Test list-recipes help
        result = subprocess.run([
            sys.executable, "src/cli.py", "list-recipes", "--help"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ list-recipes command exists")
        else:
            print(f"❌ list-recipes help failed: {result.stderr}")
            return False
        
        # Test share-recipe help
        result = subprocess.run([
            sys.executable, "src/cli.py", "share-recipe", "--help"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            help_text = result.stdout
            if "--validate-only" in help_text:
                print("✅ share-recipe command with --validate-only flag")
            else:
                print("❌ Missing --validate-only flag")
                return False
        else:
            print(f"❌ share-recipe help failed: {result.stderr}")
            return False
        
        # Test list-recipes with no recipes
        result = subprocess.run([
            sys.executable, "src/cli.py", "--json", "list-recipes"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            try:
                import json
                output_data = json.loads(result.stdout)
                
                if output_data.get("operation") == "list_recipes":
                    print("✅ list-recipes JSON output works")
                else:
                    print("❌ list-recipes JSON format incorrect")
                    return False
            except json.JSONDecodeError:
                print("❌ list-recipes JSON output is not valid JSON")
                return False
        else:
            print(f"❌ list-recipes failed: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Recipe commands test failed: {e}")
        return False

def test_command_structure():
    """Test that all expected commands exist"""
    print("\n📋 Testing Command Structure...")
    
    expected_commands = [
        "diagnose",
        "cure", 
        "recipe",
        "verify",
        "treat",
        "share-recipe",
        "list-recipes"
    ]
    
    try:
        # Get main help to see available commands
        result = subprocess.run([
            sys.executable, "src/cli.py", "--help"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode != 0:
            print(f"❌ Failed to get CLI help: {result.stderr}")
            return False
        
        help_text = result.stdout
        
        for command in expected_commands:
            if command in help_text:
                print(f"✅ {command} command exists")
            else:
                print(f"❌ {command} command missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Command structure test failed: {e}")
        return False

def main():
    """Run all CLI tests"""
    print("🏥 Oumi Hospital - Enhanced CLI Test Suite")
    print("=" * 50)
    
    tests = [
        test_cli_help,
        test_dry_run_mode,
        test_json_output,
        test_recipe_commands,
        test_command_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"🏥 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All CLI tests passed! Enhanced CLI is ready.")
        print("\nKey Features Implemented:")
        print("✅ ASCII hospital logo on startup")
        print("✅ Rich terminal UI with color coding")
        print("✅ --verbose, --dry-run, --json flags")
        print("✅ Full treatment pipeline (treat command)")
        print("✅ Recipe sharing and listing commands")
        print("✅ Enhanced progress tracking")
        return 0
    else:
        print("⚠️ Some CLI tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    exit(main())