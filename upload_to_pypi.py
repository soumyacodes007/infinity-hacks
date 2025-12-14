#!/usr/bin/env python3
"""
Manual PyPI upload script for Oumi Hospital
"""

import subprocess
import sys
import os

def upload_to_testpypi():
    """Upload to TestPyPI"""
    print("🧪 Uploading to TestPyPI...")
    print("When prompted:")
    print("  Username: __token__")
    print("  Password: [paste your TestPyPI token]")
    print()
    
    try:
        subprocess.run([
            sys.executable, "-m", "twine", "upload", 
            "--repository", "testpypi", 
            "dist/*"
        ], check=True)
        
        print("✅ Successfully uploaded to TestPyPI!")
        print("Test installation with:")
        print("pip install --index-url https://test.pypi.org/simple/ oumi-hospital")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Upload failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have a TestPyPI account")
        print("2. Create an API token at https://test.pypi.org/manage/account/token/")
        print("3. Use '__token__' as username and your token as password")

def upload_to_pypi():
    """Upload to real PyPI"""
    print("🚀 Uploading to PyPI...")
    print("When prompted:")
    print("  Username: __token__")
    print("  Password: [paste your PyPI token]")
    print()
    
    try:
        subprocess.run([
            sys.executable, "-m", "twine", "upload", 
            "dist/*"
        ], check=True)
        
        print("✅ Successfully uploaded to PyPI!")
        print("Install with:")
        print("pip install oumi-hospital")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Upload failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have a PyPI account")
        print("2. Create an API token at https://pypi.org/manage/account/token/")
        print("3. Use '__token__' as username and your token as password")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python upload_to_pypi.py test    # Upload to TestPyPI")
        print("  python upload_to_pypi.py real    # Upload to PyPI")
        return
    
    if sys.argv[1] == "test":
        upload_to_testpypi()
    elif sys.argv[1] == "real":
        upload_to_pypi()
    else:
        print("Invalid option. Use 'test' or 'real'")

if __name__ == "__main__":
    main()