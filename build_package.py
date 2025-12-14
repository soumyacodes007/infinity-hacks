#!/usr/bin/env python3
"""
Build script for Oumi Hospital package
"""

import subprocess
import sys
import shutil
from pathlib import Path

def clean_build():
    """Clean previous build artifacts"""
    print("🧹 Cleaning build artifacts...")
    
    dirs_to_clean = ["build", "dist", "*.egg-info"]
    for pattern in dirs_to_clean:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                try:
                    shutil.rmtree(path)
                    print(f"   Removed {path}")
                except PermissionError:
                    print(f"   ⚠️ Could not remove {path} (permission denied)")
    
    # Clean __pycache__ directories
    for pycache in Path(".").rglob("__pycache__"):
        try:
            shutil.rmtree(pycache)
            print(f"   Removed {pycache}")
        except PermissionError:
            print(f"   ⚠️ Could not remove {pycache} (permission denied)")

def build_package():
    """Build the package"""
    print("📦 Building package...")
    
    try:
        # Build wheel and source distribution
        result = subprocess.run([
            sys.executable, "-m", "build"
        ], check=True, capture_output=True, text=True)
        
        print("✅ Package built successfully!")
        print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_package():
    """Check the built package"""
    print("🔍 Checking package...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "twine", "check", "dist/*"
        ], check=True, capture_output=True, text=True)
        
        print("✅ Package check passed!")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Package check failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def upload_to_pypi(test=True):
    """Upload to PyPI"""
    repo = "testpypi" if test else "pypi"
    print(f"🚀 Uploading to {repo}...")
    
    try:
        cmd = [sys.executable, "-m", "twine", "upload"]
        if test:
            cmd.extend(["--repository", "testpypi"])
        cmd.append("dist/*")
        
        result = subprocess.run(cmd, check=True)
        
        print(f"✅ Successfully uploaded to {repo}!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Upload failed: {e}")
        return False

def main():
    """Main build process"""
    print("🏥 Building Oumi Hospital Package")
    print("=" * 50)
    
    # Clean previous builds
    clean_build()
    
    # Build package
    if not build_package():
        sys.exit(1)
    
    # Check package
    if not check_package():
        sys.exit(1)
    
    print("\n🎉 Package ready for publishing!")
    print("\nNext steps:")
    print("1. Test upload: python build_package.py --test-upload")
    print("2. Real upload: python build_package.py --upload")
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        if "--test-upload" in sys.argv:
            upload_to_pypi(test=True)
        elif "--upload" in sys.argv:
            upload_to_pypi(test=False)

if __name__ == "__main__":
    main()