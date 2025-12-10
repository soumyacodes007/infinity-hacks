#!/usr/bin/env python3
"""
🏥 Oumi Model Hospital - CLI Entry Point

Main entry point for the hospital CLI.
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from cli import main

if __name__ == "__main__":
    main()