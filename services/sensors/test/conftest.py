# test/conftest.py
import sys
import os
from pathlib import Path

# Add application root to Python path
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))