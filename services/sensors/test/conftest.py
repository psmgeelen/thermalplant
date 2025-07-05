# test/conftest.py
print("Loading conftest.py from:", __file__)

import sys
import os
from pathlib import Path

# Add application root to Python path
root_path = Path(__file__).parent.parent
sys.path.insert(0, str(root_path))

# Add the app/sensors directory to Python path (where utils.py is located)
sensors_path = root_path / 'app' / 'sensors'
if sensors_path.exists() and sensors_path.is_dir():
    sys.path.insert(0, str(sensors_path))

# Add the app directory to Python path
app_path = root_path / 'app'
if app_path.exists() and app_path.is_dir():
    sys.path.insert(0, str(app_path))

# Print the Python path for debugging
print(f"Python path: {sys.path}")