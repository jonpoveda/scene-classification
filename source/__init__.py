import os
import sys

# Set project paths

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
SOURCE_PATH = os.path.join(ROOT_PATH, 'source')
DATA_PATH = os.path.join(ROOT_PATH, 'data')

# Add project path to PYTHONPATH env var
sys.path.append(ROOT_PATH)
