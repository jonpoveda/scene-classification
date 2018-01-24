import os
import sys

# Set project paths
# ROOT_PATH = '/home/master10/scene-classificator'
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
SOURCE_PATH = os.path.join(ROOT_PATH, 'source')
DATA_PATH = os.path.join(ROOT_PATH, 'data')
DATA_PATCHES_PATH = os.path.join(ROOT_PATH, 'data_patch')

# Default dataset
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
VALIDATION_PATH = os.path.join(DATA_PATH, 'validation')
TEST_PATH = os.path.join(DATA_PATH, 'test')

# Special datasets
SMALL_TRAIN_PATH = os.path.join(DATA_PATH, 'train_small')
TOY_TRAIN_PATH = os.path.join(DATA_PATH, 'train_toy')

# Add project path to PYTHONPATH env var
sys.path.append(ROOT_PATH)
