import os
import sys

# Set project paths

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
SOURCE_PATH = os.path.join(ROOT_PATH, 'source')
DATA_PATH = os.path.join(ROOT_PATH, 'data')
DATA_PATCHES_PATH = os.path.join(ROOT_PATH, 'data_patch')
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
TEST_PATH = os.path.join(DATA_PATH, 'test')
REDUCED_TRAIN_PATH = os.path.join(DATA_PATH, 'reduced_train')

# Add project path to PYTHONPATH env var
sys.path.append(ROOT_PATH)
