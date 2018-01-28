import os
import sys

# Set project paths
# ROOT_PATH = '/home/master10/scene-classificator'
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
SOURCE_PATH = os.path.join(ROOT_PATH, 'source')
DATA_PATH = os.path.join(ROOT_PATH, 'data')
DATA_PATCHES_PATH = os.path.join(ROOT_PATH, 'data_patch')

# Default dataset (70/18/12%)
# The dataset train contains 1881 images (70%)
# The dataset validation contains 487 images (18%)
# The dataset test contains 320 images (12%)
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
VALIDATION_PATH = os.path.join(DATA_PATH, 'validation')
TEST_PATH = os.path.join(DATA_PATH, 'test')

# Special datasets
SMALL_TRAIN_PATH = os.path.join(DATA_PATH, 'train_small')
TOY_TRAIN_PATH = os.path.join(DATA_PATH, 'train_toy')

# REMOVE THIS LINE!!
# TRAIN_PATH = os.path.join(DATA_PATH, 'train_toy')

# Add project path to PYTHONPATH env var
sys.path.append(ROOT_PATH)
