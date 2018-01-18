import os
import shutil
import random
import numpy as np

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data')
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
REDUCED_TRAIN_PATH = os.path.join(DATA_PATH, 'reduced_train')

folders = os.listdir(TRAIN_PATH) 

for folder in folders:

    files = np.random.choice(os.listdir(os.path.join(TRAIN_PATH, folder)), 50)
    for image in files:
        file_to_copy = os.path.join(TRAIN_PATH, folder,image)
        path_to_copy = os.path.join(REDUCED_TRAIN_PATH, folder, image)
        shutil.copy(file_to_copy,path_to_copy)


            