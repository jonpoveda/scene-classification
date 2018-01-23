import getpass
import logging
import os
import sys
import time

import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from source import DATA_PATH

# Config to run on one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = getpass.getuser()[-1]

from CNN import conv_neural_network

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Do cross-validation to find best parameters
cross_validate = False
# Load pre-trained model or generate from scratch
load_model = False
# select number of epochs
n_epochs = 5

MODEL_PATH = 'results/session5/my_CNN.h5'

# Create a file logger
logger = logging.getLogger('session5')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler = logging.FileHandler('session5.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

if __name__ == "__main__":

    init = time.time()

    neural_network = conv_neural_network(logger,
                                         img_size=256,
                                         batch_size=16,
                                         dataset_dir=DATA_PATH,
                                         model_fname=MODEL_PATH)

    neural_network.build_CNN_model()

    if cross_validate:
        neural_network.cross_validate()
    else:

        # Train or load model
        if load_model:
            neural_network.load_CNN_model()
        else:
            neural_network.train_CNN_model(n_epochs)
            neural_network.plot_history()
        neural_network.plot_results()

    end = time.time()
    logger.info('Everything done in ' + str(end - init) + ' secs.\n')
