import getpass
import os
import time

import numpy as np

from MLP import multi_layer_perceptron
from source import DATA_PATCHES_PATH, DATA_PATH
from utils import Color
from utils import colorprint

# Config to run on one GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = getpass.getuser()[-1]

# Use two output MLP
two_output = False
# Compute SVM
SVM = False
# Compute bag of visual words
BoVW = True
# Do cross-validation to find best parameters
cross_validate = False
# Load pre-trained model or generate from scratch
load_model = True

MODEL_PATH = '../results/session3/my_first_mlp.h5'


def get_nn(dataset_dir=DATA_PATH, load_model=False):
    # end to end MLP implementation
    neural_network = multi_layer_perceptron(img_size=32,
                                            batch_size=16,
                                            dataset_dir=dataset_dir,
                                            model_fname=MODEL_PATH)
    if two_output:
        neural_network.build_MLP_two_outputs_model()

    else:
        neural_network.build_MLP_model()

    # Train or load model
    if load_model:
        neural_network.load_MLP_model()
    else:
        neural_network.train_MLP_model()
        neural_network.plot_history()
    return neural_network


if __name__ == "__main__":
    init = time.time()
    if SVM:
        get_nn(DATA_PATH, load_model)

    elif BoVW:
        # Train model using patches of images instead of whole images
        neural_network = get_nn(DATA_PATCHES_PATH, load_model)
        # features, labels = neural_network.get_layer_output(
        #     layer=neural_network.LAYERS.LAST, image_set='train')

        # FIXME: remove this two lines
        features = np.random.rand(64 * 2, 1024)
        labels = np.random.rand(1024)
        if cross_validate:
            neural_network.cross_validate_BoVW(features, labels)
        else:
            neural_network.train_classifier_BoVW(features, labels)
            features, labels = neural_network.get_layer_output(
                layer=neural_network.LAYERS.LAST, image_set='test')
            neural_network.evaluate_performance_BoVW(features, labels,
                                                     do_plotting=False)

    else:
        neural_network = get_nn(DATA_PATH)
        neural_network.plot_results()

    end = time.time()
    colorprint(Color.BLUE, 'Done in ' + str(end - init) + ' secs.\n')
