# Config for an specific GPU
# import getpass
# os.environ["CUDA_VISIBLE_DEVICES"]=getpass.getuser()[-1]

import time

from MLP import multi_layer_perceptron
from source import DATA_PATH
from utils import Color
from utils import colorprint

# Compute SVM
SVM = False
# Compute bag of visual words
BoVW = True
# Do cross-validation to find best parameters
cross_validate = False
# Load pre-trained model or generate from scratch
load_model = True

MODEL_PATH = '../results/session3/my_first_mlp.h5'

if __name__ == "__main__":
    init = time.time()

    # end to end MLP implementation
    neural_network = multi_layer_perceptron(img_size=32,
                                            batch_size=16,
                                            dataset_dir=DATA_PATH,
                                            model_fname=MODEL_PATH)
    neural_network.build_MLP_model()

    # Train or load model
    if not load_model:
        neural_network.train_MLP_model()
        neural_network.plot_history()
    else:
        neural_network.load_MLP_model()

    if SVM:
        # feature extraction with MLP + SVM classification
        features, labels = neural_network.get_layer_output(
            layer=neural_network.LAYERS.LAST, image_set='train')
        if not cross_validate:
            neural_network.train_classifier_SVM(features, labels)
            features, labels = neural_network.get_layer_output(
                layer=neural_network.LAYERS.LAST, image_set='test')
            neural_network.evaluate_performance_SVM(features, labels,
                                                    do_plotting=True)
        else:
            neural_network.cross_validate_SVM(features, labels)

    elif BoVW:
        features, labels = neural_network.get_layer_output(
            layer=neural_network.LAYERS.LAST, image_set='train')

        if not cross_validate:
            neural_network.train_classifier_BoVW(features, labels)
            features, labels = neural_network.get_layer_output(
                layer=neural_network.LAYERS.LAST, image_set='test')
            neural_network.evaluate_performance_BoVW(features, labels,
                                                     do_plotting=True)
        else:
            neural_network.cross_validate_BoVW(features, labels)
    else:
        neural_network.plot_results()

    end = time.time()
    colorprint(Color.BLUE, 'Done in ' + str(end - init) + ' secs.\n')
