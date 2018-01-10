import time

from MLP import multi_layer_perceptron
from source import DATA_PATH
from utils import Color
from utils import colorprint

SVM = False
cross_validate = False
load_model = True

if __name__ == "__main__":

    init = time.time()

    # end to end MLP implementation
    neural_network = multi_layer_perceptron(img_size=32, batch_size=16,
                                            dataset_dir=DATA_PATH,
                                            model_fname='my_first_mlp.h5')
    neural_network.build_MLP_model()

    # Train or load model
    if not load_model:
        neural_network.train_MLP_model()
        neural_network.plot_history()
    else:
        neural_network.load_MLP_model()

    if not SVM:
        neural_network.plot_results()

    else:
        # feature extraction with MLP + SVM classification
        features, labels = neural_network.get_layer_output(layer='last',
                                                           image_set='train')
        if not cross_validate:
            neural_network.train_classifier_SVM(features, labels)
            features, labels = neural_network.get_layer_output(
                layer='last', image_set='test')
            neural_network.evaluate_performance_SVM(features, labels,
                                                    do_plotting=True)
        else:
            neural_network.cross_validate_SVM(features, labels)

    end = time.time()
    colorprint(Color.BLUE, 'Done in ' + str(end - init) + ' secs.\n')
