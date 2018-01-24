import getpass
import logging
import os
import sys
import time

import matplotlib
# Force matplotlib to not use any Xwindows backend.
from GPyOpt.methods import BayesianOptimization
from keras import Input, optimizers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

matplotlib.use('Agg')

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from source import DATA_PATH

# Config to run on one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = getpass.getuser()[-1]

from CNN import CNN

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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

# Do cross-validation to find best parameters
cross_validate = False
# Load pre-trained model or generate from scratch
load_model = False
# select number of epochs
n_epochs = 1

MODEL_PATH = 'results/session5/my_CNN.h5'

from keras.models import Model


def get_model(model_id, image_size):  # type: (int, int) -> Model
    """ Gets a model by its id.

     Args:
         model_id: model id
         image_size: dimension of the image in pixels in width or height
            (squared images are expected)
     """

    def _model1():
        main_input = Input(shape=(image_size, image_size, 3),
                           dtype='float32',
                           name='main_input')

        x = Conv2D(32, (3, 3), activation='relu', name='conv1')(main_input)
        x = Conv2D(32, (3, 3), activation='relu', name='conv2')(x)
        x = MaxPooling2D(pool_size=(4, 4), padding='valid', name='pool')(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu', name='fc1')(x)
        x = Dense(128, activation='relu', name='fc2')(x)
        x = Dropout(0.5)(x)

        main_output = Dense(
            units=8, activation='softmax', name='predictions')(x)

        # Compile the model
        return Model(inputs=main_input, outputs=main_output, name='model1')

    def _model2():
        main_input = Input(shape=(image_size, image_size, 3),
                           dtype='float32',
                           name='main_input')
        x = Conv2D(32, (3, 3), activation='relu', name='conv1')(main_input)
        x = Conv2D(32, (3, 3), activation='relu', name='conv2')(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='valid', name='pool')(x)
        x = Conv2D(32, (3, 3), activation='relu', name='conv3')(x)
        x = Conv2D(32, (3, 3), activation='relu', name='conv4')(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='valid', name='pool2')(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu', name='fc1')(x)
        x = Dense(128, activation='relu', name='fc2')(x)
        x = Dropout(0.5)(x)
        main_output = Dense(
            units=8, activation='softmax', name='predictions')(x)

        # Compile the model
        return Model(inputs=main_input, outputs=main_output, name='model2')

    return {
        1: _model1(),
        2: _model2()
    }.get(model_id)


def do_cross_validation():
    # Random Search
    bounds = [
        {'name': 'model_id', 'type': 'discrete',
         'domain': (16, 32, 64)},
        {'name': 'batch_size', 'type': 'discrete',
         'domain': (16, 32, 64)},
        {'name': 'batch_size', 'type': 'discrete',
         'domain': (16, 32, 64)},
        {'name': 'fc1_size', 'type': 'discrete',
         'domain': (512, 1024, 2048, 4096)},
        {'name': 'fc2_size', 'type': 'discrete',
         'domain': (128, 256, 512, 1024)}]

    optimizer = BayesianOptimization(f=train_and_validate,
                                     domain=bounds,
                                     verbosity=True)
    optimizer.run_optimization(max_iter=10)
    logger.info('optimized parameters: {}'.format(optimizer.x_opt))
    logger.info('optimized accuracy: {}'.format(optimizer.fx_opt))


def train_and_validate():
    neural_network = CNN(logger,
                         dataset_dir=DATA_PATH,
                         model_fname=MODEL_PATH)

    # Hyper-parameters selection
    model = get_model(model_id=2, image_size=64)
    opt = optimizers.Adadelta(lr=0.1)
    neural_network.set_batch_size(16)
    neural_network.set_model(model=model)
    neural_network.set_optimizer(opt)

    # Configure and build the NN
    neural_network.configure()
    neural_network.build()

    # Train
    neural_network.train_CNN_model(n_epochs)

    neural_network.plot_history()
    neural_network.plot_results()


if __name__ == "__main__":
    init = time.time()

    # Train and validate the model optimizing hyper-parameters
    if cross_validate:
        do_cross_validation()

    # Load the model if exists, train otherwise
    else:
        neural_network = CNN(logger,
                             dataset_dir=DATA_PATH,
                             model_fname=MODEL_PATH)

        if load_model:
            neural_network.load_CNN_model()
        else:
            # Hyper-parameters
            model = get_model(model_id=2, image_size=64)
            opt = optimizers.Adadelta(lr=0.1)
            neural_network.set_batch_size(16)
            neural_network.set_model(model=model)
            neural_network.set_optimizer(opt)

            neural_network.configure()
            neural_network.build()

            neural_network.train_CNN_model(n_epochs, steps_per_epoch=1, validation_steps=1)
            neural_network.plot_history()

        neural_network.plot_results()

    end = time.time()
    logger.info('Everything done in {} secs.\n'.format(str(end - init)))
