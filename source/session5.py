import getpass
import logging
import os
import sys
import time

# Set python path to allow acceding to the modules inside /source without need
# to specify 'from source ...'
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Config to run on one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = getpass.getuser()[-1]

from GPyOpt.methods import BayesianOptimization
from keras import Input, optimizers
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.models import Model
import numpy as np
import matplotlib

# Force matplotlib to not use any Xwindows backend. If you need to import
# pyplot, do it after setting `Agg` as the backend.
matplotlib.use('Agg')

from source import TEST_PATH
from source import VALIDATION_PATH
from source import TRAIN_PATH

from CNN import CNN

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
cross_validate = True
# Load pre-trained model or generate from scratch
load_model = False
# select number of epochs
n_epochs = 1


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
        x = MaxPooling2D(pool_size=(2, 2), padding='valid', name='pool1')(x)
        x = Conv2D(16, (3, 3), activation='relu', name='conv2')(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='valid', name='pool2')(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu', name='fc1')(x)
        x = Dense(128, activation='relu', name='fc2')(x)
        x = Dropout(0.5)(x)
        main_output = Dense(
            units=8, activation='softmax', name='predictions')(x)

        # Compile the model
        return Model(inputs=main_input, outputs=main_output, name='model2')

    def _model3():
        main_input = Input(shape=(image_size, image_size, 3),
                           dtype='float32',
                           name='main_input')
        x = Conv2D(64, (3, 3), activation='relu', name='conv1')(main_input)
        x = MaxPooling2D(pool_size=(2, 2), padding='valid', name='pool1')(x)
        x = Conv2D(32, (3, 3), activation='relu', name='conv2')(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='valid', name='pool2')(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu', name='fc2')(x)
        x = Dropout(0.5)(x)
        main_output = Dense(
            units=8, activation='softmax', name='predictions')(x)

        # Compile the model
        return Model(inputs=main_input, outputs=main_output, name='model3')

    return {
        1: _model1(),
        2: _model2(),
        3: _model3()
    }.get(model_id)


def get_optimizer(optimizer_id,
                  learning_rate):  # type: (int, int) -> optimizer
    """ Gets a optimizer by its id.

     Args:
         optimizer_id: optimizer id
         learning_rate: learning rate of the optimizer
    """

    if optimizer_id == 1:
        optimizer = optimizers.SGD(lr=learning_rate)
    elif optimizer_id == 2:
        optimizer = optimizers.RMSprop(lr=learning_rate)
    elif optimizer_id == 3:
        optimizer = optimizers.Adam(lr=learning_rate)
    else:
        optimizer = optimizers.Adadelta(lr=learning_rate)

    return optimizer


def do_cross_validation():
    # Random Search
    bounds = [
        {'name': 'model_id', 'type': 'discrete',
         'domain': (1, 2, 3)},
        {'name': 'image_size', 'type': 'discrete',
         'domain': (32, 64, 128, 256)},
        {'name': 'batch_size', 'type': 'discrete',
         'domain': (64,)},
        # {'name': 'batch_size', 'type': 'discrete',
        #  'domain': (16, 32, 64)},
        {'name': 'optimizer_id', 'type': 'discrete',
         'domain': (4,)},
        # {'name': 'optimizer_id', 'type': 'discrete',
        #  'domain': (1, 2, 3, 4)},
        {'name': 'lr', 'type': 'discrete',
         'domain': (0.1, 0.01, 0.001, 0.0001)}]

    optimizer = BayesianOptimization(f=train_and_validate,
                                     domain=bounds,
                                     verbosity=True)
    optimizer.run_optimization(max_iter=2,
                               verbosity=True,
                               report_file='optimizer_results.txt')
    logger.info('optimized parameters: {}'.format(optimizer.x_opt))
    logger.info('optimized accuracy: {}'.format(optimizer.fx_opt))


def train_and_validate(bounds):
    b = bounds.astype(np.int64)
    model_id, image_size, batch_size, optimizer_id, lr = \
        b[:, 0][0], b[:, 1][0], b[:, 2][0], b[:, 3][0], b[:, 4][0]
    logger.info('Bounds in action {}'.format(bounds))

    MODEL_PATH = 'results/session5/CNN_{}_{}.h5'.format(model_id,
                                                        int(time.time()))
    neural_network = CNN(logger,
                         train_path=TRAIN_PATH,
                         validation_path=VALIDATION_PATH,
                         test_path=TEST_PATH,
                         model_fname=MODEL_PATH)

    # Hyper-parameters selection
    neural_network.set_batch_size(batch_size)
    neural_network.set_model(
        model=get_model(model_id=model_id, image_size=image_size))
    neural_network.set_optimizer(get_optimizer(optimizer_id, lr))
    neural_network.set_loss_function('categorical_crossentropy')
    neural_network.set_metrics(['accuracy'])

    # Configure and build the NN
    neural_network.configure()
    neural_network.build()

    # Train
    neural_network.train_CNN_model(n_epochs=n_epochs,
                                   steps_per_epoch_multiplier=10,
                                   validation_steps_multiplier=1)

    neural_network.plot_history()
    neural_network.plot_results()


if __name__ == "__main__":
    init = time.time()

    # Train and validate the model optimizing hyper-parameters
    if cross_validate:
        do_cross_validation()

    # Load the model if exists, train otherwise
    else:
        MODEL_PATH = 'results/session5/my_CNN.h5'
        neural_network = CNN(logger,
                             train_path=TRAIN_PATH,
                             validation_path=VALIDATION_PATH,
                             test_path=TEST_PATH,
                             model_fname=MODEL_PATH)

        if load_model:
            neural_network.load_CNN_model()
        else:
            # Hyper-parameters selection
            neural_network.set_batch_size(16)
            neural_network.set_model(
                model=get_model(model_id=2, image_size=64))
            neural_network.set_optimizer(optimizers.Adadelta(lr=0.1))
            neural_network.set_loss_function('categorical_crossentropy')
            neural_network.set_metrics(['accuracy'])

            neural_network.configure()
            neural_network.build()

            neural_network.train_CNN_model(n_epochs=n_epochs,
                                           steps_per_epoch_multiplier=20,
                                           validation_steps_multiplier=1)

            neural_network.plot_history()

        neural_network.plot_results()

    end = time.time()
    logger.info('Everything done in {} secs.\n'.format(str(end - init)))
