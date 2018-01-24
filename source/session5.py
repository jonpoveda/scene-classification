import getpass
import logging
import os
import sys
import time

import matplotlib
# Force matplotlib to not use any Xwindows backend.
from keras import Model, Input, optimizers
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
n_epochs = 50

MODEL_PATH = 'results/session5/my_CNN.h5'


def get_model(model_id, image_size):
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

        main_output = Dense(units=8, activation='softmax', name='predictions')(
            x)

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

        main_output = Dense(units=8, activation='softmax', name='predictions')(
            x)

        # Compile the model
        return Model(inputs=main_input, outputs=main_output, name='model2')

    return {
        1: _model1(),
        2: _model2()
    }.get(model_id)


if __name__ == "__main__":

    init = time.time()


    neural_network = CNN(logger,
                         batch_size=16,
                         dataset_dir=DATA_PATH,
                         model_fname=MODEL_PATH)

    model = get_model(model_id=2, image_size=64)
    opt = optimizers.Adadelta(lr=0.1)

    neural_network.set_model(model)
    neural_network.set_optimizer(opt)
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
    logger.info('Everything done in {} secs.\n'.format(str(end - init)))
