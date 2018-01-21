import getpass
import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.utils.vis_utils import plot_model as plot

from data_generator import DataGenerator
from data_generator_config import DataGeneratorConfig

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from source import TEST_PATH
from source import REDUCED_TRAIN_PATH
from evaluator import Evaluator

# Config to run on one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = getpass.getuser()[-1]

# Create a file logger
logger = logging.getLogger('session4')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler = logging.FileHandler('session4.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Top-level vars
VALIDATION_PATH = TEST_PATH
img_width, img_height = 224, 224
plot_history = True
running_in_server = True

if running_in_server:
    batch_size = 32
    number_of_epoch = 20
else:
    # Just a toy parameters to try everything is working
    batch_size = 5
    number_of_epoch = 1


def get_base_model():
    """ create the base pre-trained model """
    base_model = VGG16(weights='imagenet')
    plot(base_model,
         to_file='../results/session4/modelVGG16a.png',
         show_shapes=True,
         show_layer_names=True)
    return base_model


def modify_last_fc_to_classify_eight_classes(base_model):
    """ Task 0: Modify to classify 8 classes.

    Get the second-to-last layer and add a FC to classify scenes (8-class classifier).
    Namely, change the last layer from 1000 classes to 8 classes.
    Freeze the former layers to not train them.
    """
    x = base_model.layers[-2].output
    x = Dense(8, activation='softmax', name='predictions')(x)

    for layer in base_model.layers:
        layer.trainable = False

    model = Model(inputs=base_model.input, outputs=x)
    plot(model,
         to_file='../results/session4/modelVGG16b.png',
         show_shapes=True,
         show_layer_names=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model


def modify_model_before_block4(base_model, dropout=False):
    """ Task 4

    Introduce and evaluate the usage of a dropout layer.
    """
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.layers[-10].output
    x = MaxPooling2D(pool_size=(4, 4), padding='valid', name='pool')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Dense(8, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
    plot(model,
         to_file='../results/session4/modelVGG16e.png',
         show_shapes=True,
         show_layer_names=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def modify_model_before_block3(base_model, dropout=False):
    """ Task 4

    Introduce and evaluate the usage of a dropout layer.
    """
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.layers[-10].output
    x = MaxPooling2D(pool_size=(4, 4), padding='valid', name='pool')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Dense(8, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
    plot(model,
         to_file='../results/session4/modelVGG16f.png',
         show_shapes=True,
         show_layer_names=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def unlock_layers(base_model):
    """ Task 1

    unlock all layer for training.
    """

    for layer in base_model.layers:
        layer.trainable = True

    model = Model(inputs=base_model.input,
                  outputs=base_model.layers[-1].output)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    for layer in model.layers:
        logger.debug([layer.name, layer.trainable])

    return model


def do_plotting(history, history2, cm=None):
    if history2:
        # summarize history for accuracy
        plt.plot(history.history['acc'] + history2.history['acc'])
        plt.plot(history.history['val_acc'] + history2.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('../results/session4/accuracy.jpg')
        plt.close()

        # summarize history for loss
        plt.plot(history.history['loss'] + history2.history['loss'])
        plt.plot(history.history['val_loss'] + history2.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('../results/session4/loss.jpg')

    else:
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('../results/session4/accuracy.jpg')
        plt.close()

        # summarize history for loss
        plt.plot(history.history['loss'] + history2.history['loss'])
        plt.plot(history.history['val_loss'] + history2.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('../results/session4/loss.jpg')

    if cm:
        print(cm)
        plt.matshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        plt.savefig('cm.jpg')


def main():
    base_model = get_base_model()
    logger.debug('Trainability of the layers:')
    model = modify_model_before_block4(base_model, dropout=False)
    # model = modify_model_before_block3(base_model, dropout=False)
    for layer in model.layers:
        logger.debug([layer.name, layer.trainable])

    # Get train, validation and test dataset
    # preprocessing_function=preprocess_input,
    # data_generator = ImageDataGenerator(**DataGeneratorConfig.DEFAULT)
    # data_generator = ImageDataGenerator(**DataGeneratorConfig.CONFIG1)

    data_gen = DataGenerator(img_width, img_height, batch_size,
                             REDUCED_TRAIN_PATH)
    data_gen.configure(DataGeneratorConfig.CONFIG2)

    if running_in_server:
        train_generator, test_generator, validation_generator = data_gen.get(
            train_path=REDUCED_TRAIN_PATH,
            test_path=TEST_PATH,
            validate_path=TEST_PATH)

        init = time.time()
        history = model.fit_generator(train_generator,
                                      steps_per_epoch=(int(
                                          400 * 1881 / 1881 // batch_size) + 1),
                                      epochs=number_of_epoch,
                                      validation_data=validation_generator,
                                      validation_steps=807 // 64)

        # unlock all layers and train
        model = unlock_layers(model)
        history2 = model.fit_generator(train_generator,
                                       steps_per_epoch=(int(
                                           400 * 1881 / 1881 // batch_size) + 1),
                                       epochs=number_of_epoch,
                                       validation_data=validation_generator,
                                       validation_steps=807 // 64)
        end = time.time()
        logger.info('[Training] Done in ' + str(end - init) + ' secs.\n')

        init = time.time()
        scores = model.evaluate_generator(test_generator, steps=807)
        end = time.time()
        logger.info('[Evaluation] Done in ' + str(end - init) + ' secs.\n')

        # Get ground truth
        test_labels = test_generator.classes

        # Predict test images
        predictions_raw = model.predict_generator(test_generator)
        predictions = []
        for prediction in predictions_raw:
            predictions.append(np.argmax(prediction))
        # Evaluate results
        evaluator = Evaluator(test_labels, predictions,
                              label_list=list([0, 1, 2, 3, 4, 5, 6, 7]))

        logger.info(
            'Evaluator \n'
            'Acc (model)\n'
            'Accuracy: {} \n'
            'Precision: {} \n'
            'Recall: {} \n'
            'Fscore: {}'.
            format(scores[1], evaluator.accuracy, evaluator.precision,
                   evaluator.recall, evaluator.fscore) + '\n')
        cm = evaluator.confusion_matrix()

        # Plot the confusion matrix on test data
        logger.info('Confusion matrix:\n')
        logger.info(cm)
        logger.info('Final accuracy: ' + str(evaluator.accuracy) + '\n')

        end = time.time()
        logger.info('Done in ' + str(end - init) + ' secs.\n')

    else:
        logger.info('Running in a laptop! Toy mode active')
        train_generator, test_generator, validation_generator = data_gen.get(
            train_path='../data-toy/train',
            test_path='../data-toy/test',
            validate_path='../data-toy/test')

        init = time.time()
        history = model.fit_generator(train_generator,
                                      steps_per_epoch=1,
                                      epochs=1,
                                      validation_data=validation_generator,
                                      validation_steps=10)
        end = time.time()
        logger.info('[Training] Done in ' + str(end - init) + ' secs.\n')

        init = time.time()
        result = model.evaluate_generator(test_generator, steps=10)
        end = time.time()
        logger.info('[Evaluation] Done in ' + str(end - init) + ' secs.\n')

    logger.debug(result)

    # list all data in history
    if plot_history:
        do_plotting(history, history2, cm)


if __name__ == '__main__':
    try:
        os.makedirs('../results/session4')
    except OSError as expected:
        # Expected when the folder already exists
        pass
    logger.info('Start')
    main()
    logger.info('End')
