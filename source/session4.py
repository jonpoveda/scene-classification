import getpass
import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from GPyOpt.methods import BayesianOptimization
from keras import optimizers
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
from source import SMALL_TRAIN_PATH
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
plot_history = False

batch_size = 32
number_of_epoch = 20


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
                  metrics=['accuracy', 'top_k_categorical_accuracy'])
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
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model


def modify_model_before_block3(base_model, dropout=False):
    """ Task 4

    Introduce and evaluate the usage of a dropout layer.
    """
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.layers[-13].output
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
                  optimizer='adadelta',
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
    opt = optimizers.Adadelta(lr=0.1)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
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
        logger(cm)
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
                             SMALL_TRAIN_PATH)
    data_gen.configure(DataGeneratorConfig.NORMALISE)

    train_generator, test_generator, validation_generator = data_gen.get(
        train_path=SMALL_TRAIN_PATH,
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
    scores = model.evaluate_generator(test_generator, steps=807 // 64)
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

    # list all data in history
    if plot_history:
        do_plotting(history, history2, cm)


def modify(base_model, fc1_size, fc2_size, dropout=False):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.layers[-10].output
    x = MaxPooling2D(pool_size=(4, 4), padding='valid', name='pool')(x)
    x = Flatten()(x)
    x = Dense(fc1_size, activation='relu', name='fc1')(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Dense(fc2_size, activation='relu', name='fc2')(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Dense(8, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
    plot(model,
         to_file='../results/session4/modelVGG16g_{}_{}.png'.format(
             fc1_size, fc2_size),
         show_shapes=True,
         show_layer_names=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model


def function_to_optimize(bounds):  # type: (ndarray) -> int
    b = bounds.astype(np.int64)
    batch_size, fc1_size, fc2_size = b[:, 0][0], b[:, 1][0], b[:, 2][0]
    logger.info('Bounds in action {}'.format(bounds))

    base_model = get_base_model()
    logger.debug('Trainability of the layers:')
    model = modify(base_model, fc1_size, fc2_size, dropout=False)

    for layer in model.layers:
        logger.debug([layer.name, layer.trainable])

    data_gen = DataGenerator(img_width, img_height, batch_size,
                             SMALL_TRAIN_PATH)
    data_gen.configure(DataGeneratorConfig.NORM_AND_TRANSFORM)

    train_generator, test_generator, validation_generator = data_gen.get(
        train_path=SMALL_TRAIN_PATH,
        test_path=TEST_PATH,
        validate_path=TEST_PATH)

    init = time.time()
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=(int(
                                      400 * 1881 / 1881 // batch_size) + 1),
                                  epochs=number_of_epoch,
                                  validation_data=validation_generator,
                                  validation_steps=807 // batch_size)

    end = time.time()
    logger.info('[Training] Done in ' + str(end - init) + ' secs.\n')

    init = time.time()
    scores = model.evaluate_generator(test_generator, steps=807 // 64)
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

    # list all data in history
    if plot_history:
        do_plotting(history=history, history2=None, cm=cm)

    logger.info(
        'Param to optimize [Accuracy] is: {}'.format(evaluator.accuracy))
    return evaluator.accuracy


def main_with_random_search():
    bounds = [
        {'name': 'batch_size', 'type': 'discrete',
         'domain': (16, 32, 64)},
        {'name': 'fc1_size', 'type': 'discrete',
         'domain': (512, 1024, 2048, 4096)},
        {'name': 'fc2_size', 'type': 'discrete',
         'domain': (128, 256, 512, 1024)}]

    optimizer = BayesianOptimization(f=function_to_optimize, domain=bounds)
    optimizer.run_optimization(max_iter=10)
    logger.info('optimized parameters: {}'.format(optimizer.x_opt))
    logger.info('optimized accuracy: {}'.format(optimizer.fx_opt))


if __name__ == '__main__':
    try:
        os.makedirs('../results/session4')
    except OSError as expected:
        # Expected when the folder already exists
        pass
    logger.info('Start')
    logging.debug('Running as PID: {}'.format(os.getpid()))
    init = time.time()
    # main()
    main_with_random_search()
    end = time.time()
    logger.info('Everything done in ' + str(end - init) + ' secs.\n')
    logger.info('End')
