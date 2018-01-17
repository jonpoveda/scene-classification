import getpass
import logging
import os
import time

from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model as plot
import matplotlib.pyplot as plt

from source import TEST_PATH
from source import TRAIN_PATH

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
running_in_server = False
if running_in_server:
    batch_size = 32
    number_of_epoch = 20
else:
    batch_size = 5
    number_of_epoch = 1


def colour_channel_swapping(x, dim_ordering):
    """ Colour channel swapping from RGB to BGR """
    if dim_ordering == 'th':
        x = x[::-1, :, :]
    elif dim_ordering == 'tf':
        x = x[:, :, ::-1]
    else:
        raise Exception("Ordering not allowed. Use one of these: 'tf' or 'th'")
    return x


def mean_subtraction(x, dim_ordering):
    """ Mean subtraction for VGG16 model re-training

    :param x matrix with the first index following the BGR convention
    :param dim_ordering th for theano and tf for tensorflow
    """
    if dim_ordering == 'th':
        # Zero-center by mean pixel
        x[0, :, :] -= 103.939
        x[1, :, :] -= 116.779
        x[2, :, :] -= 123.68
    elif dim_ordering == 'tf':
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    else:
        raise Exception("Ordering not allowed. Use one of these: 'tf' or 'th'")
    return x


def preprocess_input(x, dim_ordering='default'):
    """ Colour space swapping and data normalization

    From RGB to BGR swapping.
    Apply a mean subtraction (the mean comes from VGG16 model
    """
    # Get type of ordering (Tensorflow or Theanos)
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    x = colour_channel_swapping(x, dim_ordering)
    x = mean_subtraction(x, dim_ordering)
    return x


def get_base_model():
    """ create the base pre-trained model """
    base_model = VGG16(weights='imagenet')
    plot(base_model,
         to_file='../results/session4/modelVGG16a.png',
         show_shapes=True,
         show_layer_names=True)
    return base_model


def modify_model_for_eight_classes(base_model):
    """ Modify to classify 8 classes.

    Get the XXX layer and add a FC to classify scenes (8-class classifier)
    """
    x = base_model.layers[-2].output
    x = Dense(8, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=x)
    plot(model,
         to_file='../results/session4/modelVGG16b.png',
         show_shapes=True,
         show_layer_names=True)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model


def do_plotting(history):
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
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('../results/session4/loss.jpg')


def get_generators(data_generator, train_path, test_path, validate_path):
    """ Get datasets generators given a data generator

    :return (train, test, validation) generators
    """

    train_generator = data_generator.flow_from_directory(
        train_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    test_generator = data_generator.flow_from_directory(
        test_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = data_generator.flow_from_directory(
        validate_path,
        target_size=(img_width,
                     img_height),
        batch_size=batch_size,
        class_mode='categorical')
    return train_generator, test_generator, validation_generator


def main():
    base_model = get_base_model()
    model = modify_model_for_eight_classes(base_model)
    for layer in model.layers:
        logger.debug(layer.name, layer.trainable)

    # Get train, validation and test dataset
    # preprocessing_function=preprocess_input,
    data_generator = ImageDataGenerator(featurewise_center=False,
                                        samplewise_center=False,
                                        featurewise_std_normalization=False,
                                        samplewise_std_normalization=False,
                                        preprocessing_function=preprocess_input,
                                        rotation_range=0.,
                                        width_shift_range=0.,
                                        height_shift_range=0.,
                                        shear_range=0.,
                                        zoom_range=0.,
                                        channel_shift_range=0.,
                                        fill_mode='nearest',
                                        cval=0.,
                                        horizontal_flip=False,
                                        vertical_flip=False,
                                        rescale=None)

    if running_in_server:
        train_generator, test_generator, validation_generator = get_generators(
            data_generator,
            train_path=TRAIN_PATH,
            test_path=TEST_PATH,
            validate_path=TEST_PATH)

        init = time.time()
        history = model.fit_generator(train_generator,
                                      steps_per_epoch=(int(
                                          400 * 1881 / 1881 // batch_size) + 1),
                                      epochs=number_of_epoch,
                                      validation_data=validation_generator,
                                      validation_steps=807)
        end = time.time()
        logger.info('[Training] Done in ' + str(end - init) + ' secs.\n')

        init = time.time()
        result = model.evaluate_generator(test_generator, val_samples=807)
        end = time.time()
        logger.info('[Evaluation] Done in ' + str(end - init) + ' secs.\n')


    else:
        logger.info('Running in a laptop! Toy mode active')
        train_generator, test_generator, validation_generator = get_generators(
            data_generator,
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
        result = model.evaluate_generator(test_generator, val_samples=10)
        end = time.time()
        logger.info('[Evaluation] Done in ' + str(end - init) + ' secs.\n')

    logger.debug(result)

    # list all data in history
    if plot_history:
        do_plotting(history)


if __name__ == '__main__':
    try:
        os.makedirs('../results/session4')
    except OSError as expected:
        # Expected when the folder already exists
        pass
    logger.info('Start')
    main()
    logger.info('End')
