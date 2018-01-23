import os
import time

import numpy as np
from keras import optimizers
from keras.layers import Dense, Input, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.models import Model
from keras.utils import plot_model
from matplotlib import pyplot as plt

from data_generator import DataGenerator
from data_generator_config import DataGeneratorConfig
from evaluator import Evaluator


class conv_neural_network(object):
    class LAYERS(object):
        FIRST = 'pool1'
        SECOND = 'fc1'
        THIRD = 'fc2'
        LAST = 'fc3'
        LABELS = 'fc4'

    def __init__(self, logger,
                 input_image_size=256, batch_size=16,
                 dataset_dir='/home/datasets/scenes/MIT_split',
                 model_fname='my_first_mlp.h5'):

        # initialize model
        self.model = None
        self.history = None
        self.image_size = input_image_size
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.model_fname = model_fname
        self.logger = logger
        self.logger.info('Creating object')

        # create data generator objects
        self.data_gen_test = DataGenerator(self.image_size, self.image_size,
                                           self.batch_size,
                                           self.dataset_dir + '/train')
        self.data_gen_test.configure(DataGeneratorConfig.DEFAULT)

        self.test_generator = self.data_gen_test.get_single(
            path=self.dataset_dir + '/testCNN')

        self.validation_generator = self.data_gen_test.get_single(
            path=self.dataset_dir + '/validationCNN')

        self.data_gen_train = DataGenerator(self.image_size, self.image_size,
                                            self.batch_size,
                                            self.dataset_dir + '/train')
        self.data_gen_train.configure(DataGeneratorConfig.CONFIG1)

        self.train_generator = self.data_gen_train.get_single(
            path=self.dataset_dir + '/train')

        if not os.path.exists(self.dataset_dir):
            self.logger.info('ERROR: dataset directory {} do not exists!'.
                             format(self.dataset_dir))

    def build_CNN_model(self):
        # Build CNN model
        init = time.time()
        self.logger.info('Building MLP model...')

        # Build the CNN model
        # Input layers
        main_input = Input(shape=(self.image_size, self.image_size, 3),
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

        # Select the optimizer:
        opt = optimizers.Adadelta(lr=0.1)

        # Compile the model
        self.model = Model(inputs=main_input, outputs=main_output)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])

        print(self.model.summary())
        self.logger.info(self.model.summary())

        plot_model(self.model,
                   to_file='results/session5/CNN.png',
                   show_shapes=True,
                   show_layer_names=True)

        self.logger.info('Done!')

        end = time.time()
        self.logger.info('Done in {} secs.'.format(str(end - init)))

    def train_CNN_model(self, n_epochs):
        # train the CNN model
        init = time.time()

        if os.path.exists(self.model_fname):
            self.logger.info('WARNING: model file {} exists and will be '
                             'overwritten!'.format(self.model_fname))

        self.logger.info('Start training...')

        self.history = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=1881 // self.batch_size * 10,
            epochs=n_epochs,
            validation_data=self.validation_generator,
            validation_steps=807 // self.batch_size)

        self.logger.info('Done!')
        self.logger.info('Saving the model into {}'.format(self.model_fname))
        self.model.save_weights(
            self.model_fname)  # always save your weights after training or during training
        self.logger.info('Done!')

        end = time.time()
        self.logger.info('Done in {} secs.'.format(str(end - init)))

    def load_CNN_model(self):
        # load a CNN model
        init = time.time()

        if not os.path.exists(self.model_fname):
            self.logger.info(
                'Error: model file {} exists and will be overwritten!'.format(
                    self.model_fname))

        self.logger.info(
            'Loading the model from {}'.format(self.model_fname))
        self.model.load_weights(
            self.model_fname)  # always save your weights after training or during training
        self.logger.info('Done!')

        end = time.time()
        self.logger.info('Done in {} secs.'.format(str(end - init)))

    def plot_history(self):

        # summarize history for accuracy
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('results/session5/accuracy.jpg')
        plt.close()

        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('results/session5/loss.jpg')

    def plot_results(self):
        # plot classification results

        self.logger.info('Getting classification results...')
        init = time.time()

        # Get ground truth
        test_labels = self.test_generator.classes

        # Predict test images
        predictions_raw = self.model.predict_generator(self.test_generator)
        predictions = []
        for prediction in predictions_raw:
            predictions.append(np.argmax(prediction))
        # Evaluate results
        evaluator = Evaluator(test_labels, predictions,
                              label_list=list([0, 1, 2, 3, 4, 5, 6, 7]))

        #
        scores = self.model.evaluate_generator(self.test_generator)
        self.logger.info(
            'Evaluator \n'
            'Acc (model) {}\n'
            'Accuracy: {} \n'
            'Precision: {} \n'
            'Recall: {} \n'
            'Fscore: {}'.format(scores[1], evaluator.accuracy,
                                evaluator.precision, evaluator.recall,
                                evaluator.fscore))
        cm = evaluator.confusion_matrix()

        # Plot the confusion matrix on test data
        self.logger.info('Confusion matrix:')
        self.logger.info(cm)

        plt.matshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        plt.savefig('results/session5/cm.jpg')
        self.logger.info(
            'Final accuracy: {}'.format(str(evaluator.accuracy)))

        end = time.time()
        self.logger.info('Done in {} secs.'.format(str(end - init)))

    def cross_validate(self):
        # cross validate the MLP model
        init = time.time()

        end = time.time()
        self.logger.info('Done in {} secs.'.format(str(end - init)))
