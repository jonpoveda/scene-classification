import os
import time

from keras import optimizers
from keras.layers import Dense, Input, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from matplotlib import pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV


from evaluator import Evaluator

from data_generator import DataGenerator
from data_generator_config import DataGeneratorConfig


class conv_neural_network(object):
    class LAYERS(object):
        FIRST = 'pool1'
        SECOND = 'fc1'
        THIRD = 'fc2'
        LAST = 'fc3'
        LABELS = 'fc4'

    def __init__(self, logger,
                 img_size=256, batch_size=16,
                 dataset_dir='/home/datasets/scenes/MIT_split',
                 model_fname='my_first_mlp.h5'):
        # initialyze model
        self.IMG_SIZE = img_size
        self.BATCH_SIZE = batch_size
        self.DATASET_DIR = dataset_dir
        self.MODEL_FNAME = model_fname
        self.logger = logger
        self.logger.info( 'Creating object\n')
        
        img_width, img_height = self.IMG_SIZE, self.IMG_SIZE
        
        #create data generator obects
        self.data_gen_test = DataGenerator(img_width, img_height, self.BATCH_SIZE,
						 self.DATASET_DIR + '/train')
        self.data_gen_test.configure(DataGeneratorConfig.DEFAULT)

        self.test_generator = self.data_gen_test.get_single(
			path=self.DATASET_DIR + '/testCNN')
			
        self.validation_generator = self.data_gen_test.get_single(
			path=self.DATASET_DIR + '/validationCNN')
			
        self.data_gen_train = DataGenerator(img_width, img_height, self.BATCH_SIZE,
						 self.DATASET_DIR + '/train')
        self.data_gen_train.configure(DataGeneratorConfig.CONFIG1)

        self.train_generator = self.data_gen_train.get_single(
			path=self.DATASET_DIR + '/train')
		
		

        if not os.path.exists(self.DATASET_DIR):
            self.logger.info(
                       'ERROR: dataset directory ' + self.DATASET_DIR + ' do not exists!\n')

    def build_CNN_model(self):
        # Build CNN model
        init = time.time()
        self.logger.info( 'Building MLP model...\n')

        # Build the CNN model
        # Input layers
        main_input = Input(shape=(self.IMG_SIZE, self.IMG_SIZE, 3),
                           dtype='float32', name='main_input')


        x = Conv2D(32, (3, 3),activation='relu')(main_input)
        x = Conv2D(32, (3, 3),activation='relu')(x)
        x = MaxPooling2D(pool_size=(4, 4), padding='valid', name='pool')(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu', name='fc1')(x)
        x = Dense(128, activation='relu', name='fc2')(x)
        x = Dropout(0.5)(x)

        main_output = Dense(units=8, activation='softmax', name='predictions')(
            x)

			
		#Select the optimizer:
        opt = optimizers.Adadelta(lr=0.1)
		
        # Compile the model
        self.model = Model(inputs=main_input, outputs=main_output)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])

        print(self.model.summary())
        self.logger.info(self.model.summary())

        plot_model(self.model, to_file='results/session5/CNN.png', show_shapes=True,
                   show_layer_names=True)

        self.logger.info( 'Done!\n')

        end = time.time()
        self.logger.info( 'Done in ' + str(end - init) + ' secs.\n')

    def train_CNN_model(self, n_epochs ):
        # train the CNN model
        init = time.time()

        if os.path.exists(self.MODEL_FNAME):
            self.logger.info(
                       'WARNING: model file ' + self.MODEL_FNAME + ' exists and will be overwritten!\n')

        self.logger.info( 'Start training...\n')


        self.history = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=1881 // self.BATCH_SIZE*10,
            epochs=n_epochs,
            validation_data=self.validation_generator,
            validation_steps=807 // self.BATCH_SIZE)

        self.logger.info( 'Done!\n')
        self.logger.info(
                   'Saving the model into ' + self.MODEL_FNAME + ' \n')
        self.model.save_weights(
            self.MODEL_FNAME)  # always save your weights after training or during training
        self.logger.info('Done!\n')

        end = time.time()
        self.logger.info( 'Done in ' + str(end - init) + ' secs.\n')

    def load_CNN_model(self):
        # load a CNN model
        init = time.time()

        if not os.path.exists(self.MODEL_FNAME):
            self.logger.info(
                       'Error: model file ' + self.MODEL_FNAME + ' exists and will be overwritten!\n')

        self.logger.info(
                   'Loading the model from ' + self.MODEL_FNAME + ' \n')
        self.model.load_weights(
            self.MODEL_FNAME)  # always save your weights after training or during training
        self.logger.info('Done!\n')

        end = time.time()
        self.logger.info('Done in ' + str(end - init) + ' secs.\n')


    def plot_history(self):

        # summarize history for accuracy
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('accuracy.jpg')
        plt.close()

        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('loss.jpg')

    def plot_results(self):
        # plot classification results

        self.logger.info('Getting classification results...\n')
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
                   'Evaluator \nAcc (model)\nAccuracy: {} \nPrecision: {} \nRecall: {} \nFscore: {}'.
                   format(scores[1], evaluator.accuracy, evaluator.precision,
                          evaluator.recall,
                          evaluator.fscore) + '\n')
        cm = evaluator.confusion_matrix()

        # Plot the confusion matrix on test data
        self.logger.info( 'Confusion matrix:\n')
        self.logger.info( cm)
        print(cm)

        plt.matshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        plt.savefig('cm.jpg')
        self.logger.info(
                   'Final accuracy: ' + str(evaluator.accuracy) + '\n')

        end = time.time()
        self.logger.info( 'Done in ' + str(end - init) + ' secs.\n')
		
		
    def cross_validate(self):
        # cross validate the MLP model
        init = time.time()


        end = time.time()
        self.logger.info( 'Done in ' + str(end - init) + ' secs.\n')


