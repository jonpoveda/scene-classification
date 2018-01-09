import os
import time

from keras.layers import Dense, Reshape
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from matplotlib import pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from evaluator import Evaluator
from utils import Color, colorprint


class multi_layer_perceptron(object):
    def __init__(self, img_size=32, batch_size=16,
                 dataset_dir='/home/datasets/scenes/MIT_split',
                 model_fname='my_first_mlp.h5'):
        # initialyze model
        self.IMG_SIZE = img_size
        self.BATCH_SIZE = batch_size
        self.DATASET_DIR = dataset_dir
        self.MODEL_FNAME = model_fname
        colorprint(Color.BLUE, 'Creating object\n')

        if not os.path.exists(self.DATASET_DIR):
            colorprint(Color.RED,
                       'ERROR: dataset directory ' + self.DATASET_DIR + ' do not exists!\n')
            quit()

    def build_MLP_model(self):
        # Build MLP model

        init = time.time()

        colorprint(Color.BLUE, 'Building MLP model...\n')

        # Build the Multi Layer Perceptron model
        self.model = Sequential()
        self.model.add(Reshape((self.IMG_SIZE * self.IMG_SIZE * 3,),
                               input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3),
                               name='first'))
        self.model.add(Dense(units=2048, activation='relu', name='second'))
        self.model.add(Dense(units=1024, activation='relu'))
        self.model.add(Dense(units=1024, activation='relu', name='last'))
        self.model.add(Dense(units=8, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])

        print(self.model.summary())

        plot_model(self.model, to_file='modelMLP.png', show_shapes=True,
                   show_layer_names=True)

        colorprint(Color.BLUE, 'Done!\n')

        end = time.time()
        colorprint(Color.BLUE, 'Done in ' + str(end - init) + ' secs.\n')

    def train_MLP_model(self):
        # train the MLP model

        init = time.time()

        if os.path.exists(self.MODEL_FNAME):
            colorprint(Color.YELLOW,
                       'WARNING: model file ' + self.MODEL_FNAME + ' exists and will be overwritten!\n')

        colorprint(Color.BLUE, 'Start training...\n')

        # this is the dataset configuration we will use for training
        # only rescaling
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=True)

        # this is the dataset configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        # this is a generator that will read pictures found in
        # subfolers of 'data/train', and indefinitely generate
        # batches of augmented image data
        train_generator = train_datagen.flow_from_directory(
            self.DATASET_DIR + '/train',  # this is the target directory
            target_size=(self.IMG_SIZE, self.IMG_SIZE),
            # all images will be resized to IMG_SIZExIMG_SIZE
            batch_size=self.BATCH_SIZE,
            classes=['coast', 'forest', 'highway', 'inside_city', 'mountain',
                     'Opencountry', 'street', 'tallbuilding'],
            class_mode='categorical')  # since we use binary_crossentropy loss, we need categorical labels

        # this is a generator that will read pictures found in
        # subfolers of 'data/test', and indefinitely generate
        # batches of augmented image data
        validation_generator = test_datagen.flow_from_directory(
            self.DATASET_DIR + '/test',
            target_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=self.BATCH_SIZE,
            classes=['coast', 'forest', 'highway', 'inside_city', 'mountain',
                     'Opencountry', 'street', 'tallbuilding'],
            class_mode='categorical')

        self.history = self.model.fit_generator(
            train_generator,
            steps_per_epoch=1881 // self.BATCH_SIZE,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=807 // self.BATCH_SIZE)

        colorprint(Color.BLUE, 'Done!\n')
        colorprint(Color.BLUE,
                   'Saving the model into ' + self.MODEL_FNAME + ' \n')
        self.model.save_weights(
            self.MODEL_FNAME)  # always save your weights after training or during training
        colorprint(Color.BLUE, 'Done!\n')

        end = time.time()
        colorprint(Color.BLUE, 'Done in ' + str(end - init) + ' secs.\n')

    def load_MLP_model(self):
        # load a MLP model

        init = time.time()

        if not os.path.exists(self.MODEL_FNAME):
            colorprint(Color.YELLOW,
                       'Error: model file ' + self.MODEL_FNAME + ' exists and will be overwritten!\n')
            quit()

        colorprint(Color.BLUE,
                   'Loading the model from ' + self.MODEL_FNAME + ' \n')
        self.model.load_weights(
            self.MODEL_FNAME)  # always save your weights after training or during training
        colorprint(Color.BLUE, 'Done!\n')

        end = time.time()
        colorprint(Color.BLUE, 'Done in ' + str(end - init) + ' secs.\n')

    def get_layer_output(self, layer='last', image_set='test'):
        # get layer output

        init = time.time()

        colorprint(Color.BLUE, 'Getting layer output...\n')
        model_layer = Model(input=self.model.input,
                            output=self.model.get_layer(layer).output)

        # this is the dataset configuration we will use for testing:
        # only rescaling
        datagen = ImageDataGenerator(rescale=1. / 255)
        # this is a generator that will read pictures found in
        # subfolers of 'data/test', and indefinitely generate
        # batches of augmented image data
        generator = datagen.flow_from_directory(
            self.DATASET_DIR + '/' + image_set,
            target_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=self.BATCH_SIZE,
            classes=['coast', 'forest', 'highway', 'inside_city', 'mountain',
                     'Opencountry', 'street', 'tallbuilding'],
            class_mode='categorical')

        labels = generator.class_indices

        # get the features from images
        features = model_layer.predict_generator(generator)
        colorprint(Color.BLUE, 'Done!\n')

        end = time.time()
        colorprint(Color.BLUE, 'Done in ' + str(end - init) + ' secs.\n')

        return features, labels

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

        colorprint(Color.BLUE, 'Getting classification results...\n')
        init = time.time()

        # this is the dataset configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        # this is a generator that will read pictures found in
        # subfolers of 'data/test', and indefinitely generate
        # batches of augmented image data

        test_generator = test_datagen.flow_from_directory(
            self.DATASET_DIR + '/test',
            target_size=(self.IMG_SIZE, self.IMG_SIZE),
            batch_size=self.BATCH_SIZE,
            classes=['coast', 'forest', 'highway', 'inside_city', 'mountain',
                     'Opencountry', 'street', 'tallbuilding'],
            class_mode='categorical')
        # Get ground truth
        test_labels = test_generator.classes

        # Predict test images
        predictions_raw = self.model.predict_generator(test_generator)
        predictions = []
        for prediction in predictions_raw:
            predictions.append(np.argmax(prediction))

        # Evaluate results
        evaluator = Evaluator(test_labels, predictions)

        colorprint(Color.BLUE,
                   'Evaluator \nAccuracy: {} \nPrecision: {} \nRecall: {} \nFscore: {}'.
                   format(evaluator.accuracy, evaluator.precision,
                          evaluator.recall,
                          evaluator.fscore) + '\n')
        cm = evaluator.confusion_matrix()

        # Plot the confusion matrix on test data
        colorprint(Color.BLUE, 'Confusion matrix:\n')
        colorprint(Color.BLUE, cm)
        print(cm)

        plt.matshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        plt.savefig('cm.jpg')
        colorprint(Color.BLUE,
                   'Final accuracy: ' + str(evaluator.accuracy) + '\n')

        end = time.time()
        colorprint(Color.BLUE, 'Done in ' + str(end - init) + ' secs.\n')

    def cross_validate_SVM(self, features, train_labels):
        """ cross_validate classifier with k stratified folds """
        colorprint(Color.BLUE, 'Cross_validating the SVM classifier...\n')
        init = time.time()
        stdSlr = StandardScaler().fit(features)
        D_scaled = stdSlr.transform(features)
        kfolds = StratifiedKFold(n_splits=5, shuffle=False, random_state=50)

        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10],
                      'gamma': np.linspace(0, 0.01, num=11)}
        grid = GridSearchCV(svm.SVC(), param_grid=parameters, cv=kfolds,
                            scoring='accuracy')
        grid.fit(D_scaled, train_labels)

        end = time.time()
        colorprint(Color.BLUE, 'Done in ' + str(end - init) + ' secs.\n')
        colorprint(Color.BLUE, "Best parameters: %s Accuracy: %0.2f\n" % (
            grid.best_params_, grid.best_score_))

    def train_classifier_SVM(self, features, train_labels):
        # Train an SVM classifier
        colorprint(Color.BLUE, 'Training the SVM classifier...\n')
        init = time.time()
        self.stdSlr = StandardScaler().fit(features)
        D_scaled = self.stdSlr.transform(features)

        # Train an SVM classifier with RBF kernel
        self.clf = svm.SVC(kernel='rbf', C=10, gamma=.002).fit(D_scaled,
                                                               train_labels)

        end = time.time()
        colorprint(Color.BLUE, 'Done in ' + str(end - init) + ' secs.\n')

    def evaluate_performance_SVM(self, features, test_labels, do_plotting):
        # Test the classification accuracy
        colorprint(Color.BLUE, 'Testing the SVM classifier...\n')
        init = time.time()
        test_data = self.stdSlr.transform(features)
        accuracy = 100 * self.clf.score(test_data, test_labels)

        predictions = self.clf.predict(test_data)
        evaluator = Evaluator(test_labels, predictions)

        colorprint(Color.BLUE,
                   'Evaluator \nAccuracy: {} \nPrecision: {} \nRecall: {} \nFscore: {}'.
                   format(evaluator.accuracy, evaluator.precision,
                          evaluator.recall,
                          evaluator.fscore) + '\n')
        cm = evaluator.confusion_matrix()

        # Plot the confusion matrix on test data
        colorprint(Color.BLUE, 'Confusion matrix:\n')
        colorprint(Color.BLUE, cm)
        print(cm)
        if do_plotting:
            plt.matshow(cm)
            plt.title('Confusion matrix')
            plt.colorbar()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.show()
            plt.savefig('cm.jpg')

        end = time.time()
        colorprint(Color.BLUE, 'Done in ' + str(end - init) + ' secs.\n')
        colorprint(Color.BLUE, 'Final accuracy: ' + str(accuracy) + '\n')
