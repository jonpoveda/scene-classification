import multiprocessing
from multiprocessing import Pool
import time

from matplotlib import pyplot as plt
import numpy as np
from typing import List

from classifier import ClassifierFactory
from database import Database
from evaluator import Evaluator
from feature_extractor import SIFT
from source import DATA_PATH


def main(classifier_type=ClassifierFactory.KNN, n_threads=1,
         **classifier_kwargs):
    # FIXME: remove this globals
    global feature_extractor
    global classifier

    # Read the train and test files
    from database import Database
    database = Database(DATA_PATH)
    train_images, test_images, train_labels, test_labels = database.get_data()

    # Create the SIFT detector object
    feature_extractor = SIFT(number_of_features=100)

    # Load or compute descriptors for training
    descriptors, labels = load_in_memory(database, 'train',
                                         train_images, train_labels)

    # Select classification model
    classifier = ClassifierFactory.build(classifier_type, **classifier_kwargs)

    # Train a classifier with train dataset
    print('Trainning model...')
    classifier.train(descriptors, labels)

    # Load or compute descriptors for testing
    descriptors, labels = load_in_memory(database, 'test',
                                         test_images, test_labels)

    # FIXME: do something with descriptors and labels
    # Assess classifier with test dataset
    print('Assessing images...')
    if n_threads == 1:
        predicted_class = predict_images(test_images, test_labels)
    else:
        predicted_class = predict_images_pool(test_images, n_threads)

    # Evaluate performance metrics
    num_test_images = 0
    num_correct = 0

    for i in range(len(test_images)):
        print('image {} was from class {} and was predicted {}'.format(
            test_images[i], test_labels[i], predicted_class[i]))
        num_test_images += 1
        if predicted_class[i] == test_labels[i]:
            num_correct += 1

    print('Final accuracy: {}'.format(num_correct * 100.0 / num_test_images))

    evaluator = Evaluator(test_labels, predicted_class)

    print('Evaluator \nAccuracy: {} \nPrecision: {} \nRecall: {} \nFscore: {}'.
          format(evaluator.accuracy, evaluator.precision, evaluator.recall,
                 evaluator.fscore))

    cm = evaluator.confusion_matrix()

    # Plot the confusion matrix on test data
    print('Confusion matrix:')
    print(cm)
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def predict_images_pool(test_images, n_threads=0):
    """ Predict images using a pool of threads

    Use the number of threads specified in ``n_threads`` in case it is not
    zero. Otherwise detect how many cores are and set it to this number.
    """
    if n_threads == 0:
        n_threads = multiprocessing.cpu_count()
        print('Detected {0} number of CPUs, running {0} number of threads'.
              format(n_threads))
    pool = Pool(processes=n_threads)
    predicted_class = pool.map(predict_image, test_images)
    return predicted_class


def predict_image(image):
    # def predict_image(image, feature_extractor, classifier):
    # FIXME: remove this globals
    global feature_extractor
    global classifier
    test_descriptor = feature_extractor.extract_pool(image)
    print('{} extracted keypoints and descriptors'.format(
        len(test_descriptor)))
    predictions = classifier.predict(test_descriptor)
    values, counts = np.unique(predictions, return_counts=True)
    predicted_class = values[np.argmax(counts)]
    return predicted_class


def predict_images(test_images, test_labels):
    # FIXME: remove this globals
    global feature_extractor
    global classifier

    prediction_list = list()
    for i in range(len(test_images)):
        test_descriptor, _ = \
            feature_extractor.extract(test_images[i], test_labels[i])
        predictions = classifier.predict(test_descriptor)

        is_a_match, predicted_class = \
            assess_a_prediction(predictions, test_images[i], test_labels[i])
        prediction_list.append(predicted_class)
    return prediction_list


def assess_a_prediction(predictions_per_descriptor, test_image, test_label):
    # FIXME: test_image is only for printing, remove it
    values, counts = np.unique(predictions_per_descriptor, return_counts=True)
    predicted_class = values[np.argmax(counts)]
    return predicted_class == test_label, predicted_class


def load_in_memory(database, name, images, labels):
    """ Loads in memory the available descriptors.

    It computes them if they do not exists yet.
    """
    # type: (Database, str, List, List) -> (List, List)
    if database.data_exists(name):
        print('Loading descriptors...')
        descriptors, labels = database.get_descriptors(name)
    else:
        print('Computing descriptors...')
        descriptors, labels = feature_extractor.extract_from(images, labels)
        database.save_descriptors(descriptors, labels, name)
    return descriptors, labels


if __name__ == '__main__':
    start = time.time()
    # Using RANDOM FOREST
    # main(classifier_type=ClassifierFactory.RANDOM_FOREST, threading='multi')

    # Using KNN
    main(classifier_type=ClassifierFactory.KNN, n_threads=0, n_neighbours=5)
    # original  : 30.48% in 302 secs
    # no pool   : 36.31% in 238 secs
    # 4-pool    : 36.31% in 129 secs

    end = time.time()
    print('Done in {} secs.'.format(end - start))
