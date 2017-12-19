import multiprocessing
from multiprocessing import Pool
import time

from matplotlib import pyplot as plt
import numpy as np

from classifier import ClassifierFactory
from database import Database
from evaluator import Evaluator
from feature_extractor import ColourHistogram, SIFT
from source import DATA_PATH


def main(feature_extractor, classifier, n_threads=1):
    do_plotting = False

    # Read the train and test files
    database = Database(DATA_PATH)
    train_images, test_images, train_labels, test_labels = database.get_data()

    # Load or compute descriptors for training
    descriptors, labels = database.load_in_memory('train', feature_extractor,
                                                  train_images, train_labels)
    # Train a classifier with train dataset
    print('Trainning model...')
    classifier.train(descriptors, labels)

    # BUG: the test descriptors cannot be saved the same way as the train ones
    # cause these have to be checked by-image not by-blob. A try is using the
    # function predict_images_pool_2 but is not finished and needs a change in
    # the Database implenentation. Idea: always saving one descriptor file per
    # image and then group them in the case of training and keep them
    # separated for prediction

    # Load or compute descriptors for testing
    # descriptors, labels = load_in_memory(database, 'test',
    #                                      test_images, test_labels)

    # FIXME: do something with descriptors and labels
    # Assess classifier with test dataset
    print('Testing classifier...')
    if n_threads == 1:
        predicted_class = predict_images(test_images, test_labels)
    else:
        print('Predicting test images')
        predicted_class = predict_images_pool(test_images, n_threads)
        # predicted_class = predict_images_pool_2(descriptors.tolist(), n_threads)

    # Evaluate performance metrics
    evaluator = Evaluator(test_labels, predicted_class)

    print('Evaluator \nAccuracy: {} \nPrecision: {} \nRecall: {} \nFscore: {}'.
          format(evaluator.accuracy, evaluator.precision, evaluator.recall,
                 evaluator.fscore))

    cm = evaluator.confusion_matrix()

    # Plot the confusion matrix on test data
    print('Confusion matrix:')
    print(cm)
    if do_plotting:
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


# NOTE: not in use
def predict_images_pool_2(test_descriptors, n_threads=0):
    """ Predict images using a pool of threads

    Use the number of threads specified in ``n_threads`` in case it is not
    zero. Otherwise detect how many cores are and set it to this number.
    """
    if n_threads == 0:
        n_threads = multiprocessing.cpu_count()
        print('Detected {0} number of CPUs, running {0} number of threads'.
              format(n_threads))
    pool = Pool(processes=n_threads)
    predicted_class = pool.map(predict_image_2, test_descriptors)
    return predicted_class


# NOTE: not in use
def predict_image_2(test_descriptor):
    # FIXME: remove this globals
    global feature_extractor
    global classifier
    predictions = classifier.predict(test_descriptor)
    values, counts = np.unique(predictions, return_counts=True)
    predicted_class = values[np.argmax(counts)]
    return predicted_class


# FIXME: remove this method and use ``predict_image_2`` instead
def predict_image(image):
    global feature_extractor
    global classifier
    test_descriptor = feature_extractor.extract_pool(image)
    # print('{} extracted keypoints and descriptors'.format(
    #     len(test_descriptor)))
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


if __name__ == '__main__':
    # FIXME: remove this globals when well implemented
    global feature_extractor
    global classifier

    # Create the SIFT detector object
    feature_extractor = SIFT(number_of_features=200)
    feature_extractor = ColourHistogram(bins=32)

    # Select classification model
    # classifier = ClassifierFactory.build(ClassifierFactory.KNN, n_neighbors=5)
    # classifier = ClassifierFactory.build(ClassifierFactory.RANDOM_FOREST)
    # classifier = ClassifierFactory.build(ClassifierFactory.GAUSSIAN_BAYES)
    # classifier = ClassifierFactory.build(ClassifierFactory.BERNOULLI_BAYES)
    # classifier = ClassifierFactory.build(ClassifierFactory.SVM)
    # classifier = ClassifierFactory.build(ClassifierFactory.LOGISTIC_REGRESSION)

    classifier = ClassifierFactory.build(ClassifierFactory.SVM, cparam=1000.0)

    start = time.time()
    main(feature_extractor, classifier, n_threads=0)
    end = time.time()
    print('Done in {} secs.'.format(end - start))


# Results
# SIFT 100 + 5-NN
#   original  : 30.48% in 302 secs
#   no pool   : 36.31% in 238 secs
#   4-pool    : 36.31% in 129 secs

def plot_cm():
    cm = np.array([[93, 12, 13, 0, 0, 0, 0, 0],
                   [56, 51, 0, 8, 0, 0, 0, 1],
                   [52, 1, 48, 0, 0, 0, 0, 0],
                   [49, 10, 0, 11, 3, 0, 0, 3],
                   [22, 4, 5, 2, 59, 0, 0, 2],
                   [72, 16, 23, 2, 0, 1, 0, 0],
                   [47, 5, 4, 4, 14, 1, 2, 3],
                   [37, 4, 11, 7, 20, 0, 1, 28]])
    cm = np.log10(cm + 1)
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
