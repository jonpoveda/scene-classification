import cPickle
import os
import time

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from source import DATA_PATH


def load_data(data_path):
    ## type: (str) -> tuple(list, list, list, list)
    """ Read the train and test files

    :rtype: tuple(list, list, list, list)
    :type data_path: str
    :param data_path: where to get the .dat files
    :return: lists containing the train and test images and their labels
    """
    with open(os.path.join(data_path,
                           'train_images_filenames.dat'), 'r') as file_train, \
        open(os.path.join(data_path,
                          'test_images_filenames.dat'), 'r') as file_test, \
        open(os.path.join(data_path,
                          'train_labels.dat'), 'r') as file_train_labels, \
        open(os.path.join(data_path,
                          'train_labels.dat'), 'r') as file_test_labels:
        train_images = cPickle.load(file_train)
        test_images = cPickle.load(file_test)
        train_labels = cPickle.load(file_train_labels)
        test_labels = cPickle.load(file_test_labels)

    print('Loaded {} training images filenames with classes {}'.
          format(len(train_images), set(train_labels)))
    print('Loaded {} testing images filenames with classes {}'.
          format(len(test_images), set(test_labels)))

    return train_images, test_images, train_labels, test_labels


def compute_descriptors(SIFT_detector, train_images, train_labels):
    ## type: (list, list) -> (np.array, np.array)
    """ Compute descriptors using SIFT

    Read the just 30 train images per class.
    Extract SIFT keypoints and descriptors.
    Store descriptors in a python list of numpy arrays.

    :rtype: tuple(list, list)
    :type train_images: list
    :type train_labels: list
    :param train_images: list of images
    :param train_labels: list of labels of the given images
    :return: descriptors and labels
    """
    train_descriptors = []
    train_label_per_descriptor = []

    for filename, train_label in zip(train_images, train_labels):
        filename_path = os.path.join(DATA_PATH, filename)
        if train_label_per_descriptor.count(train_label) < 30:
            print('Reading image ' + filename)
            ima = cv2.imread(filename_path)
            gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            kpt, des = SIFT_detector.detectAndCompute(gray, None)
            train_descriptors.append(des)
            train_label_per_descriptor.append(train_label)
            print(str(len(kpt)) + ' extracted keypoints and descriptors')

    # Transform everything to numpy arrays

    descriptors = train_descriptors[0]
    labels = np.array(
        [train_label_per_descriptor[0]] * train_descriptors[0].shape[0])

    for i in range(1, len(train_descriptors)):
        descriptors = np.vstack((descriptors, train_descriptors[i]))
        labels = np.hstack((labels, np.array(
            [train_label_per_descriptor[i]] * train_descriptors[i].shape[0])))

    return descriptors, labels


def save_descriptors(D, L):
    with open(os.path.join(DATA_PATH,
                           'descriptors.dat'), 'w') as descriptors_file, \
        open(os.path.join(DATA_PATH,
                          'labels.dat'), 'w') as labels_file:
        cPickle.dump(D, descriptors_file)
        cPickle.dump(L, labels_file)


def load_descriptors():
    with open(os.path.join(DATA_PATH,
                           'descriptors.dat'), 'r') as descriptors_file, \
        open(os.path.join(DATA_PATH,
                          'labels.dat'), 'r') as labels_file:
        D = cPickle.load(descriptors_file)
        L = cPickle.load(labels_file)
    return D, L


def main():
    # read the train and test files

    train_images, test_images, train_labels, test_labels = load_data(DATA_PATH)

    # create the SIFT detector object

    SIFT_detector = cv2.SIFT(nfeatures=100)

    # If descriptors are already computed load them
    if not os.path.isfile('descriptors.dat') or \
        not os.path.isfile('labels.dat'):

        print('Computing descriptors...')
        # read the just 30 train images per class
        # extract SIFT keypoints and descriptors
        # store descriptors in a python list of numpy arrays
        # Transform everything to numpy arrays
        D, L = compute_descriptors(SIFT_detector, train_images, train_labels)
        save_descriptors(D, L)
    else:
        print('Loading descriptors...')
        D, L = load_descriptors()

    # Train a k-nn classifier

    print('Training the knn classifier...')
    my_knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    my_knn.fit(D, L)
    print('Done!')

    # get all the test- data and predict their labels

    num_test_images = 0
    num_correct = 0
    for i in range(len(test_images)):
        filename = test_images[i]
        filename_path = os.path.join(DATA_PATH, filename)
        ima = cv2.imread(filename_path)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = SIFT_detector.detectAndCompute(gray, None)
        predictions = my_knn.predict(des)
        values, counts = np.unique(predictions, return_counts=True)
        predicted_class = values[np.argmax(counts)]
        print('image ' + filename + ' was from class ' +
              test_labels[i] + ' and was predicted ' + predicted_class)
        num_test_images += 1
        if predicted_class == test_labels[i]:
            num_correct += 1

    print('Final accuracy: ' + str(num_correct * 100.0 / num_test_images))


    ## 30.48% in 302 secs.


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('Done in ' + str(end - start) + ' secs.')
