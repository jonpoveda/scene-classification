import cPickle
import os
import time

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from descriptor_factory import SIFT
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


def save_descriptors(descriptors, labels):
    with open(os.path.join(DATA_PATH,
                           'descriptors.dat'), 'w') as descriptors_file, \
        open(os.path.join(DATA_PATH,
                          'labels.dat'), 'w') as labels_file:
        cPickle.dump(descriptors, descriptors_file)
        cPickle.dump(labels, labels_file)


def load_descriptors():
    with open(os.path.join(DATA_PATH,
                           'descriptors.dat'), 'r') as descriptors_file, \
        open(os.path.join(DATA_PATH,
                          'labels.dat'), 'r') as labels_file:
        descriptors = cPickle.load(descriptors_file)
        labels = cPickle.load(labels_file)
    return descriptors, labels


def assess(test_images, my_knn, descriptor_generator, test_labels):
    # get all the test data and predict their labels
    num_test_images = 0
    num_correct = 0

    for i in range(len(test_images)):
        filename = test_images[i]
        filename_path = os.path.join(DATA_PATH, filename)
        ima = cv2.imread(filename_path)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = descriptor_generator.detectAndCompute(gray)
        predictions = my_knn.predict(des)
        values, counts = np.unique(predictions, return_counts=True)
        predicted_class = values[np.argmax(counts)]
        print('image ' + filename + ' was from class ' +
              test_labels[i] + ' and was predicted ' + predicted_class)
        num_test_images += 1
        if predicted_class == test_labels[i]:
            num_correct += 1

    return num_correct, num_test_images


def main():
    # read the train and test files
    train_images, test_images, train_labels, test_labels = load_data(DATA_PATH)

    # create the SIFT detector object
    descriptor_generator = SIFT(number_of_features=100)

    # If descriptors are already computed load them
    if os.path.isfile('descriptors.dat') and os.path.isfile('labels.dat'):
        print('Loading descriptors...')
        D, L = load_descriptors()
    else:
        print('Computing descriptors...')
        D, L = descriptor_generator.generate(train_images, train_labels)
        save_descriptors(D, L)

    # Train a k-nn classifier
    print('Training the knn classifier...')
    my_knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    my_knn.fit(D, L)
    print('Done!')

    num_correct, num_test_images = assess(test_images,
                                          my_knn,
                                          descriptor_generator,
                                          test_labels)
    print('Final accuracy: ' + str(num_correct * 100.0 / num_test_images))
    ## 30.48% in 302 secs.


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('Done in ' + str(end - start) + ' secs.')
