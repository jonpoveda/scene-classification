import os
import time

import cv2
import numpy as np
from typing import List

from classifier import BaseClassifier
from classifier import KNN
from descriptor import BaseFeatureExtractor
from descriptor import SIFT
from source import DATA_PATH


def assess(test_images, my_knn, descriptor, test_labels):
    # type: (List, BaseClassifier, BaseFeatureExtractor, List) -> (int, int)
    # get all the test data and predict their labels
    num_test_images = 0
    num_correct = 0

    for i in range(len(test_images)):
        filename = test_images[i]
        filename_path = os.path.join(DATA_PATH, filename)
        ima = cv2.imread(filename_path)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = descriptor.extract(gray)
        predictions = my_knn.predict(des)
        values, counts = np.unique(predictions, return_counts=True)
        predicted_class = values[np.argmax(counts)]
        print(
            'image {} '
            'was from class {} '
            'and was predicted {}'.format(filename,
                                          test_labels[i],
                                          predicted_class))
        num_test_images += 1
        if predicted_class == test_labels[i]:
            num_correct += 1

    return num_correct, num_test_images


def main():
    # read the train and test files
    from database import Database
    database = Database(DATA_PATH)
    train_images, test_images, train_labels, test_labels = database.load_data()

    # create the SIFT detector object
    feature_extractor = SIFT(number_of_features=100)

    # If descriptors are already computed load them
    if database.data_exists():
        print('Loading descriptors...')
        D, L = database.load_descriptors()
    else:
        print('Computing descriptors...')
        D, L = feature_extractor.generate(train_images, train_labels)
        database.save_descriptors(D, L)

    # Train a k-nn classifier
    classifier = KNN(n_neighbours=5)
    classifier.train(D, L)
    # train(descriptors=D, labels=L)

    num_correct, num_test_images = assess(test_images,
                                          classifier,
                                          feature_extractor,
                                          test_labels)
    print('Final accuracy: ' + str(num_correct * 100.0 / num_test_images))
    ## 30.48% in 302 secs.


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('Done in ' + str(end - start) + ' secs.')
