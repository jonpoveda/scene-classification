import os
import time

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

    # FIXME: improve this loop
    for i in range(len(test_images)):
        filename = test_images[i]
        filename_path = os.path.join(DATA_PATH, filename)

        # Do not mind of labels
        des, _ = descriptor.extract_from([filename_path])
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
    train_images, test_images, train_labels, test_labels = database.get_data()

    # create the SIFT detector object
    feature_extractor = SIFT(number_of_features=100)

    # If descriptors are already computed load them
    if database.data_exists():
        print('Loading descriptors...')
        descriptors, labels = database.get_descriptors()
    else:
        print('Computing descriptors...')
        descriptors, labels = feature_extractor.extract_from(train_images,
                                                             train_labels)
        database.save_descriptors(descriptors, labels)

    # Train a k-nn classifier
    classifier = KNN(n_neighbours=5)
    classifier.train(descriptors, labels)

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
