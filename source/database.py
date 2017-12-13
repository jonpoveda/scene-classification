import cPickle
import os

from typing import List


class Database(object):
    """ Implements a directory-based database """

    def __init__(self, path):
        # type: (str) -> None
        """ Define a database with an specific folder to load and save data """
        self.path = path

        self.generated_descriptors_path = os.path.join(
            self.path, 'generated', 'descriptors.dat')

        self.generated_labels_path = os.path.join(
            self.path, 'generated', 'labels.dat')

        self.train_images_path = os.path.join(
            self.path, 'train_images_filenames.dat')

        self.test_images_path = os.path.join(
            self.path, 'test_images_filenames.dat')

        self.train_labels_path = os.path.join(self.path, 'train_labels.dat')
        self.test_labels_path = os.path.join(self.path, 'test_labels.dat')

    def get_data(self):
        # type: (str) -> (List, List, List, List)
        """ Read the train and test files

        :return: lists containing the train and test images and their labels
        """
        with open(self.train_images_path, 'r') as file_train, \
            open(self.test_images_path, 'r') as file_test, \
            open(self.train_labels_path, 'r') as file_train_labels, \
            open(self.test_labels_path, 'r') as file_test_labels:
            train_images = cPickle.load(file_train)
            test_images = cPickle.load(file_test)
            train_labels = cPickle.load(file_train_labels)
            test_labels = cPickle.load(file_test_labels)

        print('Loaded {} training images filenames with classes {}'.
              format(len(train_images), set(train_labels)))
        print('Loaded {} testing images filenames with classes {}'.
              format(len(test_images), set(test_labels)))

        return train_images, test_images, train_labels, test_labels

    def save_descriptors(self, descriptors, labels):
        # type: (str, str) -> None
        with open(self.generated_descriptors_path, 'w') as descriptors_file, \
            open(self.generated_labels_path, 'w') as labels_file:
            cPickle.dump(descriptors, descriptors_file)
            cPickle.dump(labels, labels_file)

    def get_descriptors(self):
        # type: (None) -> (List, List)
        with open(self.generated_descriptors_path, 'r') as descriptors_file, \
            open(self.generated_labels_path, 'r') as labels_file:
            descriptors = cPickle.load(descriptors_file)
            labels = cPickle.load(labels_file)
        return descriptors, labels

    def data_exists(self):
        """ Checks if there are descriptors """
        # If descriptors are already computed load them
        if os.path.isfile(self.generated_descriptors_path) and \
            os.path.isfile(self.generated_labels_path):
            return True
        return False
