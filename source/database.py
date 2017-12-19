import cPickle
import os

from typing import List


class Database(object):
    """ Implements a directory-based database """

    def __init__(self, path):
        # type: (str) -> None
        """ Define a database with an specific folder to load and save data """
        self.path = path
        self.temp_path = os.path.join(self.path, 'tmp')

        try:
            os.makedirs(self.temp_path)
        except OSError as expected:
            pass

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

    def save_descriptors(self, descriptors, labels, dataset_name):
        """ Save all descriptors in one file and all labels in another """
        # type: (str, str, str) -> None
        if not self.data_exists(dataset_name):
            os.makedirs(os.path.join(self.temp_path, dataset_name))

        descriptors_path, labels_path = self.get_paths(dataset_name)

        with open(descriptors_path, 'w') as descriptors_file, \
            open(labels_path, 'w') as labels_file:
            cPickle.dump(descriptors, descriptors_file)
            cPickle.dump(labels, labels_file)

    # NOTE: not in use
    def save_descriptors_as_files(self, names, descriptors, labels,
                                  dataset_name):
        """ Saves each descriptor and label to a different file """
        # type: (List, str, str, str) -> None
        if not self.data_exists(dataset_name):
            os.makedirs(os.path.join(self.temp_path, dataset_name))

        descriptors_path, labels_path = self.get_paths(dataset_name)

        for descriptor, name in zip(descriptors, names):
            des_path = os.path.join(descriptors_path, '{}.des'.format(name))
            try:
                os.makedirs(des_path)
            except IOError as expected:
                pass
            label_path = os.path.join(descriptors_path, name, '.lab')
            # print('Saving {} in {}'.format(name, file_path))
            with open(des_path, 'w') as descriptor_file, \
                open(label_path, 'w') as label_file:
                cPickle.dump(descriptors, descriptor_file)
                cPickle.dump(labels, label_file)

    def get_descriptors(self, dataset_name):
        # type: (str) -> (List, List)
        descriptors_path, labels_path = self.get_paths(dataset_name)

        with open(descriptors_path, 'r') as descriptors_file, \
            open(labels_path, 'r') as labels_file:
            descriptors = cPickle.load(descriptors_file)
            labels = cPickle.load(labels_file)
        return descriptors, labels

    def data_exists(self, dataset_name):
        # type: (str) -> bool
        """ Checks if there are descriptors """
        descriptors_path, labels_path = self.get_paths(dataset_name)

        # If descriptors are already computed load them
        if os.path.isfile(descriptors_path) and \
            os.path.isfile(labels_path):
            return True
        return False

    def get_paths(self, dataset_name):
        """ Returns the paths of the temporal data

        Returns the paths for descriptors and labels given a dataset name.
        """
        return os.path.join(self.temp_path, dataset_name, 'descriptors.dat'), \
               os.path.join(self.temp_path, dataset_name, 'labels.dat')

    # NOTE: replace by calling self.temp_path
    # NOTE: not in use
    def get_paths_2(self, dataset_name):
        """ Returns the paths of the temporal data

        Returns the paths for descriptors and labels given a dataset name.
        """
        return self.temp_path, self.temp_path

    def load_in_memory(self, dataset_name, feature_extractor, images, labels):
        """ Loads in memory the available descriptors.

        It computes them if they do not exists yet.
        """
        # type: (Database, str, List, List) -> (List, List)
        if self.data_exists(dataset_name):
            print('Loading descriptors: {}'.format(dataset_name))
            descriptors, labels = self.get_descriptors(dataset_name)
        else:
            print('Computing descriptors: {}'.format(dataset_name))
            descriptors, labels = feature_extractor.extract_from_a_list(images,
                                                                        labels)
            self.save_descriptors(descriptors, labels, dataset_name)
            # self.save_descriptors_as_files(images, descriptors, labels,
            #                                dataset_name)

        print('Loaded {} descriptors and {} labels'.format(len(descriptors),
                                                           len(labels)))
        return descriptors, labels
