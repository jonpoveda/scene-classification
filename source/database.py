import cPickle
import glob
import os

from typing import List


class Database(object):
    """ Implements a directory-based database """

    def __init__(self, path):
        # type: (str) -> None
        """ Define a database with an specific folder to load and save data """
        self.path = path
        self.base_path = os.path.join(self.path, 'tmp')

        try:
            os.makedirs(self.base_path)
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
        if not os.path.exists(os.path.join(self.base_path, dataset_name)):
            os.makedirs(os.path.join(self.base_path, dataset_name))

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
        if not self.dataset_exists(dataset_name):
            os.makedirs(os.path.join(self.base_path, dataset_name))

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

    def dataset_exists(self, name):
        # type: (str) -> bool
        """ Checks if there are descriptors """
        descriptors_path, labels_path = self.get_paths(name)

        # If descriptors are already computed load them
        if os.path.isfile(descriptors_path) and \
            os.path.isfile(labels_path):
            return True
        return False

    def get_paths(self, dataset_name):
        """ Returns the paths of the temporal data

        Returns the paths for descriptors and labels given a dataset name.
        """
        return os.path.join(self.base_path, dataset_name, 'descriptors.dat'), \
               os.path.join(self.base_path, dataset_name, 'labels.dat')

    # NOTE: replace by calling self.temp_path
    # NOTE: not in use
    def get_paths_2(self, dataset_name):
        """ Returns the paths of the temporal data

        Returns the paths for descriptors and labels given a dataset name.
        """
        return self.base_path, self.base_path

    def load_in_memory(self, dataset_name, feature_extractor, images, labels):
        """ Loads in memory the available descriptors.

        It computes them if they do not exists yet.
        """
        # type: (Database, str, List, List) -> (List, List)
        if self.dataset_exists(dataset_name):
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


class DatabaseFiles(Database):
    def __init__(self, path):
        # type: (str) -> None
        """ Define a database with an specific folder to load and save data

        It contains datasets where to save or load descriptors
        """
        super(DatabaseFiles, self).__init__(path)
        self.datasets = dict()

    def create_dataset(self, name):
        self.datasets.update({name: Dataset(self.base_path, name)})

    def get_dataset(self, name):
        return self.datasets.get(name)


class FilesMissing(Exception):
    pass


class Dataset(object):
    def __init__(self, basepath, name):
        # type: (str, str) -> None
        self.path = os.path.join(basepath, name)
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def exists(self):
        # type (str) -> bool
        """ True if it contains descriptors, false if it is empty """
        list_of_descriptors = glob.glob('{}/*.des'.format(self.path))
        list_of_labels = glob.glob('{}/*.lab'.format(self.path))
        if list_of_descriptors.__len__() != list_of_labels.__len__():
            raise FilesMissing(
                'It should be the same number of descriptors than labels')
        return bool(list_of_descriptors)

    def load_descriptor(self, image_relative_path):
        """ Loads a descriptor and a label from an image in memory """
        absolute_path = os.path.join(self.path, image_relative_path)
        filename_without_extension = absolute_path.rsplit('.', 1)[0]
        print filename_without_extension
        die()
        descriptor_path = filename_without_extension + '.des'
        label_path = filename_without_extension + '.lab'

        with open(descriptor_path, 'r') as descriptor_file, \
            open(label_path, 'r') as label_file:
            descriptor = cPickle.load(descriptor_file)
            label = cPickle.load(label_file)

        return descriptor, label

    def save_descriptor(self, image_relative_path, descriptor, label):
        # type: (str, List, List) -> None
        """ Save a descriptor and a label from an image to disk

        :param image_relative_path: path of the image (to retrieve the name)
        """
        absolute_path = os.path.join(self.path, image_relative_path)
        filename_without_extension = absolute_path.rsplit('.', 1)[0]
        descriptor_path = filename_without_extension + '.des'
        label_path = filename_without_extension + '.lab'

        try:
            os.makedirs(os.path.dirname(descriptor_path))
        except OSError as expected:
            pass

        with open(descriptor_path, 'w') as descriptor_file, \
            open(label_path, 'w') as label_file:
            cPickle.dump(descriptor, descriptor_file)
            cPickle.dump(label, label_file)

    def load_all(self, image_list):
        """ Loads in memory the available descriptors.

        It computes them if they do not exists yet.
        """
        # type: (str, List, List) -> (List, List)
        descriptors, labels = list(), list()

        print('Loading descriptors from : {}'.format(self.path))
        if not self.exists():
            print('Nothing loaded')
            return (), ()

        for image in image_list:
            descriptor, label = self.load_descriptor(image)
            descriptors.append(descriptor)
            labels.append(label)

        print(
            'Loaded {} descriptors and {} labels'.format(len(descriptors),
                                                         len(labels)))
        return descriptors, labels
