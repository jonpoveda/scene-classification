from __future__ import print_function

import os

import cv2
import numpy as np
from typing import List, Type

from source import DATA_PATH


class BaseFeatureExtractor(object):
    def extract_from(self, train_images, train_labels=['no_label']):
        """ Compute descriptors given a list of images and labels """
        # type: (List, List) -> Type[NotImplementedError]
        return NotImplementedError

    def _compute(self, image):
        """ Compute an image descriptor """
        # type: np.array -> Type[NotImplementedError]
        return NotImplementedError


class color_hist(BaseFeatureExtractor):
    def __init__(self, bins=10, range=None, weights=None):
        # type: (int) -> None
        # FIXME: remove number_of_features if they are not explicity needed
        self.bins = bins
        self.range = range
        self.weights = weights
        

    def _compute(self, image):
        # type: (np.array) -> List
        """ Extract descriptor from an image """
        luv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        u = luv[:,:,1]
        v = luv[:,:,2]
        u = u.flatten()
        v = v.flatten()
        hist = np.histogram2d(u, v, bins=self.bins, range=self.range, normed=True, weights=self.weights)
        values = hist[0].flatten()
        descriptors = values.reshape(1, -1)
        return descriptors

    def extract(self, filename, label):
        descriptors = list()
        label_per_descriptor = list()
        filename_path = os.path.join(DATA_PATH, filename)

        image = cv2.imread(filename_path)
        descriptor = self._compute(image)
        descriptors.append(descriptor)
        label_per_descriptor.append(label)
        print('{} extracted keypoints and descriptors'.format(
            len(descriptor)))
        # Transform everything to numpy arrays
        descriptors = descriptors[0]
        labels = 0
        return descriptor, labels
    
    def extract_pool(self, filename):
        filename_path = os.path.join(DATA_PATH, filename)

        image = cv2.imread(filename_path)
        descriptors = self._compute(image)


        print('{} extracted keypoints and descriptors'.format(
            len(descriptors)))
        # Transform everything to numpy arrays

        return descriptors

    def extract_from(self, train_images, train_labels=['no_label']):
        # type: (List, List) -> (np.array, np.array)
        """ Compute descriptors using SIFT

        Read the just 30 train images per class.
        Extract SIFT keypoints and descriptors.
        Store descriptors and labels in a python list of numpy arrays.

        Note the labels from the input are expanded to the output in a way that
        each descriptor has its label.

        :param train_images: list of images
        :param train_labels: list of labels of the given images
        :return: descriptors and labels
        """
        descriptors = []
        label_per_descriptor = []

        for filename, train_label in zip(train_images, train_labels):
            filename_path = os.path.join(DATA_PATH, filename)
            if label_per_descriptor.count(train_label) < 30:
                print('Reading image {}'.format(os.path.basename(filename)),
                      end=' ')
                image = cv2.imread(filename_path)
                descriptor = self._compute(image)
                descriptors.append(descriptor)
                label_per_descriptor.append(train_label)
                print('{} extracted keypoints and descriptors'.format(
                    len(descriptor)))
        descriptors_array = descriptors[0]
        labels = np.array(
            [label_per_descriptor[0]])

        for i in range(1, len(descriptors)):
            descriptors_array = np.vstack((descriptors_array, descriptors[i]))
            labels = np.hstack((labels, np.array(
                [label_per_descriptor[i]])))

        return descriptors_array, label_per_descriptor
