import glob

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from scipy import misc

from data_generator_config import DataGeneratorConfig


class DataGenerator(object):
    def __init__(self, img_width, img_height, batch_size, train_path):
        """ Path used for normalizing the train set afterwards """
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.train_path = train_path
        self.data_generator = ImageDataGenerator(
            **DataGeneratorConfig.NORMALISE)

    def configure(self, config):  # type: (dict) -> None
        """ Load a DataGeneratorConfig into the DataGenerator.

        If not done it uses the DataGeneratorConfig.DEFAULT
        """
        self.data_generator = ImageDataGenerator(**config)
        self._fit()

    def _fit(self):
        """ Fits the datagenerator if needed """
        paths = glob.glob(self.train_path + '/*/*.jpg')
        if not paths:
            raise ValueError('No images found in {}'.format(self.train_path))
        number_of_images = len(paths)
        print('Got {} images in {} for pre-processing'.format(
            self.train_path, number_of_images))

        example_image = misc.imread(paths[0])
        image_size = example_image.shape

        data = np.zeros(
            (number_of_images, image_size[0], image_size[1], image_size[2]))
        for (i, path) in enumerate(paths):
            image = misc.imread(path)
            data[i, :, :, :] = image

        self.data_generator.fit(data)

    def get(self, train_path, test_path, validate_path):
        """ Get datasets generators given a data generator

        :return (train, test, validation) generators
        """
        train_generator = self.data_generator.flow_from_directory(
            train_path,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical')

        test_generator = self.data_generator.flow_from_directory(
            test_path,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False)

        validation_generator = self.data_generator.flow_from_directory(
            validate_path,
            target_size=(self.img_width,
                         self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical')
        return train_generator, test_generator, validation_generator

    def get_single(self, path, shuffle=True):
        """ Get dataset generator given a data generator

        :return single generators
        """
        generator = self.data_generator.flow_from_directory(
            path,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=shuffle)

        return generator
