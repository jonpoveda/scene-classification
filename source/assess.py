import os

import cv2
import numpy as np

from classifier import KNN
from descriptor import SIFT
from source import DATA_PATH
from multiprocessing import Pool

class assessImages(object):
    model = None
    def __init__(self):
        # create the SIFT detector object
        self.descriptor = SIFT(number_of_features=100)
        # Train a k-nn classifier
        self.classifier = KNN(n_neighbours=5)
        
    def predict_class(self, filename):
        
        filename_path = os.path.join(DATA_PATH, filename)
        print(filename_path)
        ima = cv2.imread(filename_path)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = self.descriptor.detectAndCompute(gray)
        predictions = self.classifier.predict(des)
        values, counts = np.unique(predictions, return_counts=True)
        predicted_class = values[np.argmax(counts)]
        return predicted_class

    def assess(self, test_images, test_labels):
        # type: (List, BaseClassifier, BaseFeatureExtractor, List) -> (int, int)
        # get all the test data and predict their labels
        num_test_images = 0
        num_correct = 0
    
    #    for i in range(len(test_images)):
    #        filename = test_images[i]
    #        filename_path = os.path.join(DATA_PATH, filename)
    #        ima = cv2.imread(filename_path)
    #        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
    #        kpt, des = descriptor.extract(gray)
    #        predictions = my_knn.predict(des)
    #        values, counts = np.unique(predictions, return_counts=True)
    #        predicted_class = values[np.argmax(counts)]
    #        print(
    #            'image {} '
    #            'was from class {} '
    #            'and was predicted {}'.format(filename,
    #                                          test_labels[i],
    #                                          predicted_class))
    #        num_test_images += 1
    #        if predicted_class == test_labels[i]:
    #            num_correct += 1
    
        pool = Pool(processes=4) 
    
        predicted_class = pool.map(self.predict_class, test_images )
    
    
        for i in range(len(test_images)):
            print('image ' + test_images[i] + ' was from class ' +test_labels[i] + ' and was predicted ' + predicted_class[i])
            num_test_images += 1
            if predicted_class[i] == test_labels[i]:
                num_correct += 1   
    
        return num_correct, num_test_images


    def main(self):
        # read the train and test files
        from database import Database
        database = Database(DATA_PATH)
        train_images, test_images, train_labels, test_labels = database.load_data()
    
        
#        feature_extractor = SIFT(number_of_features=100)
    
        # If descriptors are already computed load them
        if database.data_exists():
            print('Loading descriptors...')
            D, L = database.load_descriptors()
        else:
            print('Computing descriptors...')
            D, L = self.descriptor.generate(train_images, train_labels)
            database.save_descriptors(D, L)
    
        # Train a k-nn classifier
#        classifier = KNN(n_neighbours=5)
        self.classifier.train(D, L)
        # train(descriptors=D, labels=L)
    
        num_correct, num_test_images = self.assess(test_images,
                                              test_labels)
        print('Final accuracy: ' + str(num_correct * 100.0 / num_test_images))
        ## 30.48% in 302 secs.

