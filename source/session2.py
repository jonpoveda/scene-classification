import cPickle
import time

import cv2
import numpy as np
from sklearn import cluster
from sklearn import svm
from sklearn.preprocessing import StandardScaler

from bag_of_visual_words import BoVW

from database import Database
from feature_extractor import SIFT2, SIFT, denseSIFT
from source import DATA_PATH


def main(feature_extractor):
    do_plotting = True
    
    start = time.time()
    
    # Read the train and test files
    database = Database(DATA_PATH)
    train_images_filenames, test_images_filenames, train_labels, test_labels = \
        database.get_data()
    
    #Create BoVW classifier
    BoVW_classifier = BoVW()
    
    #Extract image descriptors
    D,Train_descriptors = BoVW_classifier.extract_descriptors(feature_extractor,train_images_filenames,train_labels)
    
    #Compute Codebook
    visual_words = BoVW_classifier.compute_codebook( D, Train_descriptors)
    
    #Cross validate classifier
    BoVW_classifier.cross_validate(visual_words,train_labels)
   
    # Train an SVM classifier with RBF kernel
    BoVW_classifier.train_classifier( visual_words, train_labels)

    # get all the test data
    visual_words_test = BoVW_classifier.predict_images(test_images_filenames,feature_extractor)

    # Test the classification accuracy
    BoVW_classifier.evaluate_performance(visual_words_test,test_labels, do_plotting)

    end = time.time()
    print('Everything done in ' + str(end - start) + ' secs.')
    ### 69.02%
    
def cross_validate(feature_extractor):
    # Read the train and test files
    database = Database(DATA_PATH)
    train_images_filenames, test_images_filenames, train_labels, test_labels = \
        database.get_data()
    
    #Create BoVW classifier
    BoVW_classifier = BoVW()
    
    #Extract image descriptors
    D,Train_descriptors = BoVW_classifier.extract_descriptors(feature_extractor,train_images_filenames,train_labels)
    
    #Compute Codebook
    visual_words = BoVW_classifier.compute_codebook( D, Train_descriptors)
    
    #Cross validate classifier
    BoVW_classifier.cross_validate(visual_words,train_labels)


if __name__ == "__main__":
    # FIXME: use 300 n features
    #feature_extractor = SIFT2(number_of_features=30)
    #feature_extractor = SIFT(number_of_features=300)
    feature_extractor = denseSIFT()
    #cross_validate(feature_extractor)
    main(feature_extractor)
