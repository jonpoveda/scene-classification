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


def main(feature_extractor, spatial_pyramid = False, histogram_intersection = False):
    do_plotting = True
    
    start = time.time()
    
    # Read the train and test files
    database = Database(DATA_PATH)
    train_images_filenames, test_images_filenames, train_labels, test_labels = \
        database.get_data()
    
    #Create BoVW classifier
    BoVW_classifier = BoVW(spatial_pyramid = spatial_pyramid, histogram_intersection = histogram_intersection)
    
    #Extract image descriptors
    D,Train_descriptors,Keypoints = BoVW_classifier.extract_descriptors(feature_extractor,train_images_filenames,train_labels)
    
    #Compute Codebook
    BoVW_classifier.compute_codebook( D)
    
    # get train visual word encoding
    visual_words = BoVW_classifier.get_train_encoding( Train_descriptors,Keypoints)
    
    # Train an SVM classifier 
    train_data = BoVW_classifier.train_classifier( visual_words, train_labels)

    # get all the test data
    visual_words_test = BoVW_classifier.predict_images(test_images_filenames,feature_extractor)

    # Test the classification accuracy
    BoVW_classifier.evaluate_performance(visual_words_test,test_labels, do_plotting, train_data)

    end = time.time()
    print('Everything done in ' + str(end - start) + ' secs.')
    ### 69.02%
    
def cross_validate(feature_extractor, spatial_pyramid = False, histogram_intersection = False):
    # Read the train and test files
    database = Database(DATA_PATH)
    train_images_filenames, test_images_filenames, train_labels, test_labels = \
        database.get_data()
    
    #Create BoVW classifier
    BoVW_classifier = BoVW(spatial_pyramid = spatial_pyramid, histogram_intersection = histogram_intersection)
    
    #Extract image descriptors
    D,Train_descriptors,Keypoints = BoVW_classifier.extract_descriptors(feature_extractor,train_images_filenames,train_labels)
    
    #Compute Codebook
    BoVW_classifier.compute_codebook( D)
    
    # get train visual word encoding
    visual_words = BoVW_classifier.get_train_encoding( Train_descriptors,Keypoints)
    
    #Cross validate classifier
    BoVW_classifier.cross_validate(visual_words,train_labels)


if __name__ == "__main__":
    # FIXME: use 300 n features
    #feature_extractor = SIFT2(number_of_features=30)
    #feature_extractor = SIFT(number_of_features=300)
    feature_extractor = denseSIFT()
    #cross_validate(feature_extractor, spatial_pyramid = False, histogram_intersection = False)
    main(feature_extractor, spatial_pyramid = True, histogram_intersection = True)
