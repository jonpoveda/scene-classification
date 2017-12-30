import os
import time
import cPickle

import cv2
import numpy as np
from sklearn import cluster
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from source import DATA_PATH


class BoVW(object):

    def __init__(self, k = 512):
        # type: (int) -> None
        # FIXME: remove number_of_features if they are not explicity needed
        self.k = k
        self.codebook = cluster.MiniBatchKMeans(n_clusters=k, verbose=False,
                                       batch_size=k * 20, compute_labels=False,
                                       reassignment_ratio=10 ** -4,
                                       random_state=42)
        
    def extract_descriptors(self,feature_extractor,train_images_filenames,train_labels):
        # extract SIFT keypoints and descriptors
        # store descriptors in a python list of numpy arrays
        Train_descriptors = []
        Train_label_per_descriptor = []
        for i in range(len(train_images_filenames)):
            filename = train_images_filenames[i]
            filename_path = os.path.join(DATA_PATH, filename)
            print('Reading image ' + filename_path)
            ima = cv2.imread(filename_path)
            kpt, des = feature_extractor.detectAndCompute(ima)
            Train_descriptors.append(des)
            Train_label_per_descriptor.append(train_labels[i])
            print(str(len(kpt)) + ' extracted keypoints and descriptors')
    
        # Transform everything to numpy arrays
        size_descriptors = Train_descriptors[0].shape[1]
        D = np.zeros(
            (np.sum([len(p) for p in Train_descriptors]), size_descriptors),
            dtype=np.uint8)
        startingpoint = 0
        for i in range(len(Train_descriptors)):
            D[startingpoint:startingpoint + len(Train_descriptors[i])] = \
                Train_descriptors[i]
            startingpoint += len(Train_descriptors[i])
        return D,Train_descriptors

    def compute_codebook(self,D, Train_descriptors):
            # compute the codebook
        
            print('Computing kmeans with ' + str(self.k) + ' centroids')
            init = time.time()
        
            self.codebook.fit(D)
            cPickle.dump(self.codebook, open("codebook.dat", "wb"))
            end = time.time()
            print('Done in ' + str(end - init) + ' secs.')
            
            
            # get train visual word encoding
            print('Getting Train BoVW representation')
            init = time.time()
            visual_words = np.zeros((len(Train_descriptors), self.k), dtype=np.float32)
            for i in xrange(len(Train_descriptors)):
                words = self.codebook.predict(Train_descriptors[i])
                visual_words[i, :] = np.bincount(words, minlength=self.k)
            end = time.time()
            print('Done in ' + str(end - init) + ' secs.')
            return visual_words
    
    def cross_validate(self, visual_words, train_labels):
        """ cross_validate classifier with k stratified folds """
        # Train an SVM classifier with RBF kernel
        print('Training the SVM classifier...')
        init = time.time()
        stdSlr = StandardScaler().fit(visual_words)
        D_scaled = stdSlr.transform(visual_words)
        parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10], 'gamma':np.linspace(0, 0.01,num = 11)}
        kfolds = StratifiedKFold(n_splits = 5, shuffle = False, random_state = 50)
        grid = GridSearchCV(svm.SVC(), param_grid = parameters, cv = kfolds, scoring = 'accuracy')
        grid.fit(D_scaled, train_labels)
        end = time.time()
        print('Done in ' + str(end - init) + ' secs.')
        print("Best parameters: %s Accuracy: %0.2f" % (grid.best_params_, grid.best_score_))


