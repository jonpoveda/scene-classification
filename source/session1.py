import cPickle
from os import path
import time

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from source import DATA_PATH

start = time.time()

# read the train and test files

train_images_filenames = cPickle.load(
    open(path.join(DATA_PATH, 'train_images_filenames.dat'), 'r'))
test_images_filenames = cPickle.load(
    open(path.join(DATA_PATH, 'test_images_filenames.dat'), 'r'))
train_labels = cPickle.load(
    open(path.join(DATA_PATH, 'train_labels.dat'), 'r'))
test_labels = cPickle.load(open(path.join(DATA_PATH, 'test_labels.dat'), 'r'))

print('Loaded ' + str(
    len(train_images_filenames)) + ' training images filenames with classes ',
      set(train_labels))
print('Loaded ' + str(
    len(test_images_filenames)) + ' testing images filenames with classes ',
      set(test_labels))

# create the SIFT detector object

SIFTdetector = cv2.SIFT(nfeatures=100)

# read the just 30 train images per class
# extract SIFT keypoints and descriptors
# store descriptors in a python list of numpy arrays

Train_descriptors = []
Train_label_per_descriptor = []

for i in range(len(train_images_filenames)):
    filename = train_images_filenames[i]
    filename_path = path.join(DATA_PATH, filename)
    if Train_label_per_descriptor.count(train_labels[i]) < 30:
        print('Reading image ' + filename)
        ima = cv2.imread(filename_path)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = SIFTdetector.detectAndCompute(gray, None)
        Train_descriptors.append(des)
        Train_label_per_descriptor.append(train_labels[i])
        print(str(len(kpt)) + ' extracted keypoints and descriptors')

# Transform everything to numpy arrays

D = Train_descriptors[0]
L = np.array([Train_label_per_descriptor[0]] * Train_descriptors[0].shape[0])

for i in range(1, len(Train_descriptors)):
    D = np.vstack((D, Train_descriptors[i]))
    L = np.hstack((L, np.array(
        [Train_label_per_descriptor[i]] * Train_descriptors[i].shape[0])))

# Train a k-nn classifier

print('Training the knn classifier...')
myknn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
myknn.fit(D, L)
print('Done!')

# get all the test data and predict their labels

numtestimages = 0
numcorrect = 0
for i in range(len(test_images_filenames)):
    filename = test_images_filenames[i]
    filename_path = path.join(DATA_PATH, filename)
    ima = cv2.imread(filename_path)
    gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
    kpt, des = SIFTdetector.detectAndCompute(gray, None)
    predictions = myknn.predict(des)
    values, counts = np.unique(predictions, return_counts=True)
    predictedclass = values[np.argmax(counts)]
    print('image ' + filename + ' was from class ' +
          test_labels[i] + ' and was predicted ' + predictedclass)
    numtestimages += 1
    if predictedclass == test_labels[i]:
        numcorrect += 1

print('Final accuracy: ' + str(numcorrect * 100.0 / numtestimages))

end = time.time()
print('Done in ' + str(end - start) + ' secs.')

## 30.48% in 302 secs.
