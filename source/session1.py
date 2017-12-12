import cPickle
from os import path
import time

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from source import DATA_PATH

start = time.time()

# read the train and test files

with open(path.join(DATA_PATH,
                    'train_images_filenames.dat'), 'r') as file_train, \
    open(path.join(DATA_PATH,
                   'test_images_filenames.dat'), 'r') as file_test, \
    open(path.join(DATA_PATH,
                   'train_labels.dat'), 'r') as file_train_labels, \
    open(path.join(DATA_PATH,
                   'train_labels.dat'), 'r') as file_test_labels:
    train_images_filenames = cPickle.load(file_train)
    test_images_filenames = cPickle.load(file_test)
    train_labels = cPickle.load(file_train_labels)
    test_labels = cPickle.load(file_test_labels)

print('Loaded {} training images filenames with classes {}'.
      format(len(train_images_filenames), set(train_labels)))
print('Loaded {} testing images filenames with classes {}'.
      format(len(test_images_filenames), set(test_labels)))

# create the SIFT detector object

SIFT_detector = cv2.SIFT(nfeatures=100)

# read the just 30 train images per class
# extract SIFT keypoints and descriptors
# store descriptors in a python list of numpy arrays

train_descriptors = []
train_label_per_descriptor = []

for i in range(len(train_images_filenames)):
    filename = train_images_filenames[i]
    filename_path = path.join(DATA_PATH, filename)
    if train_label_per_descriptor.count(train_labels[i]) < 30:
        print('Reading image ' + filename)
        ima = cv2.imread(filename_path)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = SIFT_detector.detectAndCompute(gray, None)
        train_descriptors.append(des)
        train_label_per_descriptor.append(train_labels[i])
        print(str(len(kpt)) + ' extracted keypoints and descriptors')

# Transform everything to numpy arrays

D = train_descriptors[0]
L = np.array([train_label_per_descriptor[0]] * train_descriptors[0].shape[0])

for i in range(1, len(train_descriptors)):
    D = np.vstack((D, train_descriptors[i]))
    L = np.hstack((L, np.array(
        [train_label_per_descriptor[i]] * train_descriptors[i].shape[0])))

# Train a k-nn classifier

print('Training the knn classifier...')
my_knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
my_knn.fit(D, L)
print('Done!')

# get all the test data and predict their labels

num_test_images = 0
num_correct = 0
for i in range(len(test_images_filenames)):
    filename = test_images_filenames[i]
    filename_path = path.join(DATA_PATH, filename)
    ima = cv2.imread(filename_path)
    gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
    kpt, des = SIFT_detector.detectAndCompute(gray, None)
    predictions = my_knn.predict(des)
    values, counts = np.unique(predictions, return_counts=True)
    predicted_class = values[np.argmax(counts)]
    print('image ' + filename + ' was from class ' +
          test_labels[i] + ' and was predicted ' + predicted_class)
    num_test_images += 1
    if predicted_class == test_labels[i]:
        num_correct += 1

print('Final accuracy: ' + str(num_correct * 100.0 / num_test_images))

end = time.time()
print('Done in ' + str(end - start) + ' secs.')

## 30.48% in 302 secs.
