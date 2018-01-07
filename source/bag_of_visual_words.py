import cPickle
import os
import time

import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn import cluster
from sklearn import svm
from sklearn.mixture import gaussian_mixture
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from evaluator import Evaluator
from source import TEST_PATH, TRAIN_PATH


def histogram_intersection(X, Y):
    # Function to implement histogram intersection kernel
    x = X.shape[0]
    y = Y.shape[0]

    intersection = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            aux = np.sum(np.minimum(X[i], Y[j]))
            intersection[i][j] = aux
    return intersection


class BoVW(object):
    def __init__(self, k=512, spatial_pyramid=False,
                 histogram_intersection=False):
        # type: (int) -> None
        # FIXME: remove number_of_features if they are not explicity needed
        self.k = k
        self.codebook = self.build_codebook(k)

        self.spatial_pyramid = spatial_pyramid
        self.histogram_intersection = histogram_intersection

    def build_codebook(self, k):
        return cluster.MiniBatchKMeans(n_clusters=k, verbose=False,
                                       batch_size=k * 20,
                                       compute_labels=False,
                                       reassignment_ratio=10 ** -4,
                                       random_state=42)

    def spatial_pyramid_histogram(self, descriptors, keypoints, w=256, h=256):
        # compute spatial pyramid histogram

        words = self.codebook.predict(descriptors)
        width = int(w / 4)
        height = int(h / 4)

        level_zero = np.zeros((16, self.k))
        for i in range(len(descriptors)):
            x = keypoints[i].pt[0]
            y = keypoints[i].pt[1]
            spatial_index = int(x / width) + int(y / height) * 4
            level_zero[spatial_index][words[i]] += 1

        level_one = np.zeros((4, self.k))
        level_one[0] = level_zero[0] + level_zero[1] + level_zero[4] + \
                       level_zero[5]
        level_one[1] = level_zero[2] + level_zero[3] + level_zero[6] + \
                       level_zero[7]
        level_one[2] = level_zero[8] + level_zero[9] + level_zero[12] + \
                       level_zero[13]
        level_one[3] = level_zero[10] + level_zero[11] + level_zero[14] + \
                       level_zero[15]

        level_two = level_one[0] + level_one[1] + level_one[2] + level_one[3]

        aux_two = level_two.flatten() * 0.25
        aux_one = level_one.flatten() * 0.25
        aux_zero = level_zero.flatten() * 0.5
        result = np.concatenate((aux_two, aux_one, aux_zero))
        return result

    def extract_descriptors(self, feature_extractor, train_images_filenames,
                            train_labels):
        # extract SIFT keypoints and descriptors
        # store descriptors in a python list of numpy arrays
        Train_descriptors = []
        Keypoints = []
        Train_label_per_descriptor = []
        for i in range(len(train_images_filenames)):
            filename = train_images_filenames[i]
            filename_path = os.path.join(TRAIN_PATH, filename)
            print('Reading image ' + filename_path)
            ima = cv2.imread(filename_path)
            kpt, des = feature_extractor.detectAndCompute(ima)
            Train_descriptors.append(des)
            Keypoints.append(kpt)
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
        return D, Train_descriptors, Keypoints

    def compute_codebook(self, D):
        # compute the codebook

        print('Computing kmeans with ' + str(self.k) + ' centroids')
        init = time.time()
        self.codebook.fit(D)
        cPickle.dump(self.codebook, open("codebook.dat", "wb"))
        end = time.time()
        print('Done in ' + str(end - init) + ' secs.')

    def get_train_encoding(self, Train_descriptors, Keypoints):
        # get train visual word encoding
        """
        :return: visual words
        """
        print('Getting Train BoVW representation')
        init = time.time()
        # no spatial pyramid algorithm
        if self.spatial_pyramid is False:
            visual_words = np.zeros((len(Train_descriptors), self.k),
                                    dtype=np.float32)
            for i in xrange(len(Train_descriptors)):
                words = self.codebook.predict(Train_descriptors[i])
                visual_words[i, :] = np.bincount(words, minlength=self.k)
                # spatial pyramid algorithm
        else:
            visual_words = np.zeros((len(Train_descriptors), self.k * 21),
                                    dtype=np.float32)
            for i in xrange(len(Train_descriptors)):
                visual_words[i, :] = self.spatial_pyramid_histogram(
                    Train_descriptors[i], Keypoints[i])

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
        kfolds = StratifiedKFold(n_splits=5, shuffle=False, random_state=50)
        if self.histogram_intersection is False:
            parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10],
                          'gamma': np.linspace(0, 0.01, num=11)}
            grid = GridSearchCV(svm.SVC(), param_grid=parameters, cv=kfolds,
                                scoring='accuracy')
            grid.fit(D_scaled, train_labels)
        else:
            parameters = {'kernel': ('precomputed', 'linear')}

            grid = GridSearchCV(svm.SVC(), param_grid=parameters, cv=kfolds,
                                scoring='accuracy')
            gram = histogram_intersection(D_scaled, D_scaled)
            grid.fit(gram, train_labels)
        end = time.time()
        print('Done in ' + str(end - init) + ' secs.')
        print("Best parameters: %s Accuracy: %0.2f" % (
            grid.best_params_, grid.best_score_))

    def train_classifier(self, visual_words, train_labels):
        # Train an SVM classifier
        print('Training the SVM classifier...')
        init = time.time()
        self.stdSlr = StandardScaler().fit(visual_words)
        D_scaled = self.stdSlr.transform(visual_words)
        if self.histogram_intersection is False:
            # Train an SVM classifier with RBF kernel
            self.clf = svm.SVC(kernel='rbf', C=10, gamma=.002).fit(D_scaled,
                                                                   train_labels)
        else:
            # Train an SVM classifier with histogram intersection kernel
            self.clf = svm.SVC(kernel=histogram_intersection).fit(D_scaled,
                                                                  train_labels)
        end = time.time()
        print('Done in ' + str(end - init) + ' secs.')
        return D_scaled

    def predict_images(self, test_images_filenames, feature_extractor):
        # get all the test data
        print('Getting Test BoVW representation')
        init = time.time()
        if self.spatial_pyramid is False:
            visual_words_test = np.zeros((len(test_images_filenames), self.k),
                                         dtype=np.float32)
        else:
            visual_words_test = np.zeros(
                (len(test_images_filenames), self.k * 21),
                dtype=np.float32)

        for i in range(len(test_images_filenames)):
            filename = test_images_filenames[i]
            filename_path = os.path.join(TEST_PATH, filename)
            print('Reading image ' + filename_path)
            ima = cv2.imread(filename_path)
            kpt, des = feature_extractor.detectAndCompute(ima)
            if self.spatial_pyramid == False:
                words = self.codebook.predict(des)
                visual_words_test[i, :] = np.bincount(words, minlength=self.k)
            else:
                visual_words_test[i, :] = self.spatial_pyramid_histogram(des,
                                                                         kpt)

        end = time.time()
        print('Done in ' + str(end - init) + ' secs.')
        return visual_words_test

    def evaluate_performance(self, visual_words_test, test_labels, do_plotting,
                             train_data):
        # Test the classification accuracy
        print('Testing the SVM classifier...')
        init = time.time()
        test_data = self.stdSlr.transform(visual_words_test)
        accuracy = 100 * self.clf.score(test_data, test_labels)

        predictions = self.clf.predict(test_data)
        evaluator = Evaluator(test_labels, predictions)
        print(
            'Evaluator \nAccuracy: {} \nPrecision: {} \nRecall: {} \nFscore: {}'.
                format(evaluator.accuracy, evaluator.precision,
                       evaluator.recall,
                       evaluator.fscore))

        cm = evaluator.confusion_matrix()

        # Plot the confusion matrix on test data
        print('Confusion matrix:')
        print(cm)
        if do_plotting:
            plt.matshow(cm)
            plt.title('Confusion matrix')
            plt.colorbar()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.show()

        end = time.time()
        print('Done in ' + str(end - init) + ' secs.')
        print('Final accuracy: ' + str(accuracy))


class ExtendedBoVW(BoVW):
    """ Implements BoVW with GMMs and Fisher Vectors """

    # NOTE: this methdod is not needed in execution but it seems the only way
    # PyCharm has reference to a GMM for the codebook (without this method it
    # thinks is a kminibatch
    def __init__(self, k, spatial_pyramid=False,
                 histogram_intersection=False):
        # type: (int) -> None
        # FIXME: remove number_of_features if they are not explicity needed
        self.k = k
        self.codebook = self.build_codebook(k)

        self.spatial_pyramid = spatial_pyramid
        self.histogram_intersection = histogram_intersection

    def build_codebook(self, k):
        print('Building a GMM of {} components as a codebook'.format(k))
        return gaussian_mixture.GaussianMixture(n_components=k,
                                                verbose=False,
                                                covariance_type='diag',
                                                tol=1e-3,
                                                reg_covar=1e-6,
                                                max_iter=100)

    def compute_codebook(self, D):
        # compute the codebook
        print('Computing GMM with ' + str(self.k) + ' centroids')
        init = time.time()
        self.codebook.fit(D)
        # fv = fisher_vector(D, self.codebook)
        cPickle.dump(self.codebook, open("codebook.dat", "wb"))
        end = time.time()
        print('Done in ' + str(end - init) + ' secs.')

    def get_train_encoding(self, Train_descriptors, Keypoints):
        # get train visual word encoding
        """
        :return: visual words
        """
        print('Getting Train BoVW representation')
        init = time.time()
        # no spatial pyramid algorithm
        if self.spatial_pyramid is False:
            visual_words = np.zeros((len(Train_descriptors), self.k),
                                    dtype=np.float32)
            for i in xrange(len(Train_descriptors)):
                # words = self.codebook.predict(Train_descriptors[i])
                visual_words[i, :] = self.codebook.predict_proba(Train_descriptors[i])
                # visual_words[i, :] = np.bincount(words, minlength=self.k)
                # spatial pyramid algorithm
        else:
            visual_words = np.zeros((len(Train_descriptors), self.k * 21),
                                    dtype=np.float32)
            for i in xrange(len(Train_descriptors)):
                visual_words[i, :] = self.spatial_pyramid_histogram(
                    Train_descriptors[i], Keypoints[i])

        end = time.time()
        print('Done in ' + str(end - init) + ' secs.')
        return visual_words


def fisher_vector(xx, gmm):
    """Computes the Fisher vector on a set of descriptors.
    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors
    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.
    Returns
    -------
    fv: array_like, shape (K + 2 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.
    Reference
    ---------
    J. Krapac, J. Verbeek, F. Jurie.  Modeling Spatial Layout with Fisher
    Vectors for Image Categorization.  In ICCV, 2011.
    http://hal.inria.fr/docs/00/61/94/03/PDF/final.r1.pdf
    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
        - Q_xx_2
        - Q_sum * gmm.means_ ** 2
        # + Q_sum * gmm.covars_
        + Q_sum * gmm.covariances_
        + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))
