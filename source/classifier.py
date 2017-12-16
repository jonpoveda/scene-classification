import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from typing import List
from typing import Any


class ClassifierFactory(object):
    KNN = 'knn'
    RANDOMFOREST = 'random_forest'
    GAUSSIANBAYES = 'gaussian_bayes'
    BERNOULLIBAYES = 'bernouilli_bayes',
    SVM = 'svm'
    LOGISTICREGRESSION = 'logistic_regression'

    @staticmethod
    def build(name, **kwargs):
        """ Gets a classifier with that name """
        classifiers = {
            'knn': KNN,
            'random_forest': RandomForest,
            'gaussian_bayes': GaussianBayes,
            'bernouilli_bayes': BernoulliBayes,
            'svm': SVM,
            'logistic_regression': LogisticRegression
        }
        return classifiers.get(name, None)(**kwargs)


class BaseClassifier(object):
    model = None

    # def train(self, descriptors, labels):
    #     return NotImplementedError
    #
    # def predict(self, descriptor):
    #     return NotImplementedError

    def train(self, descriptors, labels):
        # type: (List, List) -> None
        self.model.fit(descriptors, labels)

    def predict(self, descriptor):
        # type: (np.array) -> np.array
        predictions = self.model.predict(descriptor)
        return predictions

    def predict_list(self, descriptors):
        # type: (List) -> List
        predictions = list()
        for descriptor in descriptors:
            prediction = self.predict(descriptor)
            predictions.append(prediction)
        return predictions


class KNN(BaseClassifier):
    def __init__(self, n_neighbours):
        # type: (int) -> None
        self.model = KNeighborsClassifier(n_neighbours=n_neighbours, n_jobs=-1)


class RandomForest(BaseClassifier):
    def __init__(self, n_estimators=10, max_depth=None):
        # type: (int) -> None
        self.model = RandomForestClassifier(n_estimators=n_estimators,
                                            max_depth=max_depth)


class GaussianBayes(BaseClassifier):
    def __init__(self, priors=None):
        # type: (int) -> None
        self.model = GaussianNB(priors=priors)


class BernoulliBayes(BaseClassifier):
    def __init__(self, alpha=1.0, binarize=0.0, fit_prior=True,
                 class_prior=None):
        # type: (int) -> None
        self.model = BernoulliNB(alpha=alpha,
                                 binarize=binarize,
                                 fit_prior=fit_prior,
                                 class_prior=class_prior)


class SVM(BaseClassifier):
    def __init__(self, penalty='l2'):
        # type: (int) -> None
        self.model = LinearSVC(penalty=penalty)


class LogisticRegression(BaseClassifier):
    def __init__(self, max_iterations=2500, alpha=0.1, lambda_value=0.1):
        # type: (int) -> None
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.lambdaValue = lambda_value
        self.label_list = list(
            ['mountain', 'inside_city', 'Opencountry', 'coast', 'street',
             'forest', 'tallbuilding', 'highway'])
        self.model_list = list()

    def train(self, descriptors, labels):
        # type: (Any, List) -> None
        # Train one model for each type of label (one vs all) 
        for i in range(len(self.label_list)):
            label = self.label_list[i]
            new_labels = [label == labels[x] for x in range(len(labels))]
            new_labels = 1 * np.array(new_labels)
            self.model_list.append(
                self.regularized_gradient_descent(descriptors, new_labels))

    def predict(self, descriptors):
        # type: (np.array) -> np.array
        predictions = list()
        for descriptor in descriptors:
            prediction = -1
            max_res = 0
            for i in range(len(self.label_list)):
                res = self.sigmoid(sum(np.dot(descriptor, self.model_list[i])))
                if (res >= 0.5) and (res > max_res):
                    max_res = res
                    prediction = i
            if prediction >= 0:
                predictions.append(self.label_list[prediction])
            else:
                predictions.append('none')

        return predictions

    def predict_list(self, descriptors):
        # type: (List) -> List
        predictions = list()
        for descriptor in descriptors:
            prediction = self.predict(descriptor)
            predictions.append(prediction)
        return predictions

    def sigmoid(self, x):
        """
        Computes the Sigmoid function of the input argument x.
        """
        return 1.0 / (1 + np.exp(-x))

    def regularized_gradient_descent(self, x, y):

        m, n = x.shape  # number of samples, number of features

        # y must be a column vector
        y = y.reshape(m, 1)

        # initialize the parameters
        theta = np.ones(shape=(n, 1))

        # Repeat until convergence (or max_iterations)
        for iteration in range(self.max_iterations):
            h = self.sigmoid(np.dot(x, theta))
            error = (h - y)
            gradient = np.dot(x.T, error) / m + self.lambdaValue / m * theta

            if np.array_equal(theta, theta - self.alpha * gradient):
                break
            theta = theta - self.alpha * gradient
        return theta

    def classify_vector(self, x, theta):
        """
        Evaluate the Logistic Regression model h(x) with theta parameters,
        and returns the predicted label of x.
        """
        prob = self.sigmoid(sum(np.dot(x, theta)))
        if prob > 0.5:
            return 1.0
        else:
            return 0.0
