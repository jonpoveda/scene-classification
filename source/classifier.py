import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from typing import List


class BaseClassifier(object):
    model = None

    def train(self, descriptors, labels):
        return NotImplementedError

    def predict(self, descriptor):
        return NotImplementedError


class KNN(BaseClassifier):
    def __init__(self, n_neighbours):
        # type: (int) -> None
        self.model = KNeighborsClassifier(n_neighbors=n_neighbours, n_jobs=-1)

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
