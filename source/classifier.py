from sklearn.neighbors import KNeighborsClassifier


class Classifier(object):
    model = None

    def train(self, descriptors, labels):
        return NotImplementedError

    def predict(self, descriptor):
        return NotImplementedError


class KNN(Classifier):
    def __init__(self, n_neighbours):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbours, n_jobs=-1)

    def train(self, descriptors, labels):
        self.model.fit(descriptors, labels)

    def predict(self, descriptor):
        predictions = self.model.predict(descriptor)
        return predictions
