import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


class Evaluator(object):
    def __init__(self, ground_truth, labels):
        # type: (int) -> None
        # FIXME: remove number_of_features if they are not explicity needed
        self.ground_truth = ground_truth[0:len(labels)]
        self.labels = labels
        self.label_list = \
            list(['mountain', 'inside_city', 'Opencountry', 'coast',
                  'street', 'forest', 'tallbuilding', 'highway'])
        self.__compute()

    def __compute(self):
        self.precision = precision_score(self.ground_truth,
                                         self.labels,
                                         labels=self.label_list,
                                         average='macro')
        self.recall = recall_score(self.ground_truth,
                                   self.labels,
                                   labels=self.label_list,
                                   average='macro')
        self.accuracy = accuracy_score(self.ground_truth, self.labels)
        self.fscore = \
            2 * (self.precision * self.recall) / (self.precision + self.recall)

    def confusion_matrix(self):
        # type: (None) -> np.array
        """
        Gets the confusion matrix of the evaluation
        :return: an array of shape  = [n_classes, n_classes]
        """
        return confusion_matrix(self.ground_truth, self.labels)
