import cv2
import numpy as np

from source import DATA_PATH


class performance_evaluator(object):

    def __init__(self, ground_truth, labels):
        # type: (int) -> None
        # FIXME: remove number_of_features if they are not explicity needed
        self.ground_truth = ground_truth[0:len(labels)]
        self.labels = labels
        self.label_list = list(['mountain', 'inside_city', 'Opencountry', 'coast', 'street', 'forest', 'tallbuilding', 'highway'])
        self.__compute()
        
    
    def __compute(self):
        from  sklearn.metrics import precision_score
        from  sklearn.metrics import recall_score
        from  sklearn.metrics import accuracy_score

        self.precision = precision_score(self.ground_truth, self.labels, labels = self.label_list, average = 'macro')
        self.recall = recall_score(self.ground_truth, self.labels, labels = self.label_list, average = 'macro')
        self.accuracy = accuracy_score(self.ground_truth, self.labels)
        self.Fscore = 2*(self.precision*self.recall)/(self.precision+self.recall)
        
    def confusion_matrix(self):
        # Plot the confusion matrix on test data
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.ground_truth, self.labels)
        print 'Confusion matrix:'
        print (cm)
        
        # Plot confusion matrix
        import matplotlib.pyplot as plt
        plt.matshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    