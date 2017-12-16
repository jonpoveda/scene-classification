import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC

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
    
    
class random_forest(BaseClassifier):
    def __init__(self, n_estimators=10,  max_depth=None):
        # type: (int) -> None
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth = max_depth)

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
    
class gaussian_bayes(BaseClassifier):
    def __init__(self, priors=None):
        # type: (int) -> None
        self.model = GaussianNB(priors=priors)

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
    
class berounilli_bayes(BaseClassifier):
    def __init__(self, alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None):
        # type: (int) -> None
        self.model = BernoulliNB(alpha=alpha, binarize=binarize, fit_prior=fit_prior, class_prior=class_prior)

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
    
    
class SVM(BaseClassifier):
    def __init__(self, penalty='l2'):
        # type: (int) -> None
        self.model = LinearSVC(penalty=penalty)

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
    
class logistic_regression(BaseClassifier):
    def __init__(self, max_iterations=2500, alpha=0.1, lambdaValue =0.1):
        # type: (int) -> None
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.lambdaValue = lambdaValue
        self.label_list = list(['mountain', 'inside_city', 'Opencountry', 'coast', 'street', 'forest', 'tallbuilding', 'highway'])
        self.model_list = list()

    def train(self, descriptors, labels):
        # type: (List, List) -> None
        # Train one model for each type of label (one vs all) 
        for i in range(len(self.label_list)):
            label = self.label_list[i]
            new_labels = [label == labels[x] for x in range(len(labels))]
            new_labels = 1*np.array(new_labels)
            self.model_list.append(self.RegularizedGradientDescent(descriptors, new_labels))

    def predict(self, descriptors):
        # type: (np.array) -> np.array
        predictions = list()
        for descriptor in descriptors:
            prediction = -1
            max_res = 0
            for i in range(len(self.label_list)):
                res = self.sigmoid(sum(np.dot(descriptor,self.model_list[i])))
                if (res >=0.5)and(res>max_res):
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
    
    def sigmoid(self,X):
        '''
        Computes the Sigmoid function of the input argument X.
        '''
        return 1.0/(1+np.exp(-X))


    def RegularizedGradientDescent(self,x,y):
        
        m,n = x.shape # number of samples, number of features
    
        # y must be a column vector
        y = y.reshape(m,1)
        
        #initialize the parameters
        theta = np.ones(shape=(n,1)) 
        
        # Repeat until convergence (or max_iterations)
        for iteration in range(self.max_iterations):
            h = self.sigmoid(np.dot(x,theta))
            error = (h-y)
            gradient = np.dot(x.T , error) / m + self.lambdaValue/m*theta
            
            if (np.array_equal(theta ,theta - self.alpha*gradient)):
                break
            theta = theta - self.alpha*gradient
        return theta
    
    
    def classifyVector(self,X, theta):
        '''
        Evaluate the Logistic Regression model h(x) with theta parameters,
        and returns the predicted label of x.
        '''
        prob = self.sigmoid(sum(np.dot(X,theta)))
        if prob > 0.5: return 1.0
        else: return 0.0
    