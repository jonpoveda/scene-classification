import time

import numpy as np

from classifier import KNN
from feature_extractor import SIFT
from source import DATA_PATH
from multiprocessing import Pool
from evaluator import performance_evaluator


def main():
    global feature_extractor
    global classifier
    # Read the train and test files
    from database import Database
    database = Database(DATA_PATH)
    train_images, test_images, train_labels, test_labels = database.get_data()

    # Create the SIFT detector object
    feature_extractor = SIFT(number_of_features=100)

    # Load or compute descriptors
    if database.data_exists():
        print('Loading descriptors...')
        descriptors, labels = database.get_descriptors()
    else:
        print('Computing descriptors...')
        descriptors, labels = feature_extractor.extract_from(train_images,
                                                             train_labels)
        database.save_descriptors(descriptors, labels)

    # Train a k-nn classifier with train dataset
    classifier = KNN(n_neighbours=5)
    classifier.train(descriptors, labels)

    # Assess classifier with test dataset
    num_test_images = 0
    num_correct = 0
#    for i in range(len(test_images)):
#        test_descriptor, _ = \
#            feature_extractor.extract(test_images[i], test_labels[i])
#        predictions = classifier.predict(test_descriptor)
#    
#        num_test_images += 1
#        is_a_match, predicted_class = \
#            assess_a_prediction(predictions, test_images[i], test_labels[i])
#        if is_a_match:
#            num_correct += 1
#        print('{} image {} was from class {} and was predicted {}'.format(
#            int(is_a_match), test_images[i], test_labels[i], predicted_class))

    images = list(range(807))
    for i in range(len(images)):
        images[i] = test_images[i]
        
        
    pool = Pool(processes=4)

    predicted_class = pool.map(predict_image, images )
    
    for i in range(len(images)):
        print('image ' + test_images[i] + ' was from class ' +test_labels[i] + ' and was predicted ' + predicted_class[i])
        num_test_images += 1
        if predicted_class[i] == test_labels[i]:
            num_correct += 1 
    print('Final accuracy: ' + str(num_correct * 100.0 / num_test_images))
    
    evaluator = performance_evaluator(test_labels, predicted_class)
    
    print('Evaluator accuracy: ' + str(evaluator.accuracy))
    print('Evaluator precision: ' + str(evaluator.precision))
    print('Evaluator recall: ' + str(evaluator.recall))
    print('Evaluator Fscore: ' + str(evaluator.Fscore))
    evaluator.confusion_matrix()

    # original  : 30.48% in 302 secs
    # no pool   : 36.31% in 238 secs
    # 4-pool    : 36.31% in 129 secs



def predict_image (image):
    global feature_extractor
    global classifier
    test_descriptor = feature_extractor.extract_pool(image)
    predictions = classifier.predict(test_descriptor)
    values, counts = np.unique(predictions, return_counts=True)
    predicted_class = values[np.argmax(counts)]
    #predicted_class = "pool"
    return predicted_class
    
def assess_a_prediction(predictions_per_descriptor, test_image, test_label):
    # FIXME: test_image is only for printing, remove it
    values, counts = np.unique(predictions_per_descriptor, return_counts=True)
    predicted_class = values[np.argmax(counts)]
    return predicted_class == test_label, predicted_class


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
print('Done in ' + str(end - start) + ' secs.')
