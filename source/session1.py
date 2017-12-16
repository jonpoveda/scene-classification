from multiprocessing import Pool
import time

from matplotlib import pyplot as plt
import numpy as np

from classifier import ClassifierFactory
from evaluator import Evaluator
from feature_extractor import SIFT
from source import DATA_PATH


def main(classifier_type=ClassifierFactory.KNN, threading='multi',
         **classifier_kwargs):
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

    # Select classification model
    print('Trainning model...')
    classifier = ClassifierFactory.build(classifier_type, **classifier_kwargs)

    # Train a classifier with train dataset
    print('Trainning model...')
    classifier.train(descriptors, labels)

    # Assess classifier with test dataset
    print('Assessing images...')
    if threading == 'multi':
        predicted_class = predict_images_pool(test_images)
    elif threading == 'single':
        predicted_class = predict_images(test_images, test_labels)

    # Evaluate performance metrics
    num_test_images = 0
    num_correct = 0

    images = list(range(10))
    for i in range(len(images)):
        images[i] = test_images[i]

    pool = Pool(processes=4)

    predicted_class = pool.map(predict_image, images)

    for i in range(len(images)):
        print('image {} was from class {} and was predicted {}'.format(
            test_images[i], test_labels[i], predicted_class[i]))
        num_test_images += 1
        if predicted_class[i] == test_labels[i]:
            num_correct += 1

    print('Final accuracy: ' + str(num_correct * 100.0 / num_test_images))

    evaluator = Evaluator(test_labels, predicted_class)

    print('Evaluator accuracy: {}'.format(evaluator.accuracy))
    print('Evaluator precision: {}'.format(evaluator.precision))
    print('Evaluator recall: {}'.format(evaluator.recall))
    print('Evaluator Fscore: {}'.format(evaluator.fscore))

    cm = evaluator.confusion_matrix()

    # Plot the confusion matrix on test data
    print('Confusion matrix:')
    print(cm)
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # original  : 30.48% in 302 secs
    # no pool   : 36.31% in 238 secs
    # 4-pool    : 36.31% in 129 secs


def predict_images(test_images, test_labels):
    prediction_list = list()
    for i in range(len(test_images)):
        test_descriptor, _ = \
            feature_extractor.extract(test_images[i], test_labels[i])
        predictions = classifier.predict(test_descriptor)

        is_a_match, predicted_class = \
            assess_a_prediction(predictions, test_images[i], test_labels[i])
        prediction_list.append(predicted_class)
    return prediction_list


def predict_images_pool(test_images):
    images = list(range(10))
    for i in range(len(images)):
        images[i] = test_images[i]

    pool = Pool(processes=4)
    predicted_class = pool.map(predict_image, images)
    # predicted_class = pool.map(predict_image, test_images)
    return predicted_class


def predict_image(image):
    global feature_extractor
    global classifier
    test_descriptor = feature_extractor.extract_pool(image)
    print('{} extracted keypoints and descriptors'.format(
        len(test_descriptor)))
    predictions = classifier.predict(test_descriptor)
    values, counts = np.unique(predictions, return_counts=True)
    predicted_class = values[np.argmax(counts)]
    return predicted_class


def assess_a_prediction(predictions_per_descriptor, test_image, test_label):
    # FIXME: test_image is only for printing, remove it
    values, counts = np.unique(predictions_per_descriptor, return_counts=True)
    predicted_class = values[np.argmax(counts)]
    return predicted_class == test_label, predicted_class


if __name__ == '__main__':
    start = time.time()
    main(classifier_type=ClassifierFactory.RANDOMFOREST, threading='multi')
    # main(classifier_type=ClassifierFactory.KNN, threading='multi',
    #      n_neighbours=5)
    end = time.time()
    print('Done in {} secs.'.format(end - start))
