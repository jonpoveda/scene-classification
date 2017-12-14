# # from multiprocessing import Pool
#
# import itertools
#
# from pathos.multiprocessing import Pool
# from typing import List
#
# from classifier import BaseClassifier
# from feature_extractor import BaseFeatureExtractor
#
#
# # def predict_class(filename):
# #     global feature_extractor
# #     global classifier
# #
# #     filename_path = os.path.join(DATA_PATH, filename)
# #     print(filename_path)
# #     ima = cv2.imread(filename_path)
# #     des = feature_extractor._compute(ima)
# #     predictions = classifier.predict(des)
# #     values, counts = np.unique(predictions, return_counts=True)
# #     predicted_class = values[np.argmax(counts)]
# #     return predicted_class
#
#
# # @contextmanager
# # def poolcontext(*args, **kwargs):
# #     pool = Pool(*args, **kwargs)
# #     yield pool
# #     pool.terminate()
# #
# #
# # def pred(assessor, filename):
# #     filename_path = os.path.join(DATA_PATH, filename)
# #     print(filename_path)
# #     ima = cv2.imread(filename_path)
# #     des = assessor.feature_extractor.extract_from([ima], ['no_label'])
# #     predictions = assessor.classifier.predict(des)
# #     values, counts = np.unique(predictions, return_counts=True)
# #     predicted_class = values[np.argmax(counts)]
# #     return predicted_class
#
#
# class Assessor(object):
#     def __init__(self, classifier, feature_extractor):
#         # type: (BaseClassifier, BaseFeatureExtractor) -> (int, int)
#         self.classifier = classifier
#         self.feature_extractor = feature_extractor
#         # from classifier import KNN
#         # from feature_extractor import SIFT
#         # self.classifier = KNN(5)
#         # self.feature_extractor = SIFT(number_of_features=100)
#
#     def assess(self, test_images, test_labels):
#         # type: (List, List) -> (int, int)
#         # get all the test data and predict their labels
#         num_test_images = 0
#         num_correct = 0
#
#         pool = Pool(processes=4)
#
#         test_images = test_images[0:10]
#         descriptors = pool.map(self.feature_extractor.extract_from,
#                                itertools.izip(test_images,
#                                               itertools.repeat(['no_label'])))
#         predicted_class = pool.map(self.classifier.predict, descriptors)
#
#         for i in range(len(test_images)):
#             print('image ' + test_images[i] + ' was from class ' + test_labels[
#                 i] + ' and was predicted ' + predicted_class[i])
#             num_test_images += 1
#             if predicted_class[i] == test_labels[i]:
#                 num_correct += 1
#         return num_correct, num_test_images
#
#         # def predict(self, filename):
#         #     filename_path = os.path.join(DATA_PATH, filename)
#         #     print(filename_path)
#         #     ima = cv2.imread(filename_path)
#         #     des = feature_extractor._compute(ima)
#         #     predictions = classifier.predict(des)
#         #     values, counts = np.unique(predictions, return_counts=True)
#         #     predicted_class = values[np.argmax(counts)]
#         #     return predicted_class
