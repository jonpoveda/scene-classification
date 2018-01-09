import keras
from keras import Sequential
from keras.layers import Dense
import numpy as np
from sklearn import preprocessing


def add_differences(matrix):
    # Create and array with 2 extra parameters
    tmp = np.array(np.zeros(
        (matrix.shape[0], 2 + matrix.shape[1])))
    print(matrix.shape)
    print(tmp.shape)
    print(tmp)
    for (i, vector) in enumerate(matrix):
        diff = [vector[1] - vector[0], vector[2] - vector[1]]
        expanded_vector = np.append(vector, diff)
        tmp[i] = expanded_vector
        print(expanded_vector)

    return tmp


# Generate dummy data: constant, increasing or decreasing sequences
train_data = np.array(
    [[0, 0, 0], [1, 1, 1], [2, 2, 2],
     [0, 1, 2], [1, 2, 3], [3, 5, 7],
     [6, 7, 8], [9, 8, 7], [6, 4, 2]], dtype=np.float32)
train_labels = np.array([0, 0, 0, 1, 1, 1, -1, -1, -1], dtype=np.int8)

test_data = np.array(
    [[10, 10, 10], [11, 11, 11], [22, 22, 22],
     [5, 8, 9], [2, 3, 7], [1, 5, 9],
     [6, 2, 0], [19, 18, 17], [12, 8, 4]], dtype=np.float32)
test_labels = np.array([0, 0, 0, 1, 1, 1, -1, -1, -1], dtype=np.int8)

# Normalize data
train_data_normalized = np.array(preprocessing.normalize(train_data,
                                                         norm='l2',
                                                         axis=1))
test_data_normalized = np.array(preprocessing.normalize(test_data,
                                                        norm='l2',
                                                        axis=1))

# Build one-hot vector labels
train_labels_onehot = keras.utils.to_categorical(train_labels, num_classes=3)
test_labels_onehot = keras.utils.to_categorical(test_labels, num_classes=3)

# Add extra features to vectors
train_data_normalized = add_differences(train_data_normalized)
test_data_normalized = add_differences(test_data_normalized)

# For a single-input model with 3 classes (categorical classification):
model = Sequential()
model.add(Dense(32, init='uniform', activation='relu', input_dim=5))
model.add(Dense(3, init='uniform', activation='softmax'))
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model, iterating on the data in batches of 32 samples
model.fit(train_data_normalized, train_labels_onehot, epochs=100, batch_size=2)

# Asses classifier
train_scores = model.evaluate(train_data_normalized, train_labels_onehot)
print('Train metrics: {} {}'.format(
    model.metrics_names[1], train_scores[1] * 100))

scores = model.evaluate(test_data_normalized, test_labels_onehot)
print('Validation metrics: {} {}'.format(
    model.metrics_names[1], scores[1] * 100))
