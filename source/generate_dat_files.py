import cPickle
from glob import glob
import os

from source import DATA_PATH


def for_train():
    folder_path = os.path.join(os.path.dirname(DATA_PATH), 'data-toy',
                               'train')
    paths = glob(folder_path + '**/*/*.jpg')
    relative_paths = list()
    labels = list()
    for path in paths:
        filename = os.path.basename(path)
        dirname = os.path.basename(os.path.dirname(path))
        labels.append(dirname)
        relative_paths.append(dirname + '/' + filename)

    with open(os.path.join(os.path.dirname(folder_path),
                           'train_images_filenames.dat'),
              'w') as file:
        cPickle.dump(relative_paths, file)
    with open(os.path.join(folder_path, 'train_labels.dat'), 'w') as file:
        cPickle.dump(labels, file)
    print(relative_paths)


def for_test():
    folder_path = os.path.join(os.path.dirname(DATA_PATH), 'data-toy',
                               'test')
    paths = glob(folder_path + '**/*/*.jpg')
    relative_paths = list()
    labels = list()
    for path in paths:
        filename = os.path.basename(path)
        dirname = os.path.basename(os.path.dirname(path))
        labels.append(dirname)
        relative_paths.append(dirname + '/' + filename)

    with open(os.path.join(os.path.dirname(folder_path),
                           'test_images_filenames.dat'),
              'w') as file:
        cPickle.dump(relative_paths, file)
    with open(os.path.join(folder_path, 'test_labels.dat'), 'w') as file:
        cPickle.dump(labels, file)
    print(relative_paths)


for_train()
for_test()
