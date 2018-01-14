import fnmatch
import os

import imageio
from scipy import ndimage
from sklearn.feature_extraction import image as skimage

from source import DATA_PATH


def split_into_patches(dbpath):
    print(dbpath)

    image_paths = []
    for root, dirnames, filenames in os.walk(dbpath):
        for filename in fnmatch.filter(filenames, '*.jpg'):
            image_paths.append(os.path.join(root, filename))

    for image_path in image_paths:
        image = ndimage.imread(image_path)
        print(image.shape)
        patches = skimage.extract_patches_2d(image, patch_size=(32, 32),
                                             max_patches=64)

        print(patches.shape)
        for (npatch, patch) in enumerate(patches):
            output_path = '{}_{}.jpg'.format(image_path.partition('.jpg')[0],
                                             str(npatch).zfill(4))
            output_path = output_path.replace('/data/', '/data_patch/')
            print(output_path)
            try:
                os.makedirs(os.path.dirname(output_path))
            except OSError as expected:
                pass
            imageio.imsave(output_path, patch)


# split_into_patches(dbpath=os.path.join(DATA_PATH, 'train'))
split_into_patches(dbpath=os.path.join(DATA_PATH, 'test'))
