class DataGeneratorConfig(object):
    """ Contains dictionaries of configurations """
    DEFAULT = dict(featurewise_center=False,
                   samplewise_center=False,
                   featurewise_std_normalization=False,
                   samplewise_std_normalization=False,
                   preprocessing_function=preprocess_input,
                   rotation_range=0.,
                   width_shift_range=0.,
                   height_shift_range=0.,
                   shear_range=0.,
                   zoom_range=0.,
                   channel_shift_range=0.,
                   fill_mode='nearest',
                   cval=0.,
                   horizontal_flip=False,
                   vertical_flip=False,
                   rescale=None)

    CONFIG1 = dict(featurewise_center=True,
                   samplewise_center=True,
                   featurewise_std_normalization=False,
                   samplewise_std_normalization=False,
                   preprocessing_function=preprocess_input,
                   rotation_range=15,
                   width_shift_range=0.9,
                   height_shift_range=0.,
                   shear_range=0.,
                   zoom_range=0.3,
                   channel_shift_range=0.,
                   fill_mode='reflect',
                   cval=0.,
                   horizontal_flip=True,
                   vertical_flip=False,
                   rescale=None)


def colour_channel_swapping(x, dim_ordering):
    """ Colour channel swapping from RGB to BGR """
    if dim_ordering == 'th':
        x = x[::-1, :, :]
    elif dim_ordering == 'tf':
        x = x[:, :, ::-1]
    else:
        raise Exception("Ordering not allowed. Use one of these: 'tf' or 'th'")
    return x


def mean_subtraction(x, dim_ordering):
    """ Mean subtraction for VGG16 model re-training

    :param x matrix with the first index following the BGR convention
    :param dim_ordering th for theano and tf for tensorflow
    """
    if dim_ordering == 'th':
        # Zero-center by mean pixel
        x[0, :, :] -= 103.939
        x[1, :, :] -= 116.779
        x[2, :, :] -= 123.68
    elif dim_ordering == 'tf':
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    else:
        raise Exception("Ordering not allowed. Use one of these: 'tf' or 'th'")
    return x


def preprocess_input(x, dim_ordering='default'):
    """ Colour space swapping and data normalization

    From RGB to BGR swapping.
    Apply a mean subtraction (the mean comes from VGG16 model
    """
    # Get type of ordering (Tensorflow or Theanos)
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    x = colour_channel_swapping(x, dim_ordering)
    x = mean_subtraction(x, dim_ordering)
    return x
