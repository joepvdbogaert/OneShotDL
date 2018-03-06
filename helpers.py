import os
import gzip
import numpy as np
from keras.utils import to_categorical
import random

def load_mnist(path, kind='train', normalize=True, return_4d_tensor=True, one_hot=True):
    """ Load MNIST / Fashion MNIST and prepare in specified way. 
    
    :param path: Relative path to the data. 
    :param kind: Data to retrieve. Either 'train' or 'test'.
    :param normalize: Normalize the grayscale from (0,255) to (0,1)
    """

    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    if normalize:
        images = images/255

    if return_4d_tensor:
        images = images.reshape((images.shape[0], 28, 28, 1)) # 28x28 is specific to MNIST, change for other data

    if one_hot:
        labels = to_categorical(labels, 10) # 10 classes in MNIST

    return images, labels


def split_and_select_random_data(x, y, xtest, ytest, num_target_classes=5, num_examples_per_class=1):
    """ Select random classes and return random examples of these classes.

    :param x: Numpy ndarray with the training data.
    :param y: Labels of the training data (one-hot encoded in 2d array).
    :param num_target_classes: Number of classes to select.
    :param num_examples_per_class: Number of examples to select for each class.
    :param return_other_classes: Return the data and labels of the unselected classes as well.
    :param return_unlabeled_examples: Return the not-selected classes and labels as well.
    :returns: The split datasets as ndarrays.
    """ 

    def get_random_labeled_examples(x, y, c, num_labeled):
        """ Extract random examples from datasets. 
        
        :param x: The data.
        :param y: The labels of the data.
        :param c: The class to select from.
        :param num_labeled: The number of examples to return with a label.
        :returns: The labeled data, the labels, and the unlabeled data.
        """
        
        # subset data to that of class c
        x_c = x[ y[:,c]==1 ]
        y_c = y[ y[:,c]==1 ]

        # select num_labeled random examples
        idx = random.sample(range(x_c.shape[0]), num_labeled)
        x_labeled = x_c[idx]
        x_unlabeled = x_c[~np.isin(np.arange(x_c.shape[0]), idx)]
        labels = y_c[idx]

        return x_labeled, labels, x_unlabeled

    # select random classes
    total_classes = y.shape[1]
    target_classes = random.sample(range(total_classes), num_target_classes)

    # split the data based on these classes
    target_bools = (y[:,target_classes].sum(axis=1)==1)
    y_target_all = y[target_bools,:]
    x_target_all = x[target_bools,:]
    y_auxiliary = y[~target_bools]
    x_auxiliary = x[~target_bools]

    test_bools = (ytest[:,target_classes].sum(axis=1)==1)
    x_test = xtest[test_bools,:]
    y_test = ytest[test_bools,:]

    # select random examples of target classes
    x_target_labeled, y_target, x_target_unlabeled = get_random_labeled_examples(x_target_all, y_target_all, target_classes[0], num_examples_per_class)

    for c in target_classes[1:]:
        x_temp, y_temp, x_unlabeled_temp = get_random_labeled_examples(x_target_all, y_target_all, c, num_examples_per_class)
        x_target_labeled = np.concatenate([x_target_labeled, x_temp])
        y_target = np.concatenate([y_target, y_temp])
        x_target_unlabeled = np.concatenate([x_target_unlabeled, x_unlabeled_temp])

    return x_target_labeled, y_target, x_test, y_test, x_target_unlabeled, x_auxiliary, y_auxiliary


def reinitialize_random_weights(model, layer_nr):
    """ Replace weights of a specified layer in a model with random weights.

    :param model: A Keras model.
    :param layer_nr: The number of the layer to reinitialize.
    :returns: A Keras model with random small weights at the specified layer number. 
    """
    
    # get shape of the required weights of the specified layer
    old_weights = model.layers[ layer_nr ].get_weights()
    W_shape = old_weights[0].shape
    b_shape = len(old_weights[1])

    # initialize random weights with same shape
    W_init = np.random.randn( W_shape[0], W_shape[1] ) * 0.01
    b_init = np.random.randn( b_shape ) * 0.01
    new_weights = [W_init, b_init]
    
    # give new weights to model and return model
    model.layers[ layer_nr ].set_weights( new_weights )
    
    return model

def freeze_layers(model, num_fixed_layers, count_only_trainable_layers=True, reinitialize_remaining=False, reinitialize_last=False):
    """ Freeze the first num_fixed_layers layers of a model.
    
    :param model: A keras model.
    :param num_fixed_layers: The number of layers to fix.
    :param count_only_trainable_layers: If true, only counts layers that actually have weights.
                                        If false, attempts to freeze first layers, regardless of type.
    :returns: The Keras model with the specified layers frozen.
    """
    def get_layers_with_weights(model):
        has_weights = []
        for l in range(len(model.layers)):
            W = model.layers[l].get_weights()
            if len(W)>0:
                has_weights.append(l)
        return has_weights
    
    # get the indexes of the layers that have weights
    layers_with_weights = get_layers_with_weights(model)
    for l in layers_with_weights[0:min(num_fixed_layers,len(layers_with_weights)-1)]:
        model.layers[l].trainable = False

    # reinitialize the weights of remaining layers if specified
    if reinitialize_remaining and num_fixed_layers <= len(layers_with_weights)-1:
        for l in layers_with_weights[num_fixed_layers:len(layers_with_weights)-1]:
            model = reinitialize_random_weights(model, l)

    if reinitialize_last:
        model = reinitialize_random_weights(model, layers_with_weights[-1])

    return model
    