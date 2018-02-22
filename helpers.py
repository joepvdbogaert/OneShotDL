import os
import gzip
import numpy as np
from keras.utils import to_categorical

def load_mnist(path, kind='train', normalize=False, return_4d_tensor=False, one_hot=False):
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
