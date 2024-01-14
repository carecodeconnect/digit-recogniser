# functions for data transformation

import numpy as np

def one_hot_encode(y, num_class_labels=10):
    """
    Convert class labels to one-hot encoded vectors.

    This function takes an array of class labels and converts them into one-hot encoded
    vectors. Each one-hot encoded vector represents the presence of a class label using a
    1.0 in the corresponding position and 0.0 elsewhere.

    Parameters:
    - y (array-like): An array of class labels to be one-hot encoded.
    - num_labels (int, optional): The total number of unique class labels. Defaults to 10.

    Returns:
    - one_hot (numpy.ndarray): A 2D numpy array where each column is a one-hot encoded
    vector representing a class label.

    Example usage:
    # y represents 5 people
    # columns are people
    # values are datapoints

    >>> y = [0, 2, 1, 3, 0]
    >>> one_hot_enc(y, num_labels=4)
    array([[1., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
        [1., 0., 0., 0.]])
    """
    # one_hot = np.zeros([len(y), num_labels])
    # for i in range(len(y)):
    #     one_hot[n, y[i]] = 1

    # one_hot = np.eye(num_labels)[y]
    
    #one_hot = np.zeros((num_labels, len(y)))

    #for idx, label in enumerate(y):
    #    one_hot[label, idx] = 1.0
    
    #return one_hot
    one_hot = np.zeros((len(y), num_class_labels), dtype=np.float32)
    
    for i, val in enumerate(y):
        one_hot[i, val] = 1.0

    one_hot=np.eye(num_class_labels)[y]
    
    return one_hot

def one_hot_decode(one_hot_vectors):
    """
    Convert one-hot encoded vectors to class labels.

    This function takes a 2D numpy array where each column represents a one-hot encoded vector,
    and it returns an array of class labels corresponding to the columns with 1.0 values.

    Parameters:
    - one_hot_vectors (numpy.ndarray): A 2D numpy array of one-hot encoded vectors.

    Returns:
    - decoded_labels (numpy.ndarray): A 1D numpy array containing the class labels.

    Example usage:

    >>> one_hot_vectors = np.array([[1., 0., 0.],
                                    [0., 0., 1.],
                                    [0., 1., 0.]])
    >>> one_hot_decode(one_hot_vectors)
    array([0, 2, 1])
    """
    decoded_labels = np.argmax(one_hot_vectors, axis=1)
    return decoded_labels
