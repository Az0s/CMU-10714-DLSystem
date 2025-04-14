"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl
import logging
# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
# logger = logging.getLogger()


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    # Ref Q2 of hw0
    with gzip.open(image_filesname) as f: 
        magic, ni, nr, nc = struct.unpack('>iiii', f.read(16))
        # print(magic, ni, nr, nc)
        # should follow IDX file format, but hardcode here
        assert magic == 2051
        pixels = np.frombuffer(f.read(), dtype=np.uint8)
        pixels = pixels.reshape((ni, nr * nc)).astype(np.float32) / 255.0
        # print(pixels.shape)
    with gzip.open(label_filename) as f:
        magic, ni = struct.unpack('>ii', f.read(8))
        # print(magic, ni)
        assert magic == 2049
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        # print(labels.shape)
        labels = labels.reshape((ni,)).astype(np.uint8)
        print(labels.shape)
    return pixels, labels
    ### END YOUR SOLUTION


def softmax_loss(Z: ndl.Tensor, y_one_hot: ndl.Tensor):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    # v1 = Z
    # v2 = y_one_hot
    logger = logging.getLogger('softmax_loss')
    if not logger.handlers:
        handler = logging.FileHandler('softmax_loss_debug.log')
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    def log_tensor(tensor, name):
        # Log tensor details: name, shape, values (or summary), and memory address
        try:
            shape = tensor.shape
            # Sample a few values to avoid flooding the log
            # values = tensor.numpy().flatten()[:5] if tensor.size > 5 else tensor.numpy().flatten()
            address = id(tensor)  # Python object ID
            # If ndl.Tensor wraps a numpy array, log its address too
            # data_address = id(tensor.numpy()) if hasattr(tensor, 'numpy') else 'N/A'
            logger.debug(f"{name}: shape={shape}, id={address}")
        except Exception as e:
            logger.error(f"Error logging {name}: {e}")

    B, k = Z.shape
    log_tensor(Z, "Z")
    log_tensor(y_one_hot, "y_one_hot")

    v3 = ndl.exp(Z)
    log_tensor(v3, "v3 (exp(Z))")

    v4 = ndl.summation(v3, axes=1)
    log_tensor(v4, "v4 (sum(exp(Z), axis=1))")

    v5 = ndl.log(v4)
    log_tensor(v5, "v5 (log(sum(exp(Z))))")

    v6 = Z.reshape((B, 1, k))
    log_tensor(v6, "v6 (Z reshaped)")

    v7 = y_one_hot.reshape((B, k, 1))
    log_tensor(v7, "v7 (y_one_hot reshaped)")

    v8 = v6 @ v7
    log_tensor(v8, "v8 (Z @ y_one_hot)")

    v9 = v8.reshape((B,))
    log_tensor(v9, "v9 (v8 reshaped)")

    v10 = v5 - v9
    log_tensor(v10, "v10 (v5 - v9)")

    v11 = ndl.summation(v10)
    log_tensor(v11, "v11 (sum(v10))")

    v12 = v11 / v10.shape[0]
    log_tensor(v12, "v12 (v11 / batch_size)")

    return v12
    



def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
