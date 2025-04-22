from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct 
class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
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
                # print(labels.shape)
            return pixels, labels
            ### END YOUR SOLUTION

        self.image, self.label = parse_mnist(image_filename, label_filename)
        assert self.image.shape[0] == self.label.shape[0], "Dataset size should match over image and lable."
        self.B, self.C = self.image.shape
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        # `index` can be of various of types
        # but here we just pass it down to numpy to handle it
        imgs = self.image[index].reshape(-1, 28, 28, 1)
        return (np.array([self.apply_transforms(x).reshape(784) for x in imgs]), self.label[index])
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.B
        ### END YOUR SOLUTION