import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.index = 0
        if self.shuffle: 
            idx = np.arange(len(self.dataset))
            np.random.shuffle(idx)
            self.ordering = np.array_split(idx, 
                                           range(self.batch_size, len(self.dataset), self.batch_size))
            
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.index >= len(self.ordering):
            raise StopIteration
        # this only fit dataset with mnist-like format
        # batch_x, batch_y = [], []
        # for idx in self.ordering[self.index]:
        #     x, y = self.dataset[idx]
        #     batch_x.append(x[0])
        #     batch_y.append(y)
        # self.index += 1
        # batch = [Tensor(x) for x in self.dataset[self.ordering[self.index].to_list()]]
        # return [Tensor(batch_x), Tensor(batch_y)]
        # samples = [Tensor(x) for x in self.dataset[self.ordering[self.index]]]
        samples = []
        it = self.dataset[self.ordering[self.index]]
        for x in it:
            samples.append(Tensor(x))
        self.index += 1
        return tuple(samples)
        # return self.dataset[self.ordering[self.index].item(): self.ordering[self.index].item()+self.batch_size]
        ### END YOUR SOLUTION

