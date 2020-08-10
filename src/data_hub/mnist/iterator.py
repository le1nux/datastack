import torch
from typing import List
from data_hub.io.resources import StreamedResource


class MNISTIterator:
    """ MNIST dataset iterator (http://yann.lecun.com/exdb/mnist/)
    """

    def __init__(self, samples_stream: StreamedResource, targets_stream: StreamedResource = None):
        self.samples = torch.load(samples_stream)
        self.targets = torch.load(targets_stream)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        """ Returns the sample and target of the dataset at given index position.
        :param index: index within dataset
        :return: sample, target
        """
        return self.samples[index], self.targets[index]
