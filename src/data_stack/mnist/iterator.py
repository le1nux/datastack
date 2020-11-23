import torch
from data_stack.io.resources import StreamedResource
from data_stack.dataset.iterator import SequenceDatasetIterator


class MNISTIterator(SequenceDatasetIterator):
    """ MNIST dataset iterator (http://yann.lecun.com/exdb/mnist/)
    """

    def __init__(self, samples_stream: StreamedResource, targets_stream: StreamedResource):
        targets = [int(target) for target in torch.load(targets_stream)]
        dataset_sequences = [torch.load(samples_stream), targets]
        samples_stream.close()
        super().__init__(dataset_sequences=dataset_sequences)
