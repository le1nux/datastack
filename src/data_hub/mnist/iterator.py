import torch
from data_hub.io.resources import StreamedResource
from data_hub.dataset.iterator import SequenceDatasetIterator, DatasetMetaInformation


class MNISTIterator(SequenceDatasetIterator):
    """ MNIST dataset iterator (http://yann.lecun.com/exdb/mnist/)
    """

    def __init__(self, samples_stream: StreamedResource, targets_stream: StreamedResource, dataset_meta_information: DatasetMetaInformation):
        targets = [int(target) for target in torch.load(targets_stream)]
        dataset_sequences = [torch.load(samples_stream), targets]
        super().__init__(dataset_sequences=dataset_sequences, dataset_meta_information=dataset_meta_information)
