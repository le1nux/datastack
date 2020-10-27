from abc import ABC, abstractmethod
from typing import List, Sequence
from data_hub.dataset.meta_information import DatasetMetaInformation, DatasetMetaInformationFactory


class DatasetIteratorIF(ABC):

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int):
        raise NotImplementedError

    @property
    @abstractmethod
    def underlying_iterators(self) -> List["DatasetIteratorIF"]:
        raise NotImplementedError

    @property
    @abstractmethod
    def dataset_meta_information(self) -> DatasetMetaInformation:
        raise NotImplementedError


class DatasetIterator(DatasetIteratorIF):

    def __init__(self, dataset_meta_information: DatasetMetaInformation):
        self._dataset_meta_information = dataset_meta_information

    @property
    def underlying_iterators(self) -> List["DatasetIteratorIF"]:
        return []

    @property
    def dataset_meta_information(self) -> DatasetMetaInformation:
        return self._dataset_meta_information


class SequenceDatasetIterator(DatasetIterator):

    def __init__(self, dataset_sequences: List[Sequence], dataset_meta_information: DatasetMetaInformation):
        super().__init__(dataset_meta_information)
        self._dataset_sequences = dataset_sequences

    def __len__(self):
        return len(self._dataset_sequences[0])

    def __getitem__(self, index: int):
        return tuple([s[index] for s in self._dataset_sequences])


class DatasetIteratorView(DatasetIterator):
    """Provides a view on a `DatasetIterator` for accessing elements of a given split only."""

    def __init__(self, dataset_iterator: DatasetIterator, indices: List[int], dataset_meta_information: DatasetMetaInformation):
        super().__init__(dataset_meta_information)
        self._dataset_iterator = dataset_iterator
        self._indices = indices

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index: int):
        original_dataset_index = self._indices[index]
        return self._dataset_iterator[original_dataset_index]

    @property
    def underlying_iterators(self) -> List["DatasetIteratorIF"]:
        return [self._dataset_iterator]


class CombinedDatasetIterator(DatasetIterator):

    def __init__(self, iterators: List[DatasetIterator], dataset_meta_information: DatasetMetaInformation):
        super().__init__(dataset_meta_information)
        self._iterators = iterators

    def __len__(self):
        return sum([len(iterator) for iterator in self._iterators])

    def __getitem__(self, index: int):
        index_copy = index
        iterator_lengths = [len(iterator) for iterator in self._iterators]
        for iterator_index, length in enumerate(iterator_lengths):
            if index_copy - length < 0:
                return self._iterators[iterator_index][index_copy]
            index_copy -= length
        raise IndexError

    @property
    def underlying_iterators(self) -> List["DatasetIteratorIF"]:
        return self._iterators
