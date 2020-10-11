from typing import Sequence, List
from abc import ABC, abstractmethod
from data_hub.exception import DatasetOutOfBoundsError


class DatasetIteratorIF(ABC):

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int):
        raise NotImplementedError

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def dataset_tag(self) -> str:
        raise NotImplementedError


class DatasetIterator(DatasetIteratorIF):

    def __init__(self, dataset_sequences: List[Sequence], dataset_name: str = None, dataset_tag: str = None):
        self._dataset_name = dataset_name
        self._dataset_sequences = dataset_sequences
        self._dataset_tag = dataset_tag

    def __len__(self):
        return len(self._dataset_sequences[0])

    def __getitem__(self, index: int):
        return tuple([s[index] for s in self._dataset_sequences])

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def dataset_tag(self) -> str:
        return self._dataset_tag


class DatasetIteratorView(DatasetIteratorIF):
    """Provides a view on a `DatasetIterator` for accessing elements of a given split only."""

    def __init__(self, dataset_iterator: DatasetIterator, indices: List[int], dataset_tag: str = None):
        self._dataset_iterator = dataset_iterator
        self._indices = indices
        self._dataset_tag = dataset_tag

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index: int):
        original_dataset_index = self._indices[index]
        return self._dataset_iterator[original_dataset_index]

    @property
    def dataset_name(self) -> str:
        return self._dataset_iterator.dataset_name

    @property
    def dataset_tag(self) -> str:
        return self._dataset_tag


class CombinedDatasetIterator(DatasetIteratorIF):

    def __init__(self, iterators: List[DatasetIterator], dataset_name: str = None, dataset_tag: str = None):
        self._dataset_name = dataset_name
        self._iterators = iterators
        self._dataset_tag = dataset_tag

    def __len__(self):
        return sum([len(iterator) for iterator in self._iterators])

    def __getitem__(self, index: int):
        index_copy = index
        iterator_lenghts = [len(iterator) for iterator in self._iterators]
        for iterator_index, length in enumerate(iterator_lenghts):
            if index_copy - length < 0:
                return self._iterators[iterator_index][index_copy]
            index_copy -= length
        raise IndexError

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def dataset_tag(self) -> str:
        return self._dataset_tag
