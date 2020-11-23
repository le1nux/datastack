from abc import ABC, abstractmethod
from typing import List, Sequence
from data_stack.dataset.meta import DatasetMetaIF


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


class InformedDatasetIteratorIF(DatasetIteratorIF):

    @property
    def dataset_meta(self) -> DatasetMetaIF:
        raise NotImplementedError


class DatasetIterator(DatasetIteratorIF):

    @property
    def underlying_iterators(self) -> List["DatasetIteratorIF"]:
        return []


class InformedDatasetIterator(InformedDatasetIteratorIF):

    def __init__(self, dataset_iterator: DatasetIterator, dataset_meta: DatasetMetaIF):
        self._dataset_meta = dataset_meta
        self._dataset_iterator = dataset_iterator

    def __len__(self):
        return len(self._dataset_iterator)

    def __getitem__(self, index: int):
        return self._dataset_iterator[index]

    @property
    def underlying_iterators(self) -> List["DatasetIteratorIF"]:
        return self._dataset_iterator.underlying_iterators

    @property
    def dataset_meta(self) -> DatasetMetaIF:
        return self._dataset_meta


class SequenceDatasetIterator(DatasetIterator):

    def __init__(self, dataset_sequences: List[Sequence]):
        self._dataset_sequences = dataset_sequences

    def __len__(self):
        return len(self._dataset_sequences[0])

    def __getitem__(self, index: int):
        return tuple([s[index] for s in self._dataset_sequences])

    @property
    def underlying_iterators(self) -> List["DatasetIteratorIF"]:
        return []


class DatasetIteratorView(DatasetIterator):
    """Provides a view on a `DatasetIterator` for accessing elements of a given split only."""

    def __init__(self, dataset_iterator: DatasetIterator, indices: List[int]):
        self._dataset_iterator = dataset_iterator
        self._indices = indices

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, index: int):
        if index >= len(self._indices):
            raise StopIteration
        original_dataset_index = self._indices[index]
        return self._dataset_iterator[original_dataset_index]

    @property
    def underlying_iterators(self) -> List["DatasetIteratorIF"]:
        return [self._dataset_iterator]


class CombinedDatasetIterator(DatasetIterator):

    def __init__(self, iterators: List[DatasetIterator]):
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
