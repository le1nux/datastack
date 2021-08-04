from abc import ABC
from data_stack.io.storage_connectors import StorageConnector
from data_stack.dataset.iterator import DatasetIteratorIF, InformedDatasetIteratorIF, InformedDatasetIterator, CombinedDatasetIterator, \
    DatasetIteratorView, InMemoryDatasetIterator
from typing import Tuple, List, Dict, Any
from data_stack.dataset.meta import IteratorMeta, DatasetMeta
import random


class BaseDatasetFactory(ABC):

    def __init__(self, storage_connector: StorageConnector = None):
        self.storage_connector = storage_connector

    def get_dataset_iterator(self, config: Dict[str, Any] = None) -> Tuple[DatasetIteratorIF, IteratorMeta]:
        raise NotImplementedError


class HigherOrderDatasetFactory:
    @staticmethod
    def get_combined_dataset_iterator(iterators: List[DatasetIteratorIF]) -> DatasetIteratorIF:
        return CombinedDatasetIterator(iterators)

    @staticmethod
    def get_dataset_iterator_view(iterator: DatasetIteratorIF, indices: List[int], view_tags: Dict[str, Any]) -> DatasetIteratorIF:
        return DatasetIteratorView(iterator, indices, view_tags)


class InformedDatasetFactory:

    @staticmethod
    def get_dataset_iterator(iterator: DatasetIteratorIF, meta: DatasetMeta) -> InformedDatasetIteratorIF:
        return InformedDatasetIterator(iterator, meta)

    @staticmethod
    def get_combined_dataset_iterator(iterators: List[DatasetIteratorIF], meta: DatasetMeta) -> InformedDatasetIteratorIF:
        iterator = HigherOrderDatasetFactory.get_combined_dataset_iterator(iterators)
        return InformedDatasetIterator(iterator, meta)

    @staticmethod
    def get_dataset_iterator_view(iterator: DatasetIteratorIF, meta: DatasetMeta, indices: List[int],
                                  view_tags: Dict[str, Any] = None) -> InformedDatasetIteratorIF:
        iterator = HigherOrderDatasetFactory.get_dataset_iterator_view(iterator, indices, view_tags)
        return InformedDatasetIterator(iterator, meta)

    @staticmethod
    def get_in_memory_dataset_iterator(iterator: DatasetIteratorIF, meta: DatasetMeta) -> InformedDatasetIteratorIF:
        in_memory_iterator = InMemoryDatasetIterator(iterator)
        return InformedDatasetIterator(in_memory_iterator, meta)

    @staticmethod
    def get_shuffled_dataset_iterator(iterator: DatasetIteratorIF, meta: DatasetMeta, seed: int) -> InformedDatasetIteratorIF:
        random_gen = random.Random(seed)
        indices = list(range(len(iterator)))
        random_gen.shuffle(indices)
        iterator_view = InformedDatasetFactory.get_dataset_iterator_view(iterator, meta, indices)
        return iterator_view
