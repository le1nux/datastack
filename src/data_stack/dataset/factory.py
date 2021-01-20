from abc import ABC
from data_stack.io.storage_connectors import StorageConnector
from data_stack.dataset.iterator import DatasetIteratorIF, InformedDatasetIteratorIF, InformedDatasetIterator, CombinedDatasetIterator, DatasetIteratorView
from typing import Tuple, List, Dict, Any
from data_stack.dataset.meta import IteratorMeta, DatasetMeta


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
    def get_dataset_iterator_view(iterator: DatasetIteratorIF, indices: List[int]) -> DatasetIteratorIF:
        return DatasetIteratorView(iterator, indices)


class InformedDatasetFactory:

    @staticmethod
    def get_dataset_iterator(iterator: DatasetIteratorIF, meta: DatasetMeta) -> InformedDatasetIteratorIF:
        return InformedDatasetIterator(iterator, meta)

    @staticmethod
    def get_combined_dataset_iterator(iterators: List[DatasetIteratorIF], meta: DatasetMeta) -> InformedDatasetIteratorIF:
        iterator = HigherOrderDatasetFactory.get_combined_dataset_iterator(iterators)
        return InformedDatasetIterator(iterator, meta)

    @staticmethod
    def get_dataset_iterator_view(iterator: DatasetIteratorIF, meta: DatasetMeta, indices: List[int]) -> InformedDatasetIteratorIF:
        iterator = HigherOrderDatasetFactory.get_dataset_iterator_view(iterator, indices)
        return InformedDatasetIterator(iterator, meta)
