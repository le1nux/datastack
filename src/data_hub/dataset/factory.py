from abc import ABC
from data_hub.io.storage_connectors import StorageConnector
from data_hub.dataset.iterator import DatasetIteratorIF, InformedDatasetIteratorIF, InformedDatasetIterator, CombinedDatasetIterator
from typing import Tuple, List
from data_hub.dataset.meta import IteratorMeta, DatasetMeta


class BaseDatasetFactory(ABC):

    def __init__(self, storage_connector: StorageConnector):
        self.storage_connector = storage_connector

    def get_dataset_iterator(self, split: str = None) -> Tuple[DatasetIteratorIF, IteratorMeta]:
        raise NotImplementedError


class HigherOrderDatasetFactory:
    @staticmethod
    def get_combined_dataset_iterator(iterators: List[DatasetIteratorIF]) -> InformedDatasetIteratorIF:
        return CombinedDatasetIterator(iterators)


class InformedDatasetFactory:

    @staticmethod
    def get_dataset_iterator(iterator: DatasetIteratorIF, meta: DatasetMeta) -> InformedDatasetIteratorIF:
        return InformedDatasetIterator(iterator, meta)

    @staticmethod
    def get_combined_dataset_iterator(iterators: List[DatasetIteratorIF], meta: DatasetMeta) -> InformedDatasetIteratorIF:
        iterator = HigherOrderDatasetFactory.get_combined_dataset_iterator(iterators)
        return InformedDatasetIterator(iterator, meta)
