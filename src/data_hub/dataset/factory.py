from abc import ABC
from data_hub.io.storage_connectors import StorageConnector
from data_hub.dataset.iterator import DatasetIteratorIF
from dataclasses import dataclass


class DatasetFactory(ABC):

    def __init__(self, storage_connector: StorageConnector):
        self.storage_connector = storage_connector

    def get_dataset_iterator(self, split: str = None) -> DatasetIteratorIF:
        raise NotImplementedError
