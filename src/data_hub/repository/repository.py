from typing import Dict
from data_hub.dataset.factory import BaseDatasetFactory
from data_hub.dataset.iterator import DatasetIteratorIF
from data_hub.exception import DatasetNotFoundError


class DatasetRepository:
    def __init__(self):
        self.dataset_dict: Dict[str, BaseDatasetFactory] = dict()

    def get(self, identifier: str, split: str) -> DatasetIteratorIF:
        if identifier not in self.dataset_dict.keys():
            raise DatasetNotFoundError
        return self.dataset_dict[identifier].get_dataset_iterator(split)

    def register(self, identifier: str, dataset_factory: BaseDatasetFactory):
        self.dataset_dict[identifier] = dataset_factory
