from typing import Dict, Tuple, Any
from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.dataset.iterator import DatasetIteratorIF
from data_stack.exception import DatasetNotFoundError
from data_stack.dataset.meta import IteratorMeta


class DatasetRepository:
    def __init__(self):
        self._base_factory_dict: Dict[str, BaseDatasetFactory] = dict()

    def get(self, identifier: str, config: Dict[str, Any] = None) -> Tuple[DatasetIteratorIF, IteratorMeta]:
        if identifier not in self._base_factory_dict.keys():
            raise DatasetNotFoundError
        return self._base_factory_dict[identifier].get_dataset_iterator(config)

    def register(self, identifier: str, dataset_factory: BaseDatasetFactory):
        self._base_factory_dict[identifier] = dataset_factory
