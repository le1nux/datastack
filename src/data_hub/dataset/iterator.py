from abc import ABC, abstractmethod


class DatasetIteratorIF(ABC):

    def __init__(self, dataset_name: str):
        self._dataset_name = dataset_name

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int):
        raise NotImplementedError

    @property
    def dataset_name(self) -> str:
        return self._dataset_name
