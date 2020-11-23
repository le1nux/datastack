from copy import deepcopy
from abc import ABC, abstractmethod
from dataclasses import dataclass


class IdentifiyingInfoIF(ABC):

    @property
    @abstractmethod
    def identifier(self) -> str:
        raise NotImplementedError


@dataclass
class IteratorMeta:
    sample_pos: int = 0
    target_pos: int = 1
    tag_pos: int = 2


class DatasetMetaIF(IdentifiyingInfoIF):

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def dataset_tag(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def sample_pos(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def target_pos(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def tag_pos(self) -> int:
        raise NotImplementedError


class DatasetMeta(DatasetMetaIF):

    def __init__(self, iterator_meta: IteratorMeta, identifier: str = None, dataset_name: str = None, dataset_tag: str = None):
        self._identifier = identifier
        self._dataset_name = dataset_name
        self._dataset_tag = dataset_tag
        self._iterator_meta = iterator_meta

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def dataset_tag(self) -> str:
        return self._dataset_tag

    @property
    def sample_pos(self) -> int:
        return self._iterator_meta.sample_pos

    @property
    def target_pos(self) -> int:
        return self._iterator_meta.target_pos

    @property
    def tag_pos(self) -> int:
        return self._iterator_meta.tag_pos


class WrappedDatasetMeta(DatasetMetaIF):

    def __init__(self, dataset_meta: DatasetMetaIF, identifier: str = None, dataset_name: str = None, dataset_tag: str = None):
        self._dataset_meta = dataset_meta
        self._identifier = identifier
        self._dataset_name = dataset_name
        self._dataset_tag = dataset_tag

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def dataset_name(self) -> str:
        if self._dataset_name:
            return self._dataset_name
        else:
            return self._dataset_meta.dataset_name

    @property
    def dataset_tag(self) -> str:
        if self._dataset_tag:
            return self._dataset_tag
        else:
            return self._dataset_meta.dataset_tag

    @property
    def sample_pos(self) -> int:
        return self._dataset_meta.sample_pos

    @property
    def target_pos(self) -> int:
        return self._dataset_meta.target_pos

    @property
    def tag_pos(self) -> int:
        return self._dataset_meta.tag_pos


class MetaFactory:

    @staticmethod
    def get_iterator_meta(sample_pos: int, target_pos: int, tag_pos: int) -> IteratorMeta:
        return IteratorMeta(sample_pos=sample_pos,
                            target_pos=target_pos,
                            tag_pos=tag_pos)

    @staticmethod
    def get_dataset_meta(identifier: str = None, dataset_name: str = None, dataset_tag: str = None, iterator_meta: IteratorMeta = None) -> DatasetMetaIF:
        return DatasetMeta(identifier=identifier,
                           dataset_name=dataset_name,
                           dataset_tag=dataset_tag,
                           iterator_meta=iterator_meta)

    @staticmethod
    def copy_dataset_meta(dataset_meta: DatasetMeta) -> DatasetMeta:
        return deepcopy(dataset_meta)

    @staticmethod
    def get_dataset_meta_from_existing(dataset_meta: DatasetMeta, identifier: str = None, dataset_name: str = None, dataset_tag: str = None) -> DatasetMetaIF:
        dataset_meta = MetaFactory.copy_dataset_meta(dataset_meta)

        return WrappedDatasetMeta(dataset_meta=dataset_meta,
                                  identifier=identifier,
                                  dataset_name=dataset_name,
                                  dataset_tag=dataset_tag)
