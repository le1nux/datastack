from copy import deepcopy
from abc import ABC, abstractmethod


class DatasetMetaInformationIF(ABC):

    def __init__(self, dataset_name: str = None, dataset_tag: str = None, sample_pos: int = 0, target_pos: int = 1, tag_pos: int = 2):
        self._dataset_name = dataset_name
        self._dataset_tag = dataset_tag
        self._sample_pos = sample_pos
        self._target_pos = target_pos
        self._tag_pos = tag_pos

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


class DatasetMetaInformation(DatasetMetaInformationIF):

    def __init__(self, dataset_name: str = None, dataset_tag: str = None, sample_pos: int = 0, target_pos: int = 1, tag_pos: int = 2):
        self._dataset_name = dataset_name
        self._dataset_tag = dataset_tag
        self._sample_pos = sample_pos
        self._target_pos = target_pos
        self._tag_pos = tag_pos

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def dataset_tag(self) -> str:
        return self._dataset_tag

    @property
    def sample_pos(self) -> int:
        return self._sample_pos

    @property
    def target_pos(self) -> int:
        return self._target_pos

    @property
    def tag_pos(self) -> int:
        return self._tag_pos


class WrappedDatasetMetaInformation(DatasetMetaInformationIF):

    def __init__(self, dataset_meta_information: DatasetMetaInformationIF, dataset_name: str = None, dataset_tag: str = None, sample_pos: int = 0, target_pos: int = 1, tag_pos: int = 2):
        self._dataset_meta_information = dataset_meta_information
        self._dataset_name = dataset_name
        self._dataset_tag = dataset_tag
        self._sample_pos = sample_pos
        self._target_pos = target_pos
        self._tag_pos = tag_pos

    @property
    def dataset_name(self) -> str:
        if self._dataset_name:
            return self._dataset_name
        else:
            return self._dataset_meta_information._dataset_name

    @property
    def dataset_tag(self) -> str:
        if self._dataset_tag:
            return self._dataset_tag
        else:
            return self._dataset_meta_information._dataset_tag

    @property
    def sample_pos(self) -> int:
        if self._sample_pos:
            return self._sample_pos
        else:
            return self._dataset_meta_information._sample_pos

    @property
    def target_pos(self) -> int:
        if self._target_pos:
            return self._target_pos
        else:
            return self._dataset_meta_information._target_pos

    @property
    def tag_pos(self) -> int:
        if self._tag_pos:
            return self._tag_pos
        else:
            return self._dataset_meta_information._tag_pos


class DatasetMetaInformationFactory:

    @staticmethod
    def get_dataset_meta_informmation(dataset_name: str = None, dataset_tag: str = None, sample_pos: int = 0, target_pos: int = 1, tag_pos: int = 2) -> DatasetMetaInformation:
        return DatasetMetaInformation(dataset_name=dataset_name,
                                      dataset_tag=dataset_tag,
                                      sample_pos=sample_pos,
                                      target_pos=target_pos,
                                      tag_pos=tag_pos)

    @staticmethod
    def get_dataset_meta_informmation_copy(dataset_meta_informmation: DatasetMetaInformation) -> DatasetMetaInformation:
        return deepcopy(dataset_meta_informmation)

    @staticmethod
    def get_dataset_meta_informmation(dataset_name: str = None, dataset_tag: str = None, sample_pos: int = 0, target_pos: int = 1, tag_pos: int = 2) -> DatasetMetaInformation:
        return DatasetMetaInformation(dataset_name=dataset_name,
                                      dataset_tag=dataset_tag,
                                      sample_pos=sample_pos,
                                      target_pos=target_pos,
                                      tag_pos=tag_pos)

    @staticmethod
    def get_dataset_meta_informmation_from_existing(dataset_meta_information: DatasetMetaInformation, dataset_name: str = None, dataset_tag: str = None, sample_pos: int = 0, target_pos: int = 1, tag_pos: int = 2) -> DatasetMetaInformation:
        dataset_meta_information = DatasetMetaInformationFactory.get_dataset_meta_informmation_copy(dataset_meta_information)
        return WrappedDatasetMetaInformation(dataset_meta_information=dataset_meta_information,
                                             dataset_name=dataset_name,
                                             dataset_tag=dataset_tag,
                                             sample_pos=sample_pos,
                                             target_pos=target_pos,
                                             tag_pos=tag_pos)
