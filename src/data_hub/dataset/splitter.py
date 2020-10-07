from data_hub.dataset.iterator import DatasetIteratorIF, DatasetIteratorView
from typing import List
from abc import ABC, abstractmethod
import random


class SplitterFactory:

    @staticmethod
    def get_random_splitter(ratios: List[float]):
        return Splitter(splitter_impl=RandomSplitterImpl(ratios))


class SplitterIF(ABC):

    @abstractmethod
    def split(dataset_iterator: DatasetIteratorIF) -> List[DatasetIteratorIF]:
        raise NotImplementedError


class Splitter(SplitterIF):

    def __init__(self, splitter_impl: "SplitterIF"):
        self.splitter_impl = splitter_impl

    def split(self, dataset_iterator: DatasetIteratorIF) -> List[DatasetIteratorIF]:
        return self.splitter_impl.split(dataset_iterator)


class RandomSplitterImpl(SplitterIF):

    def __init__(self, ratios: List[float], split_names: List[str] = None, seed: int = 1):
        self.ratios = ratios
        self.split_names = split_names if split_names is not None else [None]*len(ratios)
        random.seed(seed)

    def split(self, dataset_iterator: DatasetIteratorIF) -> List[DatasetIteratorIF]:
        dataset_length = len(dataset_iterator)
        splits_indices = RandomSplitterImpl._determine_split_indices(dataset_length, self.ratios)
        return [DatasetIteratorView(dataset_iterator, split_indices, split_name)
                for split_indices, split_name in zip(splits_indices, self.split_names)]

    @staticmethod
    def _determine_split_indices(dataset_length: int, ratios: List[int]) -> List[List[int]]:
        def ratio_to_index(ratio: float) -> int:
            return int(ratio*dataset_length)

        indices = list(range(dataset_length))
        random.shuffle(indices)
        lower = 0
        upper = 0
        split_indices: List[List[int]] = []
        for ratio in ratios:
            upper = upper + ratio_to_index(ratio)
            split_indices.append(indices[lower: upper])
            lower = upper
        return split_indices
