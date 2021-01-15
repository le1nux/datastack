from data_stack.dataset.iterator import DatasetIteratorIF, DatasetIteratorView
from typing import List
from abc import ABC, abstractmethod
import random


class SplitterFactory:

    @staticmethod
    def get_random_splitter(ratios: List[float], seed: int):
        return Splitter(splitter_impl=RandomSplitterImpl(ratios, seed=seed))


class SplitterIF(ABC):

    @abstractmethod
    def split(self, dataset_iterator: DatasetIteratorIF) -> List[DatasetIteratorIF]:
        raise NotImplementedError


class Splitter(SplitterIF):

    def __init__(self, splitter_impl: "SplitterIF"):
        self.splitter_impl = splitter_impl

    def split(self, dataset_iterator: DatasetIteratorIF) -> List[DatasetIteratorIF]:
        return self.splitter_impl.split(dataset_iterator)


class RandomSplitterImpl(SplitterIF):

    def __init__(self, ratios: List[float], seed: int = 1):
        self.ratios = ratios
        self.random_gen = random.Random(seed)

    def split(self, dataset_iterator: DatasetIteratorIF) -> List[DatasetIteratorIF]:
        dataset_length = len(dataset_iterator)
        splits_indices = self._determine_split_indices(dataset_length, self.ratios)

        return [DatasetIteratorView(dataset_iterator, split_indices) for split_indices in splits_indices]

    def _determine_split_indices(self, dataset_length: int, ratios: List[int]) -> List[List[int]]:
        def ratio_to_index(ratio: float) -> int:
            return int(ratio*dataset_length)

        indices = list(range(dataset_length))
        self.random_gen.shuffle(indices)
        lower = 0
        upper = 0
        split_indices: List[List[int]] = []
        for ratio in ratios:
            upper = upper + ratio_to_index(ratio)
            split_indices.append(indices[lower: upper])
            lower = upper
        # if we don't have a round split, we add the remaining samples to the last split.
        split_indices[-1] = split_indices[-1] + indices[upper:]
        return split_indices
