from data_stack.dataset.iterator import DatasetIteratorIF, DatasetIteratorView
from typing import List, Tuple
from abc import ABC, abstractmethod
import random
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
import numpy as np


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


class NestedCVSplitterImpl(SplitterIF):

    def __init__(self,
                 num_outer_loop_folds: int = 5,
                 num_inner_loop_folds: int = 2,
                 inner_stratification: bool = True,
                 outer_stratification: bool = True,
                 target_pos: int = 1,
                 shuffle: bool = True,
                 seed: int = 1):
        self.num_outer_loop_folds = num_outer_loop_folds
        self.num_inner_loop_folds = num_inner_loop_folds
        self.random_state = np.random.RandomState(seed=seed)
        self.target_pos = target_pos

        if inner_stratification:
            self.inner_splitter = StratifiedKFold(n_splits=num_inner_loop_folds, shuffle=shuffle, random_state=self.random_state)
        else:
            self.inner_splitter = KFold(n_splits=num_inner_loop_folds, shuffle=shuffle, random_state=self.random_state)
        if outer_stratification:
            self.outer_splitter = StratifiedKFold(n_splits=num_outer_loop_folds, shuffle=shuffle, random_state=self.random_state)
        else:
            self.outer_splitter = KFold(n_splits=num_outer_loop_folds, shuffle=shuffle, random_state=self.random_state)

    def split(self, dataset_iterator: DatasetIteratorIF) -> Tuple[List[DatasetIteratorIF], List[List[DatasetIteratorIF]]]:
        # create outer loop folds
        targets = [sample[self.target_pos] for sample in dataset_iterator]
        folds_indices = [fold[1] for fold in self.outer_splitter.split(X=np.zeros(len(targets)), y=targets)]
        outer_folds = [DatasetIteratorView(dataset_iterator, fold_indices) for fold_indices in folds_indices]
        # create inner loop folds
        inner_folds_list = []  # contains [inner folds of outer_fold_1, inner folds of outer_fold_2 ...]
        for iterator in outer_folds:
            targets = [sample[self.target_pos] for sample in iterator]
            folds_indices = [fold[1] for fold in self.inner_splitter.split(X=np.zeros(len(targets)), y=targets)]
            inner_folds = [DatasetIteratorView(iterator, fold_indices) for fold_indices in folds_indices]
            inner_folds_list.append(inner_folds)
        return outer_folds, inner_folds_list
