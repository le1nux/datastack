from data_stack.dataset.iterator import DatasetIteratorIF, DatasetIteratorView
from typing import List, Tuple, Any, Optional
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import random
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np


class SplitterFactory:

    @staticmethod
    def get_random_splitter(ratios: List[float], seed: int):
        return Splitter(splitter_impl=RandomSplitterImpl(ratios, seed=seed))

    @staticmethod
    def get_stratified_splitter(ratios: List[float], seed: int):
        return Splitter(splitter_impl=StratifiedSplitterImpl(ratios, seed=seed))

    @staticmethod
    def get_nested_cv_splitter(num_outer_loop_folds: int = 5, num_inner_loop_folds: int = 2,
                               inner_stratification: bool = True, outer_stratification: bool = True,
                               target_pos: int = 1, shuffle: bool = True, seed: int = 1):
        splitter_impl = NestedCVSplitterImpl(num_outer_loop_folds=num_outer_loop_folds,
                                             num_inner_loop_folds=num_inner_loop_folds,
                                             inner_stratification=inner_stratification,
                                             outer_stratification=outer_stratification,
                                             target_pos=target_pos,
                                             shuffle=shuffle,
                                             seed=seed)
        return Splitter(splitter_impl=splitter_impl)

    @staticmethod
    def get_cv_splitter(num_folds: int = 5, stratification: bool = True, target_pos: int = 1, shuffle: bool = True, seed: int = 1):
        splitter_impl = CVSplitterImpl(num_folds=num_folds,
                                       stratification=stratification,
                                       target_pos=target_pos,
                                       shuffle=shuffle,
                                       seed=seed)
        return Splitter(splitter_impl=splitter_impl)


class SplitterIF(ABC):

    @abstractmethod
    def split(self, dataset_iterator: DatasetIteratorIF) -> List[DatasetIteratorIF]:
        raise NotImplementedError

    @abstractmethod
    def get_indices(self, dataset_iterator: DatasetIteratorIF) -> Any:
        raise NotImplementedError


class Splitter(SplitterIF):

    def __init__(self, splitter_impl: "SplitterIF"):
        self.splitter_impl = splitter_impl

    def split(self, dataset_iterator: DatasetIteratorIF) -> List[DatasetIteratorIF]:
        return self.splitter_impl.split(dataset_iterator)

    def get_indices(self, dataset_iterator: DatasetIteratorIF) -> Any:
        return self.splitter_impl.get_indices(dataset_iterator)


class RandomSplitterImpl(SplitterIF):

    def __init__(self, ratios: List[float], seed: int = 1):
        self.ratios = ratios
        self.random_gen = random.Random(seed)

    def split(self, dataset_iterator: DatasetIteratorIF) -> List[DatasetIteratorView]:
        dataset_length = len(dataset_iterator)
        splits_indices = self._determine_split_indices(dataset_length, self.ratios)

        return [DatasetIteratorView(dataset_iterator, split_indices) for split_indices in splits_indices]

    def _determine_split_indices(self, dataset_length: int, ratios: List[float]) -> List[List[int]]:
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

    def get_indices(self, dataset_iterator: DatasetIteratorIF) -> List[List[int]]:
        dataset_length = len(dataset_iterator)
        splits_indices = self._determine_split_indices(dataset_length, self.ratios)
        return splits_indices


class StratifiedSplitterImpl(SplitterIF):

    def __init__(self, ratios: List[float], seed: Optional[int] = None):
        self.ratios = ratios
        self.seed = seed

    def split(self, dataset_iterator: DatasetIteratorIF) -> List[DatasetIteratorIF]:
        dataset_length = len(dataset_iterator)
        splits_indices = self._determine_split_indices(dataset_length, self.ratios, dataset_iterator)

        return [DatasetIteratorView(dataset_iterator, split_indices) for split_indices in splits_indices]

    def _determine_split_indices(self, dataset_length: int, ratios: List[float], dataset_iterator: DatasetIteratorIF)\
            -> List[List[int]]:
        indices_remaining = list(range(dataset_length))
        initial_length = len(indices_remaining)
        targets_remaining = [sample[1] for sample in dataset_iterator]

        split_indices: List[List[int]] = []

        # split the data set until the desired number of splits is reached
        for split_ratio in ratios[:-1]:
            indices_split, indices_remaining, _, targets_remaining = train_test_split(indices_remaining,
                                                                                      targets_remaining,
                                                                                      train_size=int(initial_length*split_ratio),
                                                                                      stratify=targets_remaining, random_state=self.seed, shuffle=True)
            split_indices.append(indices_split)
        # any remaining indices are added to the last split
        split_indices.append(indices_remaining)
        return split_indices

    def get_indices(self, dataset_iterator: DatasetIteratorIF) -> List[List[int]]:
        dataset_length = len(dataset_iterator)
        splits_indices = self._determine_split_indices(dataset_length, self.ratios, dataset_iterator)
        return splits_indices


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
        self.random_state = np.random.RandomState(seed=seed) if shuffle else None
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
        targets = np.array([sample[self.target_pos] for sample in dataset_iterator])
        outer_folds_indices = [fold[1] for fold in self.outer_splitter.split(X=np.zeros(len(targets)), y=targets)]
        outer_fold_iterators = [DatasetIteratorView(dataset_iterator, fold_indices) for fold_indices in outer_folds_indices]
        # create inner loop folds
        inner_folds_iterators_list = []  # contains [inner folds of outer_fold_1, inner folds of outer_fold_2 ...]
        for outer_fold_id in range(len(outer_fold_iterators)):
            # concat the indices of the splits which belong to the train splits
            train_split_ids = [i for i in range(len(outer_folds_indices)) if i != outer_fold_id]
            outer_train_fold_indices = np.array([indice for i in train_split_ids for indice in outer_folds_indices[i]])
            inner_targets = targets[outer_train_fold_indices]
            inner_folds_indices = [outer_train_fold_indices[inner_fold[1]]
                                   for inner_fold in self.inner_splitter.split(X=np.zeros(len(inner_targets)), y=inner_targets)]
            inner_folds = [DatasetIteratorView(dataset_iterator, fold_indices) for fold_indices in inner_folds_indices]
            inner_folds_iterators_list.append(inner_folds)
        return outer_fold_iterators, inner_folds_iterators_list

    def get_indices(self, dataset_iterator: DatasetIteratorIF) -> Tuple[List[List[int]], List[List[int]]]:
        outer_folds, inner_folds_list = self.split(dataset_iterator)
        outer_folds_indices = [fold.indices.tolist() for fold in outer_folds]
        inner_fold_indices = [[fold.indices.tolist() for fold in folds] for folds in inner_folds_list]

        return outer_folds_indices, inner_fold_indices


class CVSplitterImpl(SplitterIF):

    def __init__(self,
                 num_folds: int = 5,
                 stratification: bool = True,
                 target_pos: int = 1,
                 shuffle: bool = True,
                 seed: int = 1):
        self.num_folds = num_folds
        self.random_state = np.random.RandomState(seed=seed) if shuffle else None
        self.target_pos = target_pos

        if stratification:
            self.splitter = StratifiedKFold(n_splits=num_folds, shuffle=shuffle, random_state=self.random_state)
        else:
            self.splitter = KFold(n_splits=num_folds, shuffle=shuffle, random_state=self.random_state)

    def split(self, dataset_iterator: DatasetIteratorIF) -> List[DatasetIteratorView]:
        targets = np.array([sample[self.target_pos] for sample in dataset_iterator])
        folds_indices = [fold[1].tolist() for fold in self.splitter.split(X=np.zeros(len(targets)), y=targets)]
        fold_iterators = [DatasetIteratorView(dataset_iterator, fold_indices) for fold_indices in folds_indices]
        return fold_iterators

    def get_indices(self, dataset_iterator: DatasetIteratorIF) -> List[List[int]]:
        folds = self.split(dataset_iterator)
        folds_indices = [fold.indices for fold in folds]
        return folds_indices
