import pytest
from data_stack.dataset.iterator import DatasetIteratorIF, SequenceDatasetIterator
from typing import List
from data_stack.dataset.splitter import RandomSplitterImpl, Splitter, NestedCVSplitterImpl
from data_stack.dataset.meta import DatasetMeta, MetaFactory
import collections


class TestSplitter:
    @pytest.fixture
    def ratios(self) -> List[int]:
        return [0.3, 0.3, 0.2, 0.1, 0.1]

    @pytest.fixture
    def dataset_meta(self) -> DatasetMeta:
        iterator_meta = MetaFactory.get_iterator_meta(sample_pos=0, target_pos=1, tag_pos=2)
        return MetaFactory.get_dataset_meta(identifier="identifier_1",
                                            dataset_name="TEST DATASET",
                                            dataset_tag="train",
                                            iterator_meta=iterator_meta)

    @pytest.fixture
    def dataset_iterator(self) -> DatasetIteratorIF:
        return SequenceDatasetIterator(dataset_sequences=[list(range(10)), list(range(10))])

    @pytest.fixture
    def big_dataset_iterator(self) -> DatasetIteratorIF:
        targets = [j for i in range(10) for j in range(9)] + [10]*1000
        samples = [0]*len(targets)
        return SequenceDatasetIterator(dataset_sequences=[samples, targets])

    def test_random_splitter(self, ratios: List[int], dataset_iterator: DatasetIteratorIF):
        splitter_impl = RandomSplitterImpl(ratios=ratios, seed=100)
        splitter = Splitter(splitter_impl)
        iterator_splits = splitter.split(dataset_iterator)

        assert sorted([i for split in iterator_splits for i in split]) == sorted(dataset_iterator)

    @pytest.mark.parametrize(
        "num_outer_loop_folds, num_inner_loop_folds, inner_stratification, outer_stratification, shuffle",
        [(5, 2, True, True, False), (5, 2, True, True, True), (5, 2, False, False, True), (5, 2, False, False, False)],
    )
    def test_nested_cv_splitter(self, num_outer_loop_folds: int, num_inner_loop_folds: int, inner_stratification: bool,
                                outer_stratification: bool, shuffle: bool, big_dataset_iterator: DatasetIteratorIF):
        splitter_impl = NestedCVSplitterImpl(num_outer_loop_folds=num_outer_loop_folds,
                                             num_inner_loop_folds=num_inner_loop_folds,
                                             inner_stratification=inner_stratification,
                                             outer_stratification=outer_stratification,
                                             shuffle=shuffle)
        splitter = Splitter(splitter_impl)
        outer_folds, inner_folds = splitter.split(big_dataset_iterator)
        # make sure that outer folds have no intersection
        for i in range(len(outer_folds)):
            for j in range(len(outer_folds)):
                if i != j:
                    # makes sure there is no intersection
                    assert len(set(outer_folds[i].indices).intersection(set(outer_folds[j].indices))) == 0
        # make sure that inner folds have no intersection
        for i in range(len(inner_folds)):
            for j in range(len(inner_folds[i])):
                for k in range(len(inner_folds[i])):
                    if j != k:
                        # makes sure there is no intersection
                        assert len(set(inner_folds[i][j].indices).intersection(set(inner_folds[i][k].indices))) == 0
        # test stratification
        if outer_stratification:
            class_counts = dict(collections.Counter([t for _, t in big_dataset_iterator]))
            class_counts_per_fold = {target_class: int(count/num_outer_loop_folds) for target_class, count in class_counts.items()}
            for fold in outer_folds:
                fold_class_counts = dict(collections.Counter([t for _, t in fold]))
                for key in list(class_counts_per_fold.keys()) + list(fold_class_counts.keys()):
                    assert class_counts_per_fold[key] == fold_class_counts[key]

        if inner_stratification:
            for i in range(len(inner_folds)):
                class_counts = dict(collections.Counter([t for _, t in outer_folds[i]]))
                class_counts_per_fold = {target_class: int(count*(num_outer_loop_folds-1)/num_inner_loop_folds) for target_class, count in class_counts.items()}
                for fold in inner_folds[i]:
                    fold_class_counts = dict(collections.Counter([t for _, t in fold]))
                    for key in list(class_counts_per_fold.keys()) + list(fold_class_counts.keys()):
                        assert class_counts_per_fold[key] == fold_class_counts[key]

    def test_seeding(self):
        ratios = [0.4, 0.6]
        dataset_length = 100
        splitter_impl_1 = RandomSplitterImpl(ratios=ratios, seed=1)
        splitter_impl_2 = RandomSplitterImpl(ratios=ratios, seed=1)
        splitter_impl_3 = RandomSplitterImpl(ratios=ratios, seed=2)

        splits_indices_1 = splitter_impl_1._determine_split_indices(dataset_length=dataset_length, ratios=ratios)
        splits_indices_2 = splitter_impl_2._determine_split_indices(dataset_length=dataset_length, ratios=ratios)
        splits_indices_3 = splitter_impl_3._determine_split_indices(dataset_length=dataset_length, ratios=ratios)

        assert splits_indices_1[0] == splits_indices_2[0] and splits_indices_1[1] == splits_indices_2[1]
        assert splits_indices_1[0] != splits_indices_3[0] and splits_indices_1[1] != splits_indices_3[1]
