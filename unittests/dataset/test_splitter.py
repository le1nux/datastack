import pytest
from data_stack.dataset.iterator import DatasetIteratorIF, SequenceDatasetIterator
from typing import List
from data_stack.dataset.splitter import RandomSplitterImpl, Splitter
from data_stack.dataset.meta import DatasetMeta, MetaFactory


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

    def test_random_splitter(self, ratios: List[int], dataset_iterator: DatasetIteratorIF):
        splitter_impl = RandomSplitterImpl(ratios=ratios)
        splitter = Splitter(splitter_impl)
        iterator_splits = splitter.split(dataset_iterator)

        assert sorted([i for split in iterator_splits for i in split]) == sorted(dataset_iterator)

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
