import pytest
from typing import List, Dict, Any, Tuple
from data_hub.dataset.iterator import DatasetIteratorIF, DatasetIterator, DatasetIteratorView, CombinedDatasetIterator
from itertools import chain


class TestIterator:
    @pytest.fixture
    def target_position(self) -> int:
        return 1

    @pytest.fixture
    def sequences(self) -> List[Tuple[Any]]:
        # returns tuples of format [sample_sequence, target_sequence, tag_sequence]
        return [
            [(1, 2, 3), (3, 2, 2), (2, 2, 6), (5, 2, 5), (1, 5, 3)],
            [1, 0, 2, 3, 3],
            ["b", "a", "c", "d", "d"]
        ]

    @pytest.fixture
    def dataset_iterator(self, sequences) -> DatasetIteratorIF:
        return DatasetIterator(dataset_sequences=sequences,
                               dataset_name="TEST DATASET",
                               dataset_tag="train",
                               sample_pos=0,
                               target_pos=1,
                               tag_pos=2)

    @pytest.fixture
    def dataset_view_indices(self) -> List[int]:
        return [0, 2]

    @pytest.fixture
    def dataset_iterator_view(self, dataset_iterator, dataset_view_indices) -> DatasetIteratorIF:
        return DatasetIteratorView(dataset_iterator, indices=dataset_view_indices, dataset_tag="bla")

    def test_dataset_iterator(self, dataset_iterator: DatasetIteratorIF, sequences):
        for orig_sample, iterator_sample in zip(zip(*sequences), dataset_iterator):
            assert orig_sample[dataset_iterator.sample_pos] == iterator_sample[dataset_iterator.sample_pos]
            assert orig_sample[dataset_iterator.target_pos] == iterator_sample[dataset_iterator.target_pos]
            assert orig_sample[dataset_iterator.tag_pos] == iterator_sample[dataset_iterator.tag_pos]

    def test_dataset_iterator_view(self, dataset_iterator_view: DatasetIteratorIF, dataset_view_indices, sequences):
        for orig_sample, iterator_sample in zip([[s[i] for s in sequences] for i in dataset_view_indices], dataset_iterator_view):
            assert orig_sample[dataset_iterator_view.sample_pos] == iterator_sample[dataset_iterator_view.sample_pos]
            assert orig_sample[dataset_iterator_view.target_pos] == iterator_sample[dataset_iterator_view.target_pos]
            assert orig_sample[dataset_iterator_view.tag_pos] == iterator_sample[dataset_iterator_view.tag_pos]

    def test_dataset_iterator_combined(self, sequences):
        iterator_1 = DatasetIterator(dataset_sequences=sequences,
                                     dataset_name="TEST DATASET",
                                     dataset_tag="train",
                                     sample_pos=0,
                                     target_pos=1,
                                     tag_pos=2)

        iterator_2 = DatasetIterator(dataset_sequences=sequences,
                                     dataset_name="TEST DATASET",
                                     dataset_tag="train",
                                     sample_pos=0,
                                     target_pos=1,
                                     tag_pos=2)
        sequences_combined = chain(sequences, sequences)
        combined_iterator = CombinedDatasetIterator(iterators=[iterator_1, iterator_2], dataset_name="combined", dataset_tag="abc")

        for orig_sample, iterator_sample in zip(zip(*sequences_combined), combined_iterator):
            assert orig_sample[combined_iterator.sample_pos] == iterator_sample[combined_iterator.sample_pos]
            assert orig_sample[combined_iterator.target_pos] == iterator_sample[combined_iterator.target_pos]
            assert orig_sample[combined_iterator.tag_pos] == iterator_sample[combined_iterator.tag_pos]
