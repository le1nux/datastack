import pytest
from typing import List, Any, Tuple
from data_stack.dataset.iterator import DatasetIteratorIF, SequenceDatasetIterator, DatasetIteratorView, \
    CombinedDatasetIterator, InMemoryDatasetIterator
from data_stack.dataset.factory import InformedDatasetFactory
from itertools import chain
from data_stack.dataset.meta import MetaFactory


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
        return SequenceDatasetIterator(dataset_sequences=sequences)

    @pytest.fixture
    def dataset_view_indices(self) -> List[int]:
        return [0, 2]

    @pytest.fixture
    def dataset_iterator_view(self, dataset_iterator, dataset_view_indices) -> DatasetIteratorIF:
        return DatasetIteratorView(dataset_iterator, indices=dataset_view_indices)

    def test_dataset_iterator(self, dataset_iterator: DatasetIteratorIF, sequences):
        for orig_sample, iterator_sample in zip(zip(*sequences), dataset_iterator):
            assert orig_sample[0] == iterator_sample[0]
            assert orig_sample[1] == iterator_sample[1]
            assert orig_sample[2] == iterator_sample[2]

    def test_dataset_iterator_view(self, dataset_iterator_view: DatasetIteratorIF, dataset_view_indices, sequences):
        for orig_sample, iterator_sample in zip([[s[i] for s in sequences] for i in dataset_view_indices], dataset_iterator_view):
            assert orig_sample[0] == iterator_sample[0]
            assert orig_sample[1] == iterator_sample[1]
            assert orig_sample[2] == iterator_sample[2]

    def test_dataset_iterator_combined(self, sequences):
        iterator_1 = SequenceDatasetIterator(dataset_sequences=sequences)

        iterator_2 = SequenceDatasetIterator(dataset_sequences=sequences)
        sequences_combined = chain(sequences, sequences)
        combined_iterator = CombinedDatasetIterator(iterators=[iterator_1, iterator_2])

        for orig_sample, iterator_sample in zip(zip(*sequences_combined), combined_iterator):
            assert orig_sample[0] == iterator_sample[0]
            assert orig_sample[1] == iterator_sample[1]
            assert orig_sample[2] == iterator_sample[2]

    def test_in_memory_dataset_iterator(self, dataset_iterator: DatasetIteratorIF, sequences):
        in_memory_iterator = InMemoryDatasetIterator(dataset_iterator)

        for orig_sample, iterator_sample in zip(zip(*sequences), in_memory_iterator):
            assert orig_sample[0] == iterator_sample[0]
            assert orig_sample[1] == iterator_sample[1]
            assert orig_sample[2] == iterator_sample[2]

        assert len(in_memory_iterator) == len(sequences[0])

    def test_shuffled_dataset_iterator(self, dataset_iterator: DatasetIteratorIF):
        meta = MetaFactory.get_dataset_meta(identifier="id_1")
        shuffled_iterator = InformedDatasetFactory.get_shuffled_dataset_iterator(dataset_iterator, meta, seed=1)
        indices = list(range(len(dataset_iterator)))
        assert len(shuffled_iterator._dataset_iterator.indices) == len(indices) and \
            len(indices) == len(set(indices)) and \
            len(indices) == len(set(indices).intersection(set(shuffled_iterator._dataset_iterator.indices))) and \
            any([indice != indices[i] for i, indice in enumerate(shuffled_iterator._dataset_iterator.indices)])
