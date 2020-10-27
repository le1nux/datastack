import pytest
from typing import List, Any, Tuple
from data_hub.dataset.iterator import DatasetIteratorIF, SequenceDatasetIterator, DatasetIteratorView, CombinedDatasetIterator
from itertools import chain
from data_hub.dataset.meta_information import DatasetMetaInformation, DatasetMetaInformationFactory


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
    def dataset_meta_information(self) -> DatasetMetaInformation:
        return DatasetMetaInformationFactory.get_dataset_meta_informmation(dataset_name="TEST DATASET",
                                                                           dataset_tag="train",
                                                                           sample_pos=0,
                                                                           target_pos=1,
                                                                           tag_pos=2)

    @pytest.fixture
    def dataset_iterator(self, sequences, dataset_meta_information) -> DatasetIteratorIF:
        return SequenceDatasetIterator(dataset_sequences=sequences,
                                       dataset_meta_information=dataset_meta_information)

    @pytest.fixture
    def dataset_view_indices(self) -> List[int]:
        return [0, 2]

    @pytest.fixture
    def dataset_iterator_view(self, dataset_iterator, dataset_view_indices, dataset_meta_information) -> DatasetIteratorIF:
        return DatasetIteratorView(dataset_iterator, indices=dataset_view_indices, dataset_meta_information=dataset_meta_information)

    def test_dataset_iterator(self, dataset_iterator: DatasetIteratorIF, sequences):
        meta_info = dataset_iterator.dataset_meta_information
        for orig_sample, iterator_sample in zip(zip(*sequences), dataset_iterator):
            assert orig_sample[meta_info.sample_pos] == iterator_sample[meta_info.sample_pos]
            assert orig_sample[meta_info.target_pos] == iterator_sample[meta_info.target_pos]
            assert orig_sample[meta_info.tag_pos] == iterator_sample[meta_info.tag_pos]

    def test_dataset_iterator_view(self, dataset_iterator_view: DatasetIteratorIF, dataset_view_indices, sequences):
        meta_info = dataset_iterator_view.dataset_meta_information
        for orig_sample, iterator_sample in zip([[s[i] for s in sequences] for i in dataset_view_indices], dataset_iterator_view):
            assert orig_sample[meta_info.sample_pos] == iterator_sample[meta_info.sample_pos]
            assert orig_sample[meta_info.target_pos] == iterator_sample[meta_info.target_pos]
            assert orig_sample[meta_info.tag_pos] == iterator_sample[meta_info.tag_pos]

    def test_dataset_iterator_combined(self, sequences, dataset_meta_information):
        iterator_1 = SequenceDatasetIterator(dataset_sequences=sequences,
                                             dataset_meta_information=dataset_meta_information)

        iterator_2 = SequenceDatasetIterator(dataset_sequences=sequences,
                                             dataset_meta_information=dataset_meta_information)
        sequences_combined = chain(sequences, sequences)
        combined_iterator = CombinedDatasetIterator(iterators=[iterator_1, iterator_2], dataset_meta_information=dataset_meta_information)

        for orig_sample, iterator_sample in zip(zip(*sequences_combined), combined_iterator):
            assert orig_sample[dataset_meta_information.sample_pos] == iterator_sample[dataset_meta_information.sample_pos]
            assert orig_sample[dataset_meta_information.target_pos] == iterator_sample[dataset_meta_information.target_pos]
            assert orig_sample[dataset_meta_information.tag_pos] == iterator_sample[dataset_meta_information.tag_pos]
