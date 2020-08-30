import pytest
from typing import List, Dict, Any, Tuple
from data_hub.dataset.postprocessors.postprocessor import LabelMapper
from data_hub.dataset.iterator import DatasetIteratorIF, PostProcessedDatasetIterator


class TestLabelMapper:
    @pytest.fixture
    def target_position(self) -> int:
        return 1

    @pytest.fixture
    def label_mapper(self, target_position: int) -> Dict[str, Any]:
        mappings = [
            {
                "previous_labels": [0, 1, 2],
                "new_label": 0
            },
            {
                "previous_labels": [3],
                "new_label": 1
            },
        ]
        label_mapper = LabelMapper(mappings, target_position)
        return label_mapper

    @pytest.fixture
    def samples(self) -> List[Tuple[Any]]:
        # returns tuples of format (sample, target)
        return [
            ([1, 2, 3], 1),
            ([3, 2, 2], 0),
            ([2, 2, 6], 2),
            ([5, 2, 5], 3),
            ([1, 5, 3], 3)
        ]

    @pytest.fixture
    def dataset_iterator(self, samples) -> DatasetIteratorIF:
        return [sample for sample in samples]

    def test_label_mapper(self, dataset_iterator: DatasetIteratorIF, label_mapper: LabelMapper, target_position: int, samples: List[Tuple[Any]]):
        iterator = PostProcessedDatasetIterator(dataset_iterator, label_mapper)
        targets = [0, 0, 0, 1, 1]
        assert all([sample[target_position] == targets[i] for i, sample in enumerate(iterator)])
        assert all([sample[0] == samples[i][0] for i, sample in enumerate(iterator)])
