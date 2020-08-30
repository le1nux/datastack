import pytest
from data_hub.dataset.iterator import DatasetIteratorIF, DatasetIterator
from typing import List
from data_hub.dataset.splitter import RandomSplitterImpl, Splitter


class TestSplitter:
    @pytest.fixture
    def ratios(self) -> List[int]:
        return [0.3, 0.3, 0.2, 0.1, 0.1]

    @pytest.fixture
    def dataset_iterator(self) -> DatasetIteratorIF:
        return DatasetIterator(dataset_sequences=[list(range(10)), list(range(10))],
                               dataset_name="N",
                               dataset_tag="t")

    def test_random_splitter(self, ratios: List[int], dataset_iterator: DatasetIteratorIF):
        splitter_impl = RandomSplitterImpl(ratios=ratios)
        splitter = Splitter(splitter_impl)
        iterator_splits = splitter.split(dataset_iterator)

        assert sorted([i for split in iterator_splits for i in split]) == sorted(dataset_iterator)
