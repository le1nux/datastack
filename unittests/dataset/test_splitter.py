import pytest
from data_hub.dataset.iterator import DatasetIteratorIF
from typing import List
from data_hub.dataset.splitter import RandomSplitterImpl, Splitter


class MockedDatasetIterator(DatasetIteratorIF):
    def __init__(self):
        self.data = list(range(1000))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        return self.data[index]


class TestSplitter:
    @pytest.fixture
    def ratios(self) -> List[int]:
        return [0.3, 0.3, 0.2, 0.1, 0.1]

    @pytest.fixture
    def dataset_iterator(self) -> DatasetIteratorIF:
        return MockedDatasetIterator()

    def test_random_splitter(self, ratios: List[int], dataset_iterator: DatasetIteratorIF):
        splitter_impl = RandomSplitterImpl(ratios=ratios)
        splitter = Splitter(splitter_impl)
        iterator_splits = splitter.split(dataset_iterator)

        assert sorted([i for split in iterator_splits for i in split]) == sorted(dataset_iterator)
