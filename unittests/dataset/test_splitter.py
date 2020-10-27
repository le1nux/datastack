import pytest
from data_hub.dataset.iterator import DatasetIteratorIF, SequenceDatasetIterator
from typing import List
from data_hub.dataset.splitter import RandomSplitterImpl, Splitter
from data_hub.dataset.meta_information import DatasetMetaInformationFactory


class TestSplitter:
    @pytest.fixture
    def ratios(self) -> List[int]:
        return [0.3, 0.3, 0.2, 0.1, 0.1]

    @pytest.fixture
    def dataset_meta_information(self) -> DatasetMetaInformation:
        return DatasetMetaInformationFactory.get_dataset_meta_informmation(dataset_name="TEST DATASET",
                                                                           dataset_tag="train",
                                                                           sample_pos=0,
                                                                           target_pos=1,
                                                                           tag_pos=2)

    @pytest.fixture
    def dataset_iterator(self, dataset_meta_information) -> DatasetIteratorIF:
        return SequenceDatasetIterator(dataset_sequences=[list(range(10)), list(range(10))],
                                       dataset_meta_information=dataset_meta_information)

    def test_random_splitter(self, ratios: List[int], dataset_iterator: DatasetIteratorIF):
        splitter_impl = RandomSplitterImpl(ratios=ratios)
        splitter = Splitter(splitter_impl)
        iterator_splits = splitter.split(dataset_iterator)

        assert sorted([i for split in iterator_splits for i in split]) == sorted(dataset_iterator)
