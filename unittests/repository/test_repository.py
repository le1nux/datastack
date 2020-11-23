import pytest
from data_stack.repository.repository import DatasetRepository
from data_stack.dataset.factory import BaseDatasetFactory
from data_stack.dataset.iterator import DatasetIteratorIF


class MockedDatasetFactory(BaseDatasetFactory):
    def __init__(self):
        super().__init__(None)

    def get_dataset_iterator(self, split: str = None) -> DatasetIteratorIF:
        return (i for i in ["a", "b", "c"])


class TestRepository:
    @pytest.fixture
    def dataset_factory(self) -> MockedDatasetFactory:
        factory = MockedDatasetFactory()
        return factory

    @pytest.fixture
    def repository(self) -> DatasetRepository:
        return DatasetRepository()

    def test_register_and_get(self, repository: DatasetRepository, dataset_factory: MockedDatasetFactory):
        repository.register(identifier="x", dataset_factory=dataset_factory)
        dataset = repository.get(identifier="x", split="x")
        assert [i for i in dataset] == ["a", "b", "c"]
