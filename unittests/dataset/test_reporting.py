import pytest
from typing import List
from data_hub.mnist.factory import MNISTFactory
from data_hub.io.storage_connectors import StorageConnector, StorageConnectorFactory
from data_hub.dataset.reporting import DatasetIteratorReportGenerator
import tempfile
import shutil
from data_hub.dataset.iterator import CombinedDatasetIterator, DatasetMetaInformation, DatasetMetaInformationFactory


class TestReporting:

    @pytest.fixture(scope="session")
    def tmp_folder_path(self) -> str:
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path)

    @pytest.fixture(scope="session")
    def storage_connector(self, tmp_folder_path: str) -> StorageConnector:
        return StorageConnectorFactory.get_file_storage_connector(tmp_folder_path)

    @pytest.fixture(scope="session")
    def mnist_factory(self, storage_connector) -> List[int]:
        mnist_factory = MNISTFactory(storage_connector)
        return mnist_factory

    def test_plain_iterator_reporting(self, mnist_factory):
        iterator = mnist_factory.get_dataset_iterator(split="train")
        report = DatasetIteratorReportGenerator.generate_report(iterator)
        assert report.length == 60000 and not report.sub_reports

    def test_combined_iterator_reporting(self, mnist_factory):
        iterator_train = mnist_factory.get_dataset_iterator(split="train")
        iterator_test = mnist_factory.get_dataset_iterator(split="test")

        meta_information = DatasetMetaInformationFactory.get_dataset_meta_informmation_from_existing(iterator_train.dataset_meta_information, dataset_tag="my_tag")
        iterator = CombinedDatasetIterator([iterator_train, iterator_test], meta_information)
        report = DatasetIteratorReportGenerator.generate_report(iterator)
        assert report.length == 70000 and report.sub_reports[0].length == 60000 and report.sub_reports[1].length == 10000
        assert not report.sub_reports[0].sub_reports and not report.sub_reports[1].sub_reports
