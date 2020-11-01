import pytest
from typing import List
from data_hub.mnist.factory import MNISTFactory
from data_hub.io.storage_connectors import StorageConnector, StorageConnectorFactory
from data_hub.dataset.reporting import DatasetIteratorReportGenerator
import tempfile
import shutil
from data_hub.dataset.iterator import CombinedDatasetIterator, InformedDatasetIterator
from data_hub.dataset.meta import MetaFactory, IteratorMeta
from data_hub.dataset.factory import InformedDatasetFactory, HigherOrderDatasetFactory


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
        iterator, iterator_meta = mnist_factory.get_dataset_iterator(split="train")
        dataset_meta = MetaFactory.get_dataset_meta(identifier="id x", dataset_name="MNIST",
                                                    dataset_tag="train", iterator_meta=iterator_meta)

        informed_iterator = InformedDatasetIterator(iterator, dataset_meta)
        report = DatasetIteratorReportGenerator.generate_report(informed_iterator)
        print(report)
        assert report.length == 60000 and not report.sub_reports

    def test_combined_iterator_reporting(self, mnist_factory):
        iterator_train, iterator_train_meta = mnist_factory.get_dataset_iterator(split="train")
        iterator_test, iterator_test_meta = mnist_factory.get_dataset_iterator(split="test")
        meta_train = MetaFactory.get_dataset_meta(identifier="id x", dataset_name="MNIST",
                                                  dataset_tag="train", iterator_meta=iterator_train_meta)
        meta_test = MetaFactory.get_dataset_meta(identifier="id x", dataset_name="MNIST",
                                                 dataset_tag="train", iterator_meta=iterator_test_meta)

        informed_iterator_train = InformedDatasetFactory.get_dataset_iterator(iterator_train, meta_train)
        informed_iterator_test = InformedDatasetFactory.get_dataset_iterator(iterator_test, meta_test)

        meta_combined = MetaFactory.get_dataset_meta_from_existing(informed_iterator_train.dataset_meta, dataset_tag="full")

        iterator = InformedDatasetFactory.get_combined_dataset_iterator([informed_iterator_train, informed_iterator_test], meta_combined)
        report = DatasetIteratorReportGenerator.generate_report(iterator)
        assert report.length == 70000 and report.sub_reports[0].length == 60000 and report.sub_reports[1].length == 10000
        assert not report.sub_reports[0].sub_reports and not report.sub_reports[1].sub_reports
