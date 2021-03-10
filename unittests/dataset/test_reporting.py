import pytest
from data_stack.io.storage_connectors import StorageConnector, StorageConnectorFactory
from data_stack.dataset.reporting import DatasetIteratorReportGenerator
import tempfile
import shutil
from data_stack.dataset.factory import InformedDatasetFactory
from data_stack.dataset.meta import DatasetMeta, MetaFactory
from data_stack.dataset.iterator import DatasetIteratorIF, SequenceDatasetIterator, InformedDatasetIterator


class TestReporting:

    @pytest.fixture(scope="session")
    def tmp_folder_path(self) -> str:
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path)

    @pytest.fixture(scope="session")
    def storage_connector(self, tmp_folder_path: str) -> StorageConnector:
        return StorageConnectorFactory.get_file_storage_connector(tmp_folder_path)

    # @pytest.fixture(scope="session")
    # def mnist_factory(self, storage_connector) -> List[int]:
    #     mnist_factory = MNISTFactory(storage_connector)
    #     return mnist_factory

    @pytest.fixture
    def dataset_meta(self) -> DatasetMeta:
        iterator_meta = MetaFactory.get_iterator_meta(sample_pos=0, target_pos=1, tag_pos=2)
        return MetaFactory.get_dataset_meta(identifier="identifier_1",
                                            dataset_name="TEST DATASET",
                                            dataset_tag="train",
                                            iterator_meta=iterator_meta)

    @pytest.fixture
    def dataset_iterator(self) -> DatasetIteratorIF:
        targets = [j for i in range(10) for j in range(9)] + [10]*1000
        samples = [0]*len(targets)
        return SequenceDatasetIterator(dataset_sequences=[samples, targets])

    @pytest.fixture
    def informed_dataset_iterator(self, dataset_iterator, dataset_meta) -> DatasetIteratorIF:
        return InformedDatasetFactory.get_dataset_iterator(dataset_iterator, dataset_meta)

    def test_plain_iterator_reporting(self, informed_dataset_iterator):
        report = DatasetIteratorReportGenerator.generate_report(informed_dataset_iterator)
        print(report)
        assert report.length == 1090 and not report.sub_reports

    def test_combined_iterator_reporting(self, informed_dataset_iterator):
        meta_combined = MetaFactory.get_dataset_meta_from_existing(informed_dataset_iterator.dataset_meta, dataset_tag="full")
        iterator = InformedDatasetFactory.get_combined_dataset_iterator(
            [informed_dataset_iterator, informed_dataset_iterator], meta_combined)
        report = DatasetIteratorReportGenerator.generate_report(iterator)
        assert report.length == 2180 and report.sub_reports[0].length == 1090 and report.sub_reports[1].length == 1090
        assert not report.sub_reports[0].sub_reports and not report.sub_reports[1].sub_reports
