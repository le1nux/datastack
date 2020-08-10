from data_hub.io.storage_connectors import StorageConnector, FileStorageConnector, StorageConnectorFactory
from data_hub.exception import ResourceNotFoundError
import pytest
import io


class TestStorageConnectorFactory:

    @pytest.mark.usefixtures("tmp_folder_path")
    def test_get_file_storage_connector(self, tmp_folder_path):
        storage_connector = StorageConnectorFactory.get_file_storage_connector(tmp_folder_path)
        assert isinstance(storage_connector, FileStorageConnector)


class TestFileStorageConnector:

    @pytest.fixture
    @pytest.mark.usefixtures("tmp_folder_path")
    def storage_connector(self, tmp_folder_path: str) -> StorageConnector:
        return StorageConnectorFactory.get_file_storage_connector(tmp_folder_path)

    def test_get_resource_404(self, storage_connector: FileStorageConnector):
        raised = False
        try:
            storage_connector.get_resource("xxxxxxxx")
        except ResourceNotFoundError:
            raised = True
        assert raised

    def test_set_get_resource(self, storage_connector: FileStorageConnector):
        bin_stream1 = io.BytesIO(b"abcdef")
        storage_connector.set_resource("x", bin_stream1)
        bin_stream2 = storage_connector.get_resource("x")
        bin_stream1.seek(0)
        bin_stream2.seek(0)
        assert bin_stream2.read() == bin_stream1.read()
