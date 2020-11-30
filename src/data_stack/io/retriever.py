from abc import ABC, abstractmethod
import torchvision
from data_stack.util.logger import logger
from typing import List
from data_stack.io.storage_connectors import StorageConnector
from data_stack.exception import DatasetFileCorruptError
import tempfile
import os
from data_stack.util.helper import calculate_md5
from data_stack.io.resources import ResourceFactory
from data_stack.io.resource_definition import ResourceDefinition


class RetrieverFactory:

    @classmethod
    def get_http_retriever(cls, storage_connector: StorageConnector) -> "Retriever":
        retriever_impl = HTTPRetrieverImpl(storage_connector)
        return Retriever(retriever_impl)

    @classmethod
    def get_file_retriever(cls, storage_connector: StorageConnector) -> "Retriever":
        retriever_impl = FileRetrieverImpl(storage_connector)
        return Retriever(retriever_impl)


class Retriever:

    def __init__(self, retriever_impl: "RetrieverImplIF"):
        self.retriever_impl = retriever_impl

    def retrieve(self, retrieval_jobs: List[ResourceDefinition]) -> List[str]:
        return self.retriever_impl.retrieve(retrieval_jobs)


class RetrieverImplIF(ABC):

    def __init__(self, storage_connector: StorageConnector):
        self.storage_connector = storage_connector

    @abstractmethod
    def retrieve(self, retrieval_jobs: List[ResourceDefinition]) -> List[str]:
        raise NotImplementedError


class HTTPRetrieverImpl(RetrieverImplIF):
    def __init__(self, storage_connector: StorageConnector):
        super().__init__(storage_connector)

    def _download_file(self, dest_folder: str, url: str, md5: str) -> str:
        """ Downloads a file given by the url.
        :param dest_path: destination path
        :param url: URL to the dataset
        :return: Path to downloaded file
        """
        logger.debug(f"Downloading data file from {url} ...")
        # download file
        filename = url.rpartition('/')[2]
        torchvision.datasets.utils.download_url(url, root=dest_folder, filename=filename)
        logger.debug("Done.")
        file_path = os.path.join(dest_folder, filename)
        with open(file_path, 'rb') as fd:
            calculated_md5_sum = calculate_md5(fd)
        if calculated_md5_sum != md5:
            logger.fatal(f"Given MD5 hash did not match with the md5 has of file {file_path}")
            raise DatasetFileCorruptError
        return file_path

    def _download(self, retrieval_jobs: List[ResourceDefinition], dest_folder: str) -> List[str]:
        file_paths = [self._download_file(
            url=retrieval_job.source, dest_folder=dest_folder, md5=retrieval_job.md5_sum) for retrieval_job in retrieval_jobs]
        return file_paths

    def retrieve(self, retrieval_jobs: List[ResourceDefinition]):
        resource_identifiers = []
        with tempfile.TemporaryDirectory() as tmp_dest_folder:
            logger.debug(f'Created temporary directory {tmp_dest_folder} for downloading resources...')
            # download dataset files
            tmp_resource_paths = self._download(retrieval_jobs, tmp_dest_folder)
            # store datset files
            for retrieval_job, tmp_resource_path in zip(retrieval_jobs, tmp_resource_paths):
                with open(tmp_resource_path, "rb") as fd:
                    resource = ResourceFactory.get_resource(identifier=retrieval_job.identifier, file_like_object=fd)
                    self.storage_connector.set_resource(identifier=retrieval_job.identifier, resource=resource)
                    resource_identifiers.append(retrieval_job.identifier)
        return resource_identifiers


class FileRetrieverImpl(RetrieverImplIF):

    def __init__(self, storage_connector: StorageConnector):
        super().__init__(storage_connector)

    def retrieve(self, retrieval_jobs: List[ResourceDefinition]):
        resource_identifiers = []
        for retrieval_job in retrieval_jobs:
            with open(retrieval_job.source, "rb") as fd:
                resource = ResourceFactory.get_resource(identifier=retrieval_job.identifier, file_like_object=fd)
                calculated_md5_sum = calculate_md5(resource)
                if calculated_md5_sum != retrieval_job.md5_sum:
                    logger.fatal(f"Given MD5 hash did not match with the md5 has of file {retrieval_job.source}")
                    raise DatasetFileCorruptError
                self.storage_connector.set_resource(identifier=retrieval_job.identifier, resource=resource)
                resource_identifiers.append(retrieval_job.identifier)
        return resource_identifiers
