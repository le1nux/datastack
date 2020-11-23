#!/usr/bin/env python3

from abc import ABC, abstractmethod
from data_stack.util.logger import logger
from data_stack.util.helper import is_safe_path
from data_stack.exception import MaliciousFilePathError, ResourceNotFoundError
import os
from data_stack.io.resources import ResourceFactory, StreamedResource


class StorageConnectorFactory:

    @classmethod
    def get_file_storage_connector(cls, folder_path: str) -> "StorageConnector":
        return FileStorageConnector(folder_path)


class StorageConnector(ABC):

    @abstractmethod
    def get_resource(self, identifier: str, resource_type: ResourceFactory.SupportedStreamedResourceTypes = ResourceFactory.SupportedStreamedResourceTypes.STREAMED_BINARY_RESOURCE) -> StreamedResource:
        raise NotImplementedError

    @abstractmethod
    def set_resource(self, identifier: str, resource: StreamedResource):
        raise NotImplementedError

    @abstractmethod
    def has_resource(self, identifier: str) -> bool:
        raise NotImplementedError


class FileStorageConnector(StorageConnector):
    def __init__(self, root_path: str):
        self._root_path = os.path.abspath(root_path)

    @property
    def root_path(self):
        return self._root_path

    def get_resource(self, identifier: str, resource_type: ResourceFactory.SupportedStreamedResourceTypes = ResourceFactory.SupportedStreamedResourceTypes.STREAMED_BINARY_RESOURCE):
        if not self.has_resource(identifier):
            raise ResourceNotFoundError(f"Resource {identifier} not found.")
        full_path = self._get_full_path(identifier)
        fd = open(full_path, "rb")
        streamed_resource = ResourceFactory.get_resource(identifier=identifier, file_like_object=fd, resource_type=resource_type)
        return streamed_resource

    def set_resource(self, identifier: str, resource: "StreamedResource"):
        logger.debug(f"Storing resource {identifier}")
        full_path = self._get_full_path(identifier)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as f:
            for chunk in resource:
                f.write(chunk)

    def has_resource(self, identifier: str) -> bool:
        full_path = self._get_full_path(identifier)
        return os.path.exists(full_path)

    def _get_full_path(self, identifier: str) -> str:
        full_path = os.path.join(self.root_path, identifier)
        if not is_safe_path(basedir=self.root_path, path=full_path):
            raise MaliciousFilePathError
        return full_path
