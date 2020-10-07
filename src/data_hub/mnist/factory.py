#!/usr/bin/env python3

import os
from data_hub.io.retriever import RetrieverFactory, RetrievalJob
from data_hub.io.storage_connectors import StorageConnector, FileStorageConnector
from data_hub.mnist.preprocessor import MNISTPreprocessor
from data_hub.dataset.factory import DatasetFactory
from data_hub.dataset.iterator import DatasetIteratorIF
from data_hub.mnist.iterator import MNISTIterator
from data_hub.exception import ResourceNotFoundError


class MNISTFactory(DatasetFactory):

    def __init__(self, storage_connector: StorageConnector):
        self.raw_path = "mnist/raw/"
        self.preprocessed_path = "mnist/preprocessed/"
        self.resource_definitions = {
            "train": [
                MNISTFactory.ResourceDefinition(identifier=os.path.join(self.raw_path, "samples_train.gz"),
                                                source='http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                                                md5_sum="f68b3c2dcbeaaa9fbdd348bbdeb94873"),
                MNISTFactory.ResourceDefinition(identifier=os.path.join(self.raw_path, "labels_train.gz"),
                                                source='http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                                                md5_sum="d53e105ee54ea40749a09fcbcd1e9432")

            ],
            "test": [
                MNISTFactory.ResourceDefinition(identifier=os.path.join(self.raw_path, "samples_test.gz"),
                                                source='http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                                                md5_sum="9fb629c4189551a2d022fa330f9573f3"),
                MNISTFactory.ResourceDefinition(identifier=os.path.join(self.raw_path, "targets.gz"),
                                                source='http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
                                                md5_sum="ec29112dd5afa0611ce80d1b7f02629c")
            ]
        }

        super().__init__(storage_connector)

    def check_exists(self) -> bool:
        # TODO come up with a better check!
        sample_identifier = self._get_resource_id(data_type="preprocessed", split="train", element="samples.pt")
        return self.storage_connector.has_resource(sample_identifier)

    def _get_resource_id(self, data_type: str,  split: str, element: str) -> str:
        return os.path.join("mnist", data_type, split, element)

    def _retrieve_raw(self):
        retrieval_jobs = [RetrievalJob(identifier=resource_definition.identifier,
                                       source=resource_definition.source,
                                       md5_sum=resource_definition.md5_sum)
                          for split, definitions_list in self.resource_definitions.items()
                          for resource_definition in definitions_list]
        retriever = RetrieverFactory.get_http_retriever(self.storage_connector)
        retriever.retrieve(retrieval_jobs)

    def _prepare_split(self, split: str):
        preprocessor = MNISTPreprocessor(self.storage_connector)
        sample_identifier = self._get_resource_id(data_type="preprocessed", split=split, element="samples.pt")
        target_identifier = self._get_resource_id(data_type="preprocessed", split=split, element="targets.pt")
        prep_resources = preprocessor.preprocess(*[r.identifier for r in self.resource_definitions[split]],
                                                 sample_identifier=sample_identifier,
                                                 target_identifier=target_identifier)
        for resource in prep_resources:
            self.storage_connector.set_resource(identifier=resource.identifier, resource=resource)

    def _get_iterator(self, split: str):
        sample_identifier = self._get_resource_id(data_type="preprocessed", split=split, element="samples.pt")
        target_identifier = self._get_resource_id(data_type="preprocessed", split=split, element="targets.pt")
        sample_resource = self.storage_connector.get_resource(identifier=sample_identifier)
        target_resource = self.storage_connector.get_resource(identifier=target_identifier)
        return MNISTIterator(sample_resource, target_resource, split)

    def get_dataset_iterator(self, split: str = None) -> DatasetIteratorIF:
        splits = self.resource_definitions.keys()
        if split not in splits:
            raise ResourceNotFoundError(f"Split {split} is not defined.")
        if not self.check_exists():
            self._retrieve_raw()
            for s in splits:
                self._prepare_split(s)
        return self._get_iterator(split)


if __name__ == "__main__":
    import data_hub
    from matplotlib import pyplot as plt

    data_hub_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(data_hub.__file__))))
    example_file_storage_path = os.path.join(data_hub_root, "example_file_storage")
    storage_connector = FileStorageConnector(root_path=example_file_storage_path)

    mnist_factory = MNISTFactory(storage_connector)
    mnist_iterator = mnist_factory.get_dataset_iterator(split="train")
    img, target = mnist_iterator[0]
    plt.imshow(img)
    plt.show()
