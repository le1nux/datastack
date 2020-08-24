from typing import Sequence, List


class DatasetIteratorIF:

    def __init__(self, dataset_sequences: List[Sequence], dataset_name: str = None, dataset_tag: str = None):
        self._dataset_name = dataset_name
        self._dataset_sequences = dataset_sequences
        self._dataset_tag = dataset_tag

    def __len__(self):
        return len(self._dataset_sequences[0])

    def __getitem__(self, index: int):
        return tuple([s[index] for s in self._dataset_sequences])

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def dataset_tag(self) -> str:
        return self._dataset_tag


class SplittedDatasetIteratorIF:
    """Provides a view on a `DatasetIterator` for accessing elements of a given split only."""

    def __init__(self, dataset_iterator: DatasetIteratorIF, indices: List[int], dataset_tag: str = None):
        self._dataset_name = dataset_iterator.dataset_name
        self._dataset_iterator = dataset_iterator
        self._indices = indices
        self._dataset_tag = dataset_tag

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        original_dataset_index = self._indices[index]
        return self._dataset_iterator[original_dataset_index]

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    @property
    def dataset_tag(self) -> str:
        return self._dataset_tag
