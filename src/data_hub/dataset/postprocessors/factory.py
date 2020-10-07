from typing import Any, Dict, List
from data_hub.dataset.iterator import DatasetIteratorIF, PostProcessedDatasetIterator, DatasetIteratorView
from data_hub.dataset.postprocessors.postprocessor import LabelMapper


class PostprocesssedDatasetIteratorFactory:

    def get_mapped_labels_iterator(iterator: DatasetIteratorIF, mappings: Dict, target_position: int) -> DatasetIteratorIF:
        label_mapper = LabelMapper(mappings=mappings, target_position=target_position)
        return PostProcessedDatasetIterator(iterator, label_mapper)

    def get_filtered_labels_iterator(iterator: DatasetIteratorIF, filtered_labels: List[Any], target_position: int) -> DatasetIteratorIF:
        valid_indices = [i for i in range(len(iterator)) if iterator[i][target_position] in filtered_labels]
        return DatasetIteratorView(iterator, valid_indices)
