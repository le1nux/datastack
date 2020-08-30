from typing import Tuple, Any, Dict, List
from abc import ABC, abstractmethod


class PostProcessorIf(ABC):

    @abstractmethod
    def postprocess(self, *tuples: Tuple[Any]) -> Tuple[Any]:
        raise NotImplementedError


class LabelMapper(PostProcessorIf):

    class Mapping:
        def __init__(self, previous_labels: List[Any], new_label: Any):
            self.previous_labels: List[Any] = previous_labels
            self.new_label: Any = new_label

        def __contains__(self, label: Any):
            return label in self.previous_labels

    def __init__(self, mappings: List[Dict], target_position: int = 2):
        self.target_position = target_position
        self.mappings: List[LabelMapper.Mapping] = [LabelMapper.Mapping(**mapping) for mapping in mappings]

    def postprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for mapping in self.mappings:
            if sample[self.target_position] in mapping:
                # have to convert to list as tuples are immutable
                sample = list(sample)
                sample[self.target_position] = mapping.new_label
                return tuple(sample)
        return sample
