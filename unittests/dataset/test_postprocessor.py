import pytest
from typing import List, Dict, Any, Tuple
# from data_hub.dataset.postprocessors.postprocessor import LabelMapper


class TestLabelMapper:
    @pytest.fixture
    def mappings(self) -> Dict[str, Any]:
        return [
            {
                "previous_labels": [0, 1, 2],
                "new_label": 0
            },
            {
                "previous_labels": [3],
                "new_label": 1
            },
        ]

    @pytest.fixture
    def target_position(self) -> int:
        return 1

    @pytest.fixture
    def samples(self) -> List[Tuple[Any]]:
        # returns tuples of format (sample, target)
        return [
            ([1, 2, 3], 1),
            ([3, 2, 2], 0),
            ([2, 2, 6], 2),
            ([5, 2, 5], 3),
            ([1, 5, 3], 3)
        ]

    # def test_label_mapper(self, mappings, target_position, samples: List[Tuple[Any]]):
    #     label_mapper = LabelMapper(mappings, target_position)
    #     mapped_targets = [label_mapper.postprocess(sample)[target_position] for sample in samples]
    #     assert mapped_targets == [0, 0, 0, 1, 1]
