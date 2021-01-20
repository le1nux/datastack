from data_stack.dataset.iterator import DatasetIteratorIF
from dataclasses import dataclass
from typing import List, Union, Dict, Any
from collections import Counter
import time
from enum import Enum
import json
import yaml
import dataclasses
from data_stack.dataset.iterator import InformedDatasetIterator


@dataclass
class DatasetIteratorReport:
    identifier: str
    name: str
    tag: str
    length: int
    sample_pos: int
    target_pos: int
    tag_pos: int
    sample_shape: List[int]
    target_dist: Dict[Union[str, int], int]
    iteration_speed: float
    sub_reports: List["DatasetIteratorReport"]


class DatasetIteratorReportGenerator:
    class ReportFormat(Enum):
        DATA_CLASS = "data_class"
        YAML = "yaml"
        JSON = "json"
        DICT = "dict"

    def generate_report(iterator: InformedDatasetIterator, report_format: ReportFormat = ReportFormat.DATA_CLASS):
        sub_reports = [DatasetIteratorReportGenerator.generate_report(sub_iterator) for sub_iterator in iterator.underlying_iterators]
        meta = iterator.dataset_meta
        target_dist = {k: v for k, v in sorted(Counter([row[meta.target_pos] for row in iterator]).items())}
        iteration_speed = DatasetIteratorReportGenerator.measure_iteration_speed(iterator)
        # generate report
        report = DatasetIteratorReport(meta.identifier, meta.dataset_name, meta.dataset_tag, len(iterator), meta.sample_pos,
                                       meta.target_pos, meta.tag_pos, list(iterator[0][meta.sample_pos].shape), target_dist,
                                       iteration_speed, sub_reports)
        # format report
        if report_format == DatasetIteratorReportGenerator.ReportFormat.JSON:
            return DatasetIteratorReportGenerator._to_json(report)
        elif report_format == DatasetIteratorReportGenerator.ReportFormat.YAML:
            return DatasetIteratorReportGenerator._to_yaml(report)
        elif report_format == DatasetIteratorReportGenerator.ReportFormat.DICT:
            return DatasetIteratorReportGenerator._to_dict(report)
        else:
            return report

    def measure_iteration_speed(iterator: DatasetIteratorIF) -> float:
        def full_iteration(iterator: DatasetIteratorIF):
            for _ in iterator:
                pass

        diff = 0
        iterations = 0
        while(diff < 10):
            start = time.time()
            full_iteration(iterator)
            end = time.time()
            diff += end-start
            iterations += 1
        return len(iterator)*iterations / diff  # iterations per second

    def _to_json(report: DatasetIteratorReport) -> str:
        return json.dumps(DatasetIteratorReportGenerator._to_dict(report), sort_keys=True, indent=4)

    def _to_yaml(report: DatasetIteratorReport) -> str:
        return yaml.dump(DatasetIteratorReportGenerator._to_dict(report), sort_keys=True, indent=4)

    def _to_dict(report: DatasetIteratorReport) -> Dict[str, Any]:
        report_dict = dataclasses.asdict(report)
        return report_dict


if __name__ == "__main__":
    from data_stack.mnist.factory import MNISTFactory
    import data_stack
    import os
    from data_stack.io.storage_connectors import FileStorageConnector
    from data_stack.dataset.metarmation import MetaFactory

    data_stack_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(data_stack.__file__))))
    example_file_storage_path = os.path.join(data_stack_root, "example_file_storage")
    storage_connector = FileStorageConnector(root_path=example_file_storage_path)
    mnist_factory = MNISTFactory(storage_connector)
    iterator = mnist_factory.get_dataset_iterator({"split": "train"})
    report = DatasetIteratorReportGenerator.generate_report(iterator)
    print(report)

    from data_stack.dataset.iterator import CombinedDatasetIterator
    iterator_train = mnist_factory.get_dataset_iterator({"split": "train"})
    iterator_test = mnist_factory.get_dataset_iterator({"split": "test"})
    meta = MetaFactory.get_dataset_meta_from_existing(
        iterator_test.dataset_meta, "combined", "my_tag")
    iterator = CombinedDatasetIterator([iterator_train, iterator_train, iterator_train], meta)
    report = DatasetIteratorReportGenerator.generate_report(iterator)
    print(report)
