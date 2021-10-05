import torch
import codecs
import numpy as np
from data_stack.dataset.preprocesor import PreprocessingHelpers
from data_stack.io.resources import ResourceFactory, StreamedResource
import io
from data_stack.io.storage_connectors import StorageConnector
from torchvision import transforms


class MNISTPreprocessor:

    def __init__(self, storage_connector: StorageConnector):
        self.storage_connector = storage_connector

    def preprocess(self, raw_sample_identifier: str, raw_target_identifier: str, sample_identifier: str, target_identifier: str):
        with self._preprocess_sample_resource(raw_sample_identifier, sample_identifier) as sample_resource:
            self.storage_connector.set_resource(identifier=sample_resource.identifier, resource=sample_resource)
        with self._preprocess_target_resource(raw_target_identifier, target_identifier) as target_resource:
            self.storage_connector.set_resource(identifier=target_resource.identifier, resource=target_resource)

    def _torch_tensor_to_streamed_resource(self, identifier: str, tensor: torch.Tensor) -> StreamedResource:
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        resource = ResourceFactory.get_resource(identifier=identifier, file_like_object=buffer)
        return resource

    def _preprocess_target_resource(self, raw_identifier: str, prep_identifier: str) -> StreamedResource:
        with self.storage_connector.get_resource(raw_identifier) as raw_resource:
            with PreprocessingHelpers.get_gzip_stream(resource=raw_resource) as unzipped_resource:
                torch_tensor = MNISTPreprocessor._read_label_file(unzipped_resource)
        resource = self._torch_tensor_to_streamed_resource(prep_identifier, torch_tensor)
        return resource

    def _preprocess_sample_resource(self, raw_identifier: str, prep_identifier: str) -> StreamedResource:
        with self.storage_connector.get_resource(raw_identifier) as raw_resource:
            with PreprocessingHelpers.get_gzip_stream(resource=raw_resource) as unzipped_resource:
                torch_tensor = MNISTPreprocessor._read_image_file(unzipped_resource)
        img_transformed = [transforms.ToTensor()(torch_tensor[i].numpy()) for i in range(len(torch_tensor))]
        img_tensor = torch.cat(img_transformed, dim=0)
        resource = self._torch_tensor_to_streamed_resource(prep_identifier, img_tensor)
        return resource

    @classmethod
    def _get_int(cls, b):
        return int(codecs.encode(b, 'hex'), 16)

    @classmethod
    def _read_label_file(cls, resource: StreamedResource):
        data = resource.read()
        assert cls._get_int(data[:4]) == 2049
        length = cls._get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        torch_tensor = torch.from_numpy(parsed).view(length).long()
        torch_tensor = torch_tensor.long()
        return torch_tensor

    @classmethod
    def _read_image_file(cls, resource: StreamedResource):
        data = resource.read()
        assert cls._get_int(data[:4]) == 2051
        return MNISTPreprocessor.read_sn3_pascalvincent_tensor(data, strict=False)

    @staticmethod
    def read_sn3_pascalvincent_tensor(data, strict: bool = True) -> torch.Tensor:
        """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
        Argument may be a filename, compressed filename, or file object.
        """
        def get_int(b: bytes) -> int:
            return int(codecs.encode(b, 'hex'), 16)

        SN3_PASCALVINCENT_TYPEMAP = {
            8: (torch.uint8, np.uint8, np.uint8),
            9: (torch.int8, np.int8, np.int8),
            11: (torch.int16, np.dtype('>i2'), 'i2'),
            12: (torch.int32, np.dtype('>i4'), 'i4'),
            13: (torch.float32, np.dtype('>f4'), 'f4'),
            14: (torch.float64, np.dtype('>f8'), 'f8')
        }
        # parse
        magic = get_int(data[0:4])
        nd = magic % 256
        ty = magic // 256
        assert 1 <= nd <= 3
        assert 8 <= ty <= 14
        m = SN3_PASCALVINCENT_TYPEMAP[ty]
        s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
        parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
        assert parsed.shape[0] == np.prod(s) or not strict
        return torch.from_numpy(parsed.astype(m[2])).view(*s)
