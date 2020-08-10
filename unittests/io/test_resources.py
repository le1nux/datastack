from data_hub.io.resources import StreamedBinaryResource
import os
import tempfile
import pytest
from itertools import product


class TestStreamedBinaryResource:

    @pytest.fixture
    def file_size(self) -> int:
        return 2044

    @pytest.fixture
    def content(self, file_size: int) -> str:
        return "a"*file_size

    @pytest.fixture
    def path_to_random_bin_file(self, content: str) -> str:
        fd, path = tempfile.mkstemp()
        try:
            with os.fdopen(fd, 'w') as tmp:
                tmp.write(content)
            yield path
        finally:
            os.remove(path)

    @pytest.mark.parametrize("chunk_size,copy_to_memory", product([1, 8, 16, 4096], [True, False]))
    def test_lazy_streamed_binary_file_resource(self, path_to_random_bin_file: str, content: str, file_size: int, chunk_size: int, copy_to_memory: bool):
        fd = open(path_to_random_bin_file, "rb")
        sbfr = StreamedBinaryResource("my_resource", fd, copy_to_memory=copy_to_memory, chunk_size=chunk_size)
        chunks = [chunk for chunk in sbfr]
        assert (len(chunks)-1)*chunk_size + len(chunks[-1]) == file_size

    @pytest.mark.parametrize("chunk_size,copy_to_memory", product([1, 8, 16, 4096], [True, False]))
    def test_reentry_lazy_streamed_binary_file_resource(self, path_to_random_bin_file: str, content: str, file_size: int, chunk_size: int, copy_to_memory: bool):
        fd = open(path_to_random_bin_file, "rb")
        sbfr = StreamedBinaryResource("my_resource", fd, copy_to_memory=copy_to_memory, chunk_size=chunk_size)
        chunks = [chunk for chunk in sbfr]
        chunks = [chunk for chunk in sbfr]
        assert (len(chunks)-1)*chunk_size + len(chunks[-1]) == file_size

    @pytest.mark.parametrize("copy_to_memory", [True, False])
    def test_context_manager(self, path_to_random_bin_file: str, content: str, file_size: int, copy_to_memory: bool):
        fd = open(path_to_random_bin_file, "rb")
        chunk_size = 10
        with StreamedBinaryResource("my_resource", fd, copy_to_memory=copy_to_memory, chunk_size=chunk_size) as sbfr:
            chunks = [chunk for chunk in sbfr]
            chunks = [chunk for chunk in sbfr]
            assert (len(chunks)-1)*chunk_size + len(chunks[-1]) == file_size
        try:
            catched_error = False
            chunks = [chunk for chunk in sbfr]
        except ValueError:
            catched_error = True
        finally:
            assert catched_error
