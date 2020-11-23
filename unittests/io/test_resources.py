from data_stack.io.resources import StreamedResource, StreamedTextResource
import os
import tempfile
import pytest
from itertools import product


class BaseTest:
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


class TestStreamedResource(BaseTest):

    @pytest.mark.parametrize("chunk_size", [1, 8, 16, 4096])
    def test_lazy_streamed_binary_file_resource(self, path_to_random_bin_file: str, content: str, file_size: int, chunk_size: int):
        fd = open(path_to_random_bin_file, "rb")
        sbfr = StreamedResource("my_resource", fd, chunk_size=chunk_size)
        chunks = [chunk for chunk in sbfr]
        assert (len(chunks)-1)*chunk_size + len(chunks[-1]) == file_size

    @pytest.mark.parametrize("chunk_size", [1, 8, 16, 4096])
    def test_reentry_lazy_streamed_binary_file_resource(self, path_to_random_bin_file: str, content: str, file_size: int, chunk_size: int):
        fd = open(path_to_random_bin_file, "rb")
        sbfr = StreamedResource("my_resource", fd, chunk_size=chunk_size)
        chunks = [chunk for chunk in sbfr]
        chunks = [chunk for chunk in sbfr]
        assert (len(chunks)-1)*chunk_size + len(chunks[-1]) == file_size

    def test_context_manager(self, path_to_random_bin_file: str, content: str, file_size: int):
        fd = open(path_to_random_bin_file, "rb")
        chunk_size = 10
        with StreamedResource("my_resource", fd, chunk_size=chunk_size) as sbfr:
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


class TestStreamedTextResource(BaseTest):

    @pytest.fixture
    def streamed_resource(self, path_to_random_bin_file: str, content: str, file_size: int):
        fd = open(path_to_random_bin_file, "rb")
        return StreamedResource("my_resource", fd, chunk_size=30)

    @pytest.mark.parametrize("encoding", ["utf-8"])
    def test_from_streamed_resouce(self, streamed_resource: StreamedResource, encoding: str, content: str):
        streamed_text_resource = StreamedTextResource.from_streamed_resouce(streamed_resource, encoding=encoding)
        assert streamed_text_resource.read() == content
