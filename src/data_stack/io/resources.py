from abc import ABC, abstractmethod
import io
from typing import AnyStr, List
from enum import Enum


class ResourceFactory:

    class SupportedStreamedResourceTypes(Enum):
        STREAMED_BINARY_RESOURCE = "STREAMED_BINARY_RESOURCE"
        STREAMED_TEXT_RESOURCE = "STREAMED_TEXT_RESOURCE"

    @staticmethod
    def get_resource(identifier: str, file_like_object: io.IOBase, chunk_size: int = 1024, resource_type: SupportedStreamedResourceTypes = SupportedStreamedResourceTypes.STREAMED_BINARY_RESOURCE) -> "StreamedResource":
        if resource_type == ResourceFactory.SupportedStreamedResourceTypes.STREAMED_TEXT_RESOURCE:
            return StreamedTextResource(identifier, file_like_object, chunk_size)
        else:
            return StreamedResource(identifier, file_like_object, chunk_size)


class IterableIF(ABC):

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError


class Buffer(IterableIF):
    def __init__(self, buffer: io.IOBase, chunk_size: int = 1024):
        self._chunk_size = chunk_size
        self._buffer = buffer

    # ==============================PROPERTIES==================================

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    # ==============================ITERATOR====================================

    def __iter__(self):
        self._buffer.seek(0)
        while True:
            chunk = self._buffer.read(self.chunk_size)
            if chunk:
                yield chunk
            else:
                break

    # ===========================CONTEXT=MANAGER===============================

    def __enter__(self) -> "StreamedResource":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(f"Ecxception: {exc_type}")
        self._buffer.close()

    # ========================IO=METHOD=WRAPPERS==============================

    def close(self) -> None:
        self._buffer.close()

    @property
    def closed(self) -> bool:
        return self._buffer.closed

    def fileno(self) -> int:
        return self._buffer.fileno()

    # read methods

    def read(self, n: int = -1) -> AnyStr:
        return self._buffer.read(n)

    def readable(self) -> bool:
        return self._buffer.readable()

    def readline(self, limit: int = -1) -> AnyStr:
        return self._buffer.readline(limit)

    def readlines(self, hint: int = -1) -> List[AnyStr]:
        return self._buffer.readlines(hint)

    # stream position operations

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._buffer.seek(offset, whence)

    def seekable(self) -> bool:
        return self._buffer.seekable()

    def tell(self) -> int:
        return self._buffer.tell()

    # write methods

    def writable(self) -> bool:
        self._buffer.writable()

    def write(self, s: AnyStr) -> int:
        self._buffer.write(s)

    def writelines(self, lines: List[AnyStr]) -> None:
        self._buffer.writelines(lines)

    def flush(self) -> None:
        self._buffer.flush()


class StreamedResource(Buffer):
    """"Implements Iterable and context manager"""

    def __init__(self, identifier: str, buffer: io.IOBase, chunk_size: int = 1024):
        super().__init__(buffer, chunk_size)
        self._identifier = identifier

    @property
    def identifier(self) -> str:
        return self._identifier

    # def _load_to_memory(self, file_like) -> BinaryIO:
    #     bytesIO_buffer = io.BytesIO(file_like.read())
    #     file_like.close()
    #     return bytesIO_buffer


class StreamedTextResource(StreamedResource):
    def __init__(self, identifier: str, buffer: io.IOBase, chunk_size: int = 1024, encoding: str = "utf-8"):
        text_buffer = io.TextIOWrapper(buffer, encoding=encoding)
        super().__init__(identifier, text_buffer,  chunk_size)

    @staticmethod
    def from_streamed_resouce(resource: StreamedResource, encoding: str = "utf-8") -> "StreamedTextResource":
        return StreamedTextResource(resource.identifier, resource, resource.chunk_size, encoding)
