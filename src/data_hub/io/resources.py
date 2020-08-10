from abc import ABC, abstractmethod
# from data_hub.util.logger import logger
# from data_hubStorageConnector.exception import MalformedIdentifierError
import io
from typing import BinaryIO, abstractproperty, AnyStr, List


class ResourceType:
    BINARY = "binary"
    TEXTUAL = "textual"

    # mapping = {"+bin": BINARY, "+txt": TEXTUAL}

    @classmethod
    def get_resource_type(cls, identifier: str):
        return cls.BINARY


class ResourceFactory:
    @staticmethod
    def get_resource(identifier: str, file_like_object, in_memory: bool = True, chunk_size: int = 1024) -> "StreamedResource":
        return StreamedBinaryResource(identifier, file_like_object, in_memory, chunk_size)


class IterableIF(ABC):

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError


class StreamedResource(IterableIF):
    """"Implements Iterable and context manager"""

    def __init__(self, identifier: str, buffer: io.IOBase):
        self.identifier = identifier
        self.buffer = buffer

    @abstractmethod
    def replace_buffer(self, buffer: BinaryIO, in_place: bool = False):
        raise NotImplementedError

    # ===========================CONTEXT=MANAGER===============================

    @abstractmethod
    def __enter__(self) -> "StreamedResource":
        raise NotImplementedError

    @abstractmethod
    def __exit__(self):
        raise NotImplementedError

    # ========================IO=METHOD=WRAPPERS==============================

    def close(self) -> None:
        self.buffer.close()

    def closed(self) -> bool:
        return self.buffer.closed()

    def fileno(self) -> int:
        return self.buffer.fileno()

    # read methods

    def read(self, n: int = -1) -> AnyStr:
        return self.buffer.read(n)

    def readable(self) -> bool:
        return self.buffer.readable()

    def readline(self, limit: int = -1) -> AnyStr:
        return self.buffer.readline(limit)

    def readlines(self, hint: int = -1) -> List[AnyStr]:
        return self.buffer.readlines(hint)

    # stream position operations

    def seek(self, offset: int, whence: int = 0) -> int:
        return self.buffer.seek(offset, whence)

    def seekable(self) -> bool:
        return self.buffer.seekable()

    def tell(self) -> int:
        return self.buffer.tell()

    # write methods

    def writable(self) -> bool:
        pass

    def write(self, s: AnyStr) -> int:
        pass

    def writelines(self, lines: List[AnyStr]) -> None:
        pass

    def flush(self) -> None:
        self.buffer.flush()


class StreamedBinaryResource(StreamedResource):

    def __init__(self, identifier: str, file_like: BinaryIO, copy_to_memory: bool = False, chunk_size: int = 1024):
        self.copy_to_memory = copy_to_memory
        self.chunk_size = chunk_size
        buffer: BinaryIO = self._load_to_memory(file_like) if copy_to_memory else file_like
        super().__init__(identifier, buffer)

    def __iter__(self):
        self.buffer.seek(0)
        while True:
            chunk = self.buffer.read(self.chunk_size)
            if chunk:
                yield chunk
            else:
                break

    def _load_to_memory(self, file_like) -> BinaryIO:
        bytesIO_buffer = io.BytesIO(file_like.read())
        file_like.close()
        return bytesIO_buffer

    def __enter__(self) -> "StreamedBinaryResource":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(f"Ecxception: {exc_type}")
        self.buffer.close()

    def close(self):
        self.buffer.close()

    def replace_buffer(self, buffer: BinaryIO, in_place: bool = False):
        if in_place:
            self.buffer = buffer
            return self

        else:
            return StreamedBinaryResource(self.identifier, buffer, False, self.chunk_size)
