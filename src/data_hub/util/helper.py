import os
import hashlib
import io


def is_safe_path(basedir, path, follow_symlinks=True):
    # resolves symbolic links
    if follow_symlinks:
        return os.path.realpath(path).startswith(basedir)

    return os.path.abspath(path).startswith(basedir)


def calculate_md5(byte_stream: io.BytesIO, chunk_size: int = 1024 * 1024):
    md5 = hashlib.md5()
    for chunk in iter(lambda: byte_stream.read(chunk_size), b''):
        md5.update(chunk)
    return md5.hexdigest()
