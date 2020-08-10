import tempfile
import pytest
import shutil


@pytest.fixture
def tmp_folder_path() -> str:
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)
