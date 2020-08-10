import gzip
from data_hub.io.resources import StreamedResource


class PreprocessingHelpers:
    def get_gzip_stream(resource: StreamedResource):
        # NOTE: In contrast to tar or zip, a gzip file can only compress a single file
        buffer = gzip.GzipFile(fileobj=resource.buffer)
        new_resource = resource.replace_buffer(buffer, in_place=False)
        return new_resource

    # def extract_tar(archive_path: str):
    #     mode = 'r:gz' if archive_path.endswith(".tar.gz") else 'r'
    #     with tarfile.open(archive_path, mode) as tar:
    #         tar.extractall(os.path.dirname(archive_path))
