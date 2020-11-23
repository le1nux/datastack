import gzip
from data_stack.io.resources import StreamedResource, ResourceFactory


class PreprocessingHelpers:
    def get_gzip_stream(resource: StreamedResource):
        # NOTE: In contrast to tar or zip, a gzip file can only compress a single file
        buffer = gzip.GzipFile(fileobj=resource)
        new_resource = ResourceFactory.get_resource(resource.identifier, buffer)
        return new_resource

    # def extract_tar(archive_path: str):
    #     mode = 'r:gz' if archive_path.endswith(".tar.gz") else 'r'
    #     with tarfile.open(archive_path, mode) as tar:
    #         tar.extractall(os.path.dirname(archive_path))
