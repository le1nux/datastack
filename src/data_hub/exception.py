class DatasetNotFoundError(Exception):
    """Exception raised when a dataset could not be found."""
    pass


class DatasetOutOfBoundsError(Exception):
    """Exception raised when an index >= len(Dataset) is used."""
    pass


class ModelNotTrainedError(Exception):
    """Exception raised when model state is requested that is training dependent"""
    pass


class DatasetFileCorruptError(Exception):
    """Thrown when integrity checks indicate that a given file is corrupt."""
    pass


class MaliciousFilePathError(Exception):
    """Thrown when a given file path is possibly malicious, e.g., due to directory hopping."""
    pass


class ResourceNotFoundError(Exception):
    """Thrown when no resource could be found given an identifier"""
    pass


class MalformedIdentifierError(Exception):
    """Thrown when an identifier string is malformed."""

