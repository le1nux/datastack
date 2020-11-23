from dataclasses import dataclass


@dataclass
class ResourceDefinition:
    identifier: str
    source: str
    md5_sum: str
