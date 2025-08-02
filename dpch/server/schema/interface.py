from abc import ABC, abstractmethod

from dpch.common.schema import Schema


class SchemaValueError(ValueError):
    pass


class SchemaProviderMixin(ABC):
    @abstractmethod
    async def get_schema(self) -> Schema:
        pass
