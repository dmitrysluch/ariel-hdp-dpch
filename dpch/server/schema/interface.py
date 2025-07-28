from abc import ABC, abstractmethod

from dpch.common.schema import Schema


class SchemaProviderMixin(ABC):
    @abstractmethod
    async def get_schema(self) -> Schema:
        pass
