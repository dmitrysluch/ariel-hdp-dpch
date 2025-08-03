from abc import ABC, abstractmethod

from dpch.common.api import (
    DebugQueryResponse,
    QueryRequest,
    RunQueryResponse,
    SchemaResponse,
)

# Make ClientMixin class. it is ABC class and provides methods query(which takes QueryRequest as parameter) returning RunQueryResponse or DebugQueryResponse and schema without arguments returning SchemaResponse. Don't implement these methods


class ClientMixin(ABC):
    @abstractmethod
    def query(self, request: QueryRequest) -> RunQueryResponse | DebugQueryResponse:
        pass

    @abstractmethod
    def schema(self) -> SchemaResponse:
        pass
