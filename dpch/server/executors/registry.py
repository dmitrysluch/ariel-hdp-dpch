from typing import Generic, TypeVar

from dpch.common.queries.interface import QueryProto

QueryProcessorT = TypeVar("QueryProcessorT")


class QueryProcessorRegistry(Generic[QueryProcessorT]):
    def __init__(self):
        self.registry: dict[str:QueryProcessorT] = {}

    def register_query_processor(self, executor: type):
        self.registry[getattr(executor, "tp")] = executor
        return executor

    def parse_query_processor(self, query: QueryProto, **kwargs) -> QueryProcessorT:
        return self.registry[getattr(query, "tp")](**kwargs)
