import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from dpch.common.queries.interface import QueryProto
from dpch.common.schema import SchemaDataset


# Make an SQLQuery dataclass which consists of a query as str and list of parameters as dict
@dataclass
class SQLQuery:
    query: str
    parameters: dict[str, Any]
    rv_names: list[str]


class SQLQueryMixin(ABC):
    def __init__(self, query: QueryProto, ds: SchemaDataset):
        self.query = query
        self.ds = ds

    @abstractmethod
    def get_sql(self, args: list[SQLQuery]) -> SQLQuery:
        """
        Returns SQL query computing this column

        Returns:
            str: SQL query
        """
        pass


COLUMN_NAME_REGEX = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
