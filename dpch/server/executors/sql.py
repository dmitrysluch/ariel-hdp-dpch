import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


# Make an SQLQuery dataclass which consists of a query as str and list of parameters as dict
@dataclass
class SQLQuery:
    query: str
    parameters: dict[str, Any]
    rv_names: list[str]


class SQLQueryMixin(ABC):
    @abstractmethod
    def get_sql(self, query: Any, args: list[Any]) -> SQLQuery:
        """
        Returns SQL query computing this column

        Returns:
            str: SQL query
        """
        pass


COLUMN_NAME_REGEX = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
