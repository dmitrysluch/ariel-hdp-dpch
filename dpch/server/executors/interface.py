from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from dpch.common.queries.interface import QueryProto
from dpch.common.schema import Schema
from dpch.server.auth.interface import ServerSession


class ExecutorValueError(ValueError):
    pass


class ExecutorMixin(ABC):
    """
    Returns:
        tuple[numpy.ndarray, str]: A tuple containing:
            - 2D numpy array representing the untampered dataset
            - Debug information represented as a tree convertable to json
    """

    @abstractmethod
    async def execute(
        self, query: QueryProto, schema: Schema, session: ServerSession
    ) -> tuple[np.ndarray, Any]:
        pass
