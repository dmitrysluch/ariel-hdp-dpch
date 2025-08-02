import importlib
import re

from dpch.server.config import ExecutorConfig
from dpch.server.executors.dummy_clickhouse import DummyClickHouseExecutor
from dpch.server.executors.interface import ExecutorMixin
from dpch.server.utils import IMPORT_RE


def get_executor(conf: ExecutorConfig) -> ExecutorMixin:
    if conf.provider == "dummy_clickhouse":
        return DummyClickHouseExecutor(**conf.kwargs)
    elif re.match(IMPORT_RE, conf.provider) is not None:
        *import_path, provider = conf.provider.split(":")
        try:
            provider = getattr(importlib.import_module(import_path), provider)
            if not isinstance(provider, ExecutorMixin):
                raise ValueError("Imported class is not executor")
        except (ImportError, AttributeError) as e:
            raise ValueError("Failed importing executor") from e
        return provider(**conf.kwargs)
    else:
        raise ValueError("Incorrect executor import string")
