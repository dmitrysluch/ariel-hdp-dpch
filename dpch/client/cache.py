from cachetools import TTLCache

from dpch.client.client.interface import ClientMixin
from dpch.common.schema import SchemaDataset
from dpch.common.session import Session


class Cache:
    def __init__(self, client: ClientMixin, ttl_seconds: int = 60):
        self.client = client
        self._cache = TTLCache(maxsize=1, ttl=ttl_seconds)

    def schema(self) -> SchemaDataset:
        if "schema" in self._cache:
            return self._cache["schema"]
        schema = self.client.schema().ch_schema
        self._cache["schema"] = schema
        return schema

    def session(self) -> Session:
        if "session" in self._cache:
            return self._cache["session"]
        session = self.client.session().session
        self._cache["session"] = session
        return session

    def clear_cache(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
