from cachetools import TTLCache

from dpch.client.client.interface import ClientMixin
from dpch.common.api import SchemaResponse


class SchemaCache:
    def __init__(self, client: ClientMixin, ttl_seconds: int = 60):
        self.client = client
        self._cache = TTLCache(maxsize=1, ttl=ttl_seconds)

    def get_schema(self) -> SchemaResponse:
        if "schema" in self._cache:
            return self._cache["schema"]
        schema = self.client.schema()
        self._cache["schema"] = schema
        return schema
