import importlib
import re

from dpch.common.schema import SchemaProviderMixin
from dpch.server.config import SchemaConfig
from dpch.server.schema.fs_schema_provider import FSSchemaProvider
from dpch.server.utils import IMPORT_RE


# Allows to get schema providers from other modules.
# Useful if at some point we would like to store schema in DB and update on fly.
def get_schema_provider(conf: SchemaConfig) -> SchemaProviderMixin:
    if conf.provider == "fs":
        return FSSchemaProvider(**conf.kwargs)
    elif re.match(IMPORT_RE, conf.provider) is not None:
        *import_path, provider = conf.provider.split(":")
        try:
            provider = getattr(importlib.import_module(import_path), provider)
            if not isinstance(provider, SchemaProviderMixin):
                raise ValueError("Imported class is not schema provider")
        except (ImportError, AttributeError) as e:
            raise ValueError("Failed importing schema provider") from e
        return provider(**conf.kwargs)
    else:
        raise ValueError("Incorrect schema provider import string")
