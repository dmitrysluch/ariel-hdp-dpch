import importlib
import re

from starlette.types import ASGIApp

from dpch.server.auth.interface import AuthProviderMixin
from dpch.server.auth.mock_auth_provider import (
    MockAuthSessionProvider,
    create_mock_auth_http_handler,
)
from dpch.server.config import AuthConfig
from dpch.server.schema.interface import SchemaProviderMixin
from dpch.server.utils import IMPORT_RE


def get_auth_http_handler(
    conf: AuthConfig, schema_provider: SchemaProviderMixin
) -> ASGIApp:
    if conf.provider == "mock":
        return create_mock_auth_http_handler(
            schema_provider=schema_provider, **conf.kwargs
        )
    elif re.match(IMPORT_RE, conf.provider) is not None:
        *import_path, http_handler, provider = conf.provider.split(":")
        try:
            module = importlib.import_module(import_path)
            http_handler = getattr(module, http_handler)
        except (ImportError, AttributeError) as e:
            raise ValueError("Failed importing auth provider") from e
        return http_handler(schema_provider=schema_provider, **conf.kwargs)
    else:
        raise ValueError("Incorrect auth provider import string")


def get_auth_provider(conf: AuthConfig) -> AuthProviderMixin:
    if conf.provider == "mock":
        return MockAuthSessionProvider(**conf.kwargs)
    elif re.match(IMPORT_RE, conf.provider) is not None:
        *import_path, http_handler, provider = conf.provider.split(":")
        try:
            module = importlib.import_module(import_path)
            provider = getattr(module, provider)
            if not isinstance(provider, AuthProviderMixin):
                raise ValueError("Imported class is not auth provider")
        except (ImportError, AttributeError) as e:
            raise ValueError("Failed importing auth provider") from e
        return provider(**conf.kwargs)
    else:
        raise ValueError("Incorrect auth provider import string")
