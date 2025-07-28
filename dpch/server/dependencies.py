import os

from fastapi import Depends, Request

from dpch.common.schema import Schema
from dpch.server.auth.factory import get_auth_provider as get_auth_provider_impl
from dpch.server.auth.interface import AuthProviderMixin, ServerSession
from dpch.server.config import AppConfig
from dpch.server.config import load_config as load_config_impl
from dpch.server.executors.factory import get_executor as get_executor_impl
from dpch.server.executors.interface import ExecutorMixin
from dpch.server.schema.factory import get_schema_provider as get_schema_provider_impl
from dpch.server.schema.interface import SchemaProviderMixin
from dpch.server.utils import once


@once
def load_config() -> AppConfig:
    path = os.environ.get("DPCH_CONFIG_PATH")
    if path is None:
        path = "/etc/dpch/config.yaml"
    return load_config_impl(path)


@once
def get_schema_provider(
    config: AppConfig = Depends(load_config, use_cache=True),
) -> SchemaProviderMixin:
    return get_schema_provider_impl(config)


async def get_schema(
    provider: SchemaProviderMixin = Depends(get_schema_provider, use_cache=True),
) -> Schema:
    return await provider.get_schema()


@once
def get_auth_provider(
    config=Depends(load_config, use_cache=True),
) -> AuthProviderMixin:
    return get_auth_provider_impl(config)


async def get_session(
    request: Request,
    schema: Schema = Depends(get_schema, use_cache=True),
    auth_provider: AuthProviderMixin = Depends(get_auth_provider, use_cache=True),
) -> ServerSession:
    return await auth_provider.get_session(request, schema)


# Please implement get_executor method in the same manner as get_auth_provider
# and get_schema_provider are written
@once
def get_executor(
    config: AppConfig = Depends(load_config, use_cache=True),
) -> ExecutorMixin:
    return get_executor_impl(config)
