import os

from fastapi import Depends, HTTPException, Request

from dpch.common.schema import Schema
from dpch.server.auth.factory import get_auth_provider as get_auth_provider_impl
from dpch.server.auth.interface import AuthError, AuthProviderMixin, ServerSession
from dpch.server.config import AppConfig
from dpch.server.config import load_config as load_config_impl
from dpch.server.executors.factory import get_executor as get_executor_impl
from dpch.server.executors.interface import ExecutorMixin
from dpch.server.schema.factory import get_schema_provider as get_schema_provider_impl
from dpch.server.schema.interface import SchemaProviderMixin, SchemaValueError
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
    return get_schema_provider_impl(config.ch_schema)


# Schema MAY be updated, it is normal, and we SHOULD reload it between requests.
# Caching is left for schema_provider implementation. Yet schema MUST remain the same
# throughout request, so always set use_cache true (it is default, yet I insist on setting it explicitly).
async def get_schema(
    provider: SchemaProviderMixin = Depends(get_schema_provider, use_cache=True),
) -> Schema:
    try:
        return await provider.get_schema()
    except SchemaValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@once
def get_auth_provider(
    config=Depends(load_config, use_cache=True),
) -> AuthProviderMixin:
    return get_auth_provider_impl(config.auth)


async def get_session(
    request: Request,
    schema: Schema = Depends(get_schema, use_cache=True),
    auth_provider: AuthProviderMixin = Depends(get_auth_provider, use_cache=True),
) -> ServerSession:
    try:
        return await auth_provider.get_session(request, schema)
    except AuthError as e:
        raise HTTPException(status_code=403, detail=str(e))


async def get_session_transaction_handler(
    request: Request,
    schema: Schema = Depends(get_schema, use_cache=True),
    auth_provider: AuthProviderMixin = Depends(get_auth_provider, use_cache=True),
) -> ServerSession:
    try:
        return await auth_provider.get_transaction_handler(request, schema)
    except AuthError as e:
        raise HTTPException(status_code=403, detail=str(e))


# Please implement get_executor method in the same manner as get_auth_provider
# and get_schema_provider are written
@once
def get_executor(
    config: AppConfig = Depends(load_config, use_cache=True),
) -> ExecutorMixin:
    return get_executor_impl(config.executor)


def get_use_tlv(request: Request) -> bool:
    # Check Accept header for TLV serialization
    accept_header = request.headers.get("accept", "")
    return "application/x-tlv" in accept_header
