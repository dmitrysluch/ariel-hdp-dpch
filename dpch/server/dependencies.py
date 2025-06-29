from clickhouse_connect import get_client
from clickhouse_connect.driver.client import Client
from dpch.server.config import load_config
from fastapi import Depends

_cached_client: Client | None = None


def get_clickhouse_client(config=Depends(load_config)) -> Client:
    config = load_config()
    ch = config.clickhouse
    return get_client(
        host=ch.host,
        port=ch.port,
        username=ch.user,
        password=ch.password,
        database=ch.database,
    )
