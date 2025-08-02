from clickhouse_connect import get_async_client
from clickhouse_connect.driver.asyncclient import AsyncClient

from dpch.server.config import ClickHouseConfig


async def get_clickhouse_client(config: ClickHouseConfig) -> AsyncClient:
    return await get_async_client(
        host=config.host,
        port=config.port,
        username=config.user,
        password=config.password,
        database=config.database,
    )
