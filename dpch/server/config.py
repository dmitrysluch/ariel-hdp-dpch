from pydantic import BaseModel, Field
import yaml
from functools import lru_cache
from typing import Optional
from dpch.common.column import Column


class ClickHouseConfig(BaseModel):
    host: str
    port: int = Field(..., ge=1, le=65535)
    user: str
    password: Optional[str] = Field(default="")
    database: Optional[str] = Field(default=None)


class AppConfig(BaseModel):
    clickhouse: ClickHouseConfig
    columns_schema: dict[str, list[Column]]
    debug: bool = Field(default=False)


@lru_cache
def load_config(path: str = "/etc/dpch/config.yaml") -> AppConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)
