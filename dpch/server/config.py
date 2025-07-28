from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class ClickHouseConfig(BaseModel):
    host: str
    port: int = Field(..., ge=1, le=65535)
    user: str
    password: Optional[str] = Field(default="")
    database: Optional[str] = Field(default=None)


class SchemaConfig(BaseModel):
    provider: str
    kwargs: Any


class AuthConfig(BaseModel):
    provider: str
    kwargs: Any


class ExecutorConfig(BaseModel):
    provider: str
    kwargs: Any


class AppConfig(BaseModel):
    ch_schema: SchemaConfig = Field(alias="schema")  # schema is inbuilt pydantic field.
    auth: AuthConfig
    executor: ExecutorConfig
    debug: bool = Field(default=False)


def load_config(path: str = "/etc/dpch/config.yaml") -> AppConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)
