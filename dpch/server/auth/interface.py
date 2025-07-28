from abc import ABC, abstractmethod

from fastapi import Request

from dpch.common.schema import Schema
from dpch.common.session import Session


class ServerSession(Session):
    id: int


class AuthProviderMixin(ABC):
    @abstractmethod
    async def get_session(self, request: Request, schema: Schema) -> ServerSession:
        pass

    @abstractmethod
    async def update_session(self, session: ServerSession) -> None:
        pass
